import os, sys
import math
import copy
import threading
import time 
import random
import numpy as np
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from .base import BaseScheduler
from utils.util import search_dict_list, estimate_overhead
from .solver import MIPSolver, compute_maximum_lease, compute_emergence
from utils import profiles

from runtime.rpc import scheduler_client
from runtime.rpc import scheduler_server

MAX_SEARCH_JOB = 1200
MAX_LEASE_NUM = 30 * 24 * 20 * 60 # day * hour * lease_number_in_one_hour
MIN_NODE_NUM = 10
RESOURCE_LIMIT = 1000
MAX_EXPECT_VALUE = 10

TOLERATE_RATIO = 1.0

# first way
# ACCEPTED_SLO = 1
# UNACCEPTED_SLO = 2
# BEST_EFFORT = 0

LOWEST=-1
ACCEPTED_SLO = 0
UNACCEPTED_SLO = 1
BEST_EFFORT = 2

# time 
BEST_SLOT = 240
#BEST_SLOT = 60 # for pollux


BEST_EFFORT_ALPHA = 10
# 
CHECKPOINT_OVER_HEAD = 0.1
RESUME_OVER_HEAD = 0.4

# debug
DEBUG_PLACEMENT = False
# TODO 
# 1. resample test data set
# 2. consider acceleration about optimization algorithm
# 3. select which jobs to run, we should consider optimization algorithm
# 4. check whether exists bug

MASTER_PORT = 22222
RESTART_THRESHOLD = 75

THROUGHPUTS = profiles.THROUGHPUTS

def best_func(e):
    force = 0
    if 'fake_force_guarantee' in e and e['fake_force_guarantee'] == True: force = 1
    return (force, min(e.__getitem__('queue_priority'), UNACCEPTED_SLO), e.__getitem__('emergence'), -e.__getitem__('submit_time'))
    
    queue_priority = e.__getitem__('queue_priority')
    if queue_priority == BEST_EFFORT:
        queue_priority = 0
    elif queue_priority == UNACCEPTED_SLO:
        queue_priority = BEST_EFFORT
    return (force, queue_priority, e.__getitem__('emergence'), -e.__getitem__('submit_time'))

def compute_progress(required_gpu_num, topology, job_info):
    return 1.0 
    if required_gpu_num > 8:
        return 1.0
    node_num = len(topology.node_group)
    if node_num == 1:
        return 1.0
    elif node_num == 2:
        return job_info['2']
    else:
        if '4' in job_info:
            return job_info['4']
        else:
            return job_info['2']


def lowest_throughput(model_info, job):
    if job.required_gpu_num == 1:
        return 1.0 
    job_info = model_info[job['model']['name']][str(convert_to_bin(job.required_gpu_num))]
    if job.required_gpu_num == 2:
        return job_info['2']
    return job_info['4']
        

def convert_to_bin(gpu_num):
    if gpu_num > 8 : return gpu_num 
    if gpu_num in [5, 6, 7]: return 8
    if gpu_num in [3, 4]: return 4
    return gpu_num


class TimeAwareWithLeaseScheduler(BaseScheduler):
    def __init__(self, JOBS, CLUSTER, USERS, placement, name, logger, **kwargs):
        self.PM, self.consolidatePM = placement
        super(TimeAwareWithLeaseScheduler, self).__init__(JOBS=JOBS, CLUSTER=CLUSTER, USERS=USERS, placement=self.PM, name=name, logger=logger)
        assert self.name == 'time-aware-with-lease'

        self.pending_jobs = JOBS.pending_jobs
        self.running_jobs = JOBS.running_jobs
        self.event_jobs = JOBS.job_events # collect on-going ending jobs
        self.end_jobs = list()
        self.end_events = list() 
        self.check_time_interval = kwargs.get('check_time_interval', 1)
        self.lease_term_interval = kwargs.get('lease_term_interval', 60) # 30 * 60)
        self.solve_starvation = kwargs.get('solve_starvation', 0) # supports avoiding starvation method
        self.save_dir = kwargs.get('save_dir', 'result/')
        self.aggressive = kwargs.get('aggressive', False)
        self.disable_turn_off = kwargs.get('disable_turn_off', False)
        self.disable_force_guarantee = kwargs.get('disable_force_guarantee', False)
        self.disable_noise_tolerance = kwargs.get('disable_noise_tolerance', True)
        self.noise_diff = kwargs.get('noise_diff', 0)

        # lease related 
        self.cur_lease_index = 0
        self.next_lease_time = self.lease_term_interval
        # mip_solver

        # hour time
        self.hour_metric = 60
        self.turn_off_resource_time = 0 * self.hour_metric
        self.turn_on_resource_time =  8 * self.hour_metric
        self.turn_off_interval = 8 # + 4
        self.cur_time_by_hour = 0
        self.hour_of_day = 24
        self.full_resource_state = False

        # guarantee resource
        self.guarantee_resource = 0
        self.spot_resource =  CLUSTER.check_total_gpus()
        
        # resource reduction
        total_resource_num = CLUSTER.check_total_gpus()
        self.resource_by_lease = [total_resource_num for _ in range(MAX_LEASE_NUM)]
        self.FULL_CLUSTER= CLUSTER
        self.mip_solver = MIPSolver('greedy')
        self.mip_objective = kwargs.get('mip_objective', 'random')
        self.adaptive = False
        if self.mip_objective == 'adaptive':
            self.adaptive = True


        # model info
        self.model_info = kwargs.get('model_info', None)
        self.batch_place_job_num_list = list()
        self.init_spot_table()
        self.number_of_accepted = 0
        self.latency_list =  list()
        global MAX_SEARCH_JOB
        MAX_SEARCH_JOB = MAX_SEARCH_JOB * self.lease_term_interval // 60 

        # for end-to-end experiments
        self.gpu_type = kwargs.get('gpu_type', 'A100')
        if self.gpu_type == "T4":
            throughput_path = "../scheduler/throughputs_T4/"
        else:
            throughput_path = "../scheduler/throughputs_A100/"
        for throughput_file in os.listdir(throughput_path):
            profiles.parse_throughput_file(throughput_path + throughput_file)
        self.effiency_list = list()
        self.time_list = list()
        self.job_to_be_killed = False
        self.global_lock = threading.Lock()
        self.global_ready_lock = threading.Lock()
        self.last_round_running_jobs = dict()
        self.this_round_running_jobs = dict()
        self.job_stable = dict()
        self.commands = list()
        self.gpu_allocations = [[0 for gpu in range(self.CLUSTER.num_gpu_p_node)] for _ in range(self.CLUSTER.num_node)]
        self.last_round_gpu_allocations = [[0 for gpu in range(self.CLUSTER.num_gpu_p_node)] for _ in range(self.CLUSTER.num_node)]
        self.simulation = True
        self.schedule_count = 0
        if not kwargs.get('simulation', True):
            self.simulation = False
            # RPC client to master
            self.scheduler_rpc_client = scheduler_client.SchedulerRpcClient('127.0.0.1', 6888)
            # run master rpc server in the background
            scheduler_server_port = 6890
            callbacks = {
                'ReportStable' : self.report_stable_callback,
                'ReportReady' : self.report_ready_callback,
            }
            server_thread = threading.Thread(target=scheduler_server.serve, 
                args=(scheduler_server_port, callbacks))
            server_thread.setDaemon(True)
            server_thread.start()


    def report_stable_callback(self, job_idx):
        print("received fastforward request of job", job_idx)
        receive_time = time.time()
        self.global_lock.acquire()
        if job_idx not in self.job_stable:
            print("job", job_idx, "requested fast forward before scaling")
        if self.job_stable[job_idx] != 0:
            print("unexpected request from job", job_idx, self.job_stable)
        assert job_idx in self.job_stable and self.job_stable[job_idx] == 0
        self.job_stable[job_idx] = 1
        job = search_dict_list(self.running_jobs, 'job_idx', job_idx)
        job['overhead'] = math.floor(receive_time - self.this_round_begin_time) / 60
        print("job", job_idx, "overhead", job['overhead'])
        # TODO: record overhead
        for each_job in self.running_jobs:
            if each_job['status'] != 'RUNNING':
                continue
            if each_job['job_idx'] not in self.job_stable:
                if each_job['job_idx'] in self.this_round_running_jobs:
                    each_job['overhead'] = 0
                    continue
                self.global_lock.release()
                return
            if self.job_stable[each_job['job_idx']] == 0:
                self.global_lock.release()
                return
        self.fast_forward_permission = True
        self.global_lock.release()
        print("ALL JOBS READY")

    def report_ready_callback(self, trainer_id):
        print("received report ready request of trainer_id", trainer_id)
        self.global_ready_lock.acquire()
        self.last_round_gpu_allocations[trainer_id // self.CLUSTER.num_gpu_p_node][trainer_id % self.CLUSTER.num_gpu_p_node] = 0
        for node in self.last_round_gpu_allocations:
            for gpu in node:
                if gpu == 1:
                    self.global_ready_lock.release()
                    return
        for command in self.commands:
            self.scheduler_rpc_client.schedule(command)
        self.scheduler_rpc_client.schedule('F')
        self.scheduler_rpc_client.schedule('T')
        # all jobs have been killed. no running jobs in cluster
        self.job_to_be_killed = False
        self.last_round_gpu_allocations = self.gpu_allocations
        self.global_ready_lock.release()
        

    def init_spot_table(self, ):
        self.spot_table = [self.spot_resource for _ in range(self.lease_term_interval // BEST_SLOT)]
    
    
    def place_force_job(self, job, duration, cur_time):
        in_lease_remaining_time = self.cur_lease_index * self.lease_term_interval - cur_time
        index = (self.lease_term_interval - in_lease_remaining_time) // BEST_SLOT
        occupy_block = int(math.ceil(duration * 1.0 / BEST_SLOT))
        feasible = False
        for i in range(index, len(self.spot_table)):
            for j in range(i, i+occupy_block):
                if self.spot_table[j] < job.required_gpu_num:
                    feasible = True
                if feasible:
                    job['force_start_time'] = cur_time + (i - index) * BEST_SLOT
                    break
        return feasible


    # abstract method
    def check_resource(self, **kwargs):
        resource_name = kwargs.get('resource', 'gpu')
        assert isinstance(resource_name, str)
        if 'gpu' == resource_name:
            return self.CLUSTER.check_free_gpus()
        if 'cpu' == resource_name:
            return self.CLUSTER.check_free_cpus()
    

    def opportunistic_place_jobs(self, jobs, cur_time):
        # if self.placement.name == 'local_search':
        if False: 
            remaining_gpu_num = self.CLUSTER.check_free_gpus()
            resuming_jobs = list()
            for job in jobs: 
                if remaining_gpu_num < job.required_gpu_num:
                    continue 
                resuming_jobs.append(job)
                remaining_gpu_num -= job.required_gpu_num
            self.place_jobs(resuming_jobs, cur_time)

            # for job in jobs: 
            #     if remaining_gpu_num < job.required_gpu_num:
            #         self.place_jobs(resuming_jobs, cur_time)
            #         for i_job in resuming_jobs:
            #             if i_job['status'] != 'RUNNING':
            #                 remaining_gpu_num += i_job.required_gpu_num
                     
            #     if job.required_gpu_num > 8:
            #         self.place_jobs(resuming_jobs, cur_time)
            #         self.place_jobs([job], cur_time, force_consolidation=True)
            #         remaining_gpu_num -= job.required_gpu_num
            #         for i_job in resuming_jobs + [job]:
            #             if i_job['status'] != 'RUNNING':
            #                 remaining_gpu_num += i_job.required_gpu_num
            #         resuming_jobs = list() 
            #     else:
            #         resuming_jobs.append(job)
            #         remaining_gpu_num -= job.required_gpu_num
            
        else:
            remaining_gpu_num = self.CLUSTER.check_free_gpus()
            resuming_jobs = list()
            for job in jobs: 
                if remaining_gpu_num < job.required_gpu_num:
                    continue 
                resuming_jobs.append(job)
                remaining_gpu_num -= job.required_gpu_num
            self.place_jobs(resuming_jobs, cur_time)


    # abstract method
    def place_jobs(self, jobs, cur_time, force_consolidation=False):
        if not self.placement.name.startswith('local_search'):
            self.logger.info('list job {}'.format([job.required_gpu_num for job in jobs]))
            for job in jobs:
                if self.placement.place_jobs(job):
                    job['status'] = 'RUNNING'
                    if job['start_time'] == sys.maxsize:
                        job['start_time'] = cur_time
                    # self.logger.info('pending job {} is resumed at time {}'.format(job['job_id'], cur_time))
            return 

        jobs = sorted(jobs, key=lambda e: -e.required_gpu_num)


        if force_consolidation:
            for job in jobs:
                if self.consolidatePM.place_jobs(job):
                    job['status'] = 'RUNNING'
                    if job['start_time'] == sys.maxsize:
                        job['start_time'] = cur_time
                    self.logger.info('pending job {} is resumed at time {}'.format(job['job_id'], cur_time))
            return 

        unallocated_jobs = list()
        
        for job in jobs:
            if job.required_gpu_num > 8:
                if self.consolidatePM.place_jobs(job):
                    job['status'] = 'RUNNING'
                    if job['start_time'] == sys.maxsize:
                        job['start_time'] = cur_time
                    self.logger.info('pending job {} is resumed at time {}'.format(job['job_id'], cur_time))
                    continue

                unallocated_jobs.append(job)
            else:
                unallocated_jobs.append(job)
        
        jobs = unallocated_jobs # TODO 

        if (self.placement.name == 'local_search' or self.placement.name == 'local_search_rev') and not force_consolidation:
            all_required_gpu_num = sum([job.required_gpu_num for job in jobs])
            self.batch_place_job_num_list.append((cur_time, len(jobs), all_required_gpu_num))
            self.batch_place_jobs(jobs, cur_time)
            return 

        for job in jobs:
            if not self.try_allocate_resoure(job): # TODO, make it allowed allocate across nodes
                continue
            job['status'] = 'RUNNING'
            if job['start_time'] == sys.maxsize:
                job['start_time'] = cur_time
            self.logger.info('pending job {} is resumed at time {}'.format(job['job_id'], cur_time))


    def release_job_resource(self, job, status='END'):
        if self.placement.name == 'gandiva':
            # ret = self.CLUSTER.release_gandiva_job_resource(job, status)
            raise NotImplementedError
        else:
            ret = self.CLUSTER.release_job_resource(job, status)
        return ret

    
    def batch_place_jobs(self, jobs, cur_time):
        assert self.placement.name == 'local_search' or self.placement.name == 'local_search_rev'
        remaining_gpu_num = self.CLUSTER.check_free_gpus()
        filter_jobs = list()
        for job in jobs:
            if remaining_gpu_num >= job['num_gpu']:
                remaining_gpu_num -= job['num_gpu']
                filter_jobs.append(job)
        if len(filter_jobs) == 0:
            return 
        
        assert self.placement.place_jobs(filter_jobs, self.logger) == True

        for job in filter_jobs:
            if job['topology'] is None: 
                continue 

            job['status'] = 'RUNNING'
            if job['start_time'] == sys.maxsize:
                job['start_time'] = cur_time
            self.logger.info('pending job {} is resumed at time {}'.format(job['job_id'], cur_time))
    

    # allocate resource 
    def try_allocate_resoure(self, job):
        if self.placement.name in ['random', 'consolidate', 'gandiva', 'local_search', 'consolidate_random', 'local_search_rev']:
            ret = self.placement.place_jobs(job)
        else:
            raise NotImplementedError
        return ret


    # update job status info
    def update_job_time(self, job, cur_time, name):
        if name == 'RUNNING':
            job_info = {}
            #if job.required_gpu_num <= 8 and job.required_gpu_num >= 2:
            #    job_info = self.model_info[job['model']['name']][str(convert_to_bin(job.required_gpu_num))]
            
            delta_time = int(cur_time - job['last_check_time'])
            if job['force_guarantee']:
                true_delta_time = delta_time
            else:
                true_delta_time = delta_time * compute_progress(job.required_gpu_num, job['topology'], job_info)
            
            job['total_executed_time'] = int(job['total_executed_time'] + delta_time)
            job['executed_time'] = int(job['executed_time'] + delta_time)
            job['progress'] += true_delta_time
            job['last_check_time'] = cur_time
            
        elif name == 'PENDING':
            delta_time = int(cur_time - job['last_check_time'])
            job['pending_time'] += delta_time
            job['last_check_time'] = cur_time
            if job['executed_time'] > 0:
                job['last_pending_time'] += delta_time
        else:
            raise NotImplementedError


    def opportunistic_run(self, cur_time):
        # run 1
        temp_resource = sum([job.required_gpu_num for job in self.running_jobs if job['occupy_lease'] or job['force_guarantee']])
        assert temp_resource == self.guarantee_resource
        remaining_gpu_num = self.CLUSTER.check_free_gpus()
        if remaining_gpu_num > 0:
            # opportunistic preemption
            
            self.pending_jobs.sort(key=best_func, reverse=True)
            if self.lease_term_interval > 30:
                resuming_jobs = [job for job in self.pending_jobs if job['queue_priority'] != ACCEPTED_SLO] # TODO
            else:
                resuming_jobs = [job for job in self.pending_jobs] # TODO

            self.opportunistic_place_jobs(resuming_jobs, cur_time)

            for job in resuming_jobs:
                # if self.placement.name == 'local_search':
                #     assert job['status'] == 'RUNNING'
                #     assert job['topology'] is not None
                
                if job['status'] == 'RUNNING':
                    job['last_check_time'] = cur_time
                    job['last_pending_time'] = 0
                    self.pending_jobs.remove(job)
                    self.running_jobs.append(job)
            if self.lease_term_interval <= 30:
                # try to run it 
                resuming_jobs = [job for job in self.pending_jobs if job['queue_priority'] == ACCEPTED_SLO] 
                self.opportunistic_place_jobs(resuming_jobs, cur_time)
                for job in resuming_jobs:
                    if job['status'] == 'RUNNING':
                        job['last_check_time'] = cur_time
                        job['last_pending_time'] = 0
                        self.pending_jobs.remove(job)
                        self.running_jobs.append(job)


    
    def flush_running_jobs(self, prev_time, cur_time):
        global RESTART_THRESHOLD
        restart_trainers = (self.schedule_count % RESTART_THRESHOLD == 0) and self.schedule_count != 0
        temp_resource = sum([job.required_gpu_num for job in self.running_jobs if job['occupy_lease'] or job['force_guarantee']])
        assert temp_resource == self.guarantee_resource
        need_remove_jobs = list()

        for job in self.running_jobs:
            if job['job_id'] == 2 and 'cache_solution' in job and job['cache_solution'] is not None:
                print("$$$", sum(job['cache_solution']), math.ceil(1.0 * (job['estimate_duration'] - job['progress']) / self.lease_term_interval))
                print(job['job_id'])
                print(job['occupy_lease'], job['force_guarantee'], job['status'], job['queue_priority'])
            time_diff = int(cur_time - job['last_check_time'])
            # if self.disable_noise_tolerance == False and job['access_noise_tolerance'] == False:
            #     job['access_noise_tolerance'] = True
            #     if random.random() <= 0.1:
            #         job['duration'] = time_diff + job['progress']
            #         job['early_termination'] = True

            #if 'overhead' not in job:
            #    job['overhead'] = estimate_overhead(job['num_gpu'], restart_trainers)
            if self.simulation:
                job_preempt_over_head = job['resume'] * (RESUME_OVER_HEAD + CHECKPOINT_OVER_HEAD)  * (1 + math.log2(job.required_gpu_num))
            else:
                job_preempt_over_head = job['resume'] * job['overhead']  * (1 + math.log2(job.required_gpu_num))
            if (time_diff + job['progress'] >= job['true_duration'] + job_preempt_over_head):
                # job['end_time'] = job['last_check_time'] + min(job['estimate_duration'] - job['total_executed_time'], job['duration'] + job_preempt_overhead - job['progress'])
                job['end_time'] = job['last_check_time'] + job['true_duration'] - job['progress'] + job_preempt_over_head
                self.update_job_time(job, job['end_time'], name='RUNNING')
                assert self.release_job_resource(job) == True
                job['status'] = 'END'
                self.end_jobs.append(job) 
                need_remove_jobs.append(job) # lazy remove
                if job['occupy_lease'] or job['force_guarantee']: 
                    self.guarantee_resource -= job.required_gpu_num
                    self.spot_resource += job.required_gpu_num
                self.logger.info('complete job {} at time {}'.format(job['job_id'], job['end_time']))
                # end-to-end: remove end jobs
            elif (time_diff + job['total_executed_time'] >= job['estimate_duration'] and (cur_time % self.lease_term_interval == 0 or job['occupy_lease'] == False)):
                job['queue_priority'] = UNACCEPTED_SLO
                job['cache_solution'] = None
                job['fake_force_guarantee'] = True
                job['require_mip_optimization'] = False
                self.update_job_time(job, cur_time, name='RUNNING')
            else:
                self.update_job_time(job, cur_time, name='RUNNING')

            # TODO update lease count
            if not (job['occupy_lease'] or job['force_guarantee']) and job['status'] != 'END' and job['queue_priority'] == ACCEPTED_SLO:
                still_required_lease_count = math.ceil(1.0 * (job['estimate_duration'] - job['progress']) / self.lease_term_interval)
                if still_required_lease_count != job['still_required_lease_count']:
                    assert still_required_lease_count < job['still_required_lease_count'], 'only allow less {} / {}'.format(still_required_lease_count, job['still_required_lease_count'])
                    job['still_required_lease_count'] = still_required_lease_count
                    if job['cache_solution'] is not None:
                        self.logger.info('update cache solution')
                        # for cache_iter in range(len(job['cache_solution']) - 1, -1, -1):
                        for cache_iter in range(len(job['cache_solution'])):
                            if job['cache_solution'][cache_iter] == 1:
                                job['cache_solution'][cache_iter] = 0
                                break
                        if sum(job['cache_solution']) != still_required_lease_count:
                            print("@@@", sum(job['cache_solution']), still_required_lease_count)
                            print(job['job_id'])
                            print(job['occupy_lease'], job['force_guarantee'], job['status'], job['queue_priority'])
                            
                        assert sum(job['cache_solution']) == still_required_lease_count
            
            if job['true_duration'] < job['duration'] and self.noise_diff <= 40:
                coef = 1 + (1 + math.log2(1.0 * job.required_gpu_num)) * (RESUME_OVER_HEAD + CHECKPOINT_OVER_HEAD)  / self.lease_term_interval
                progress_ratio = job['progress'] / job['duration']
                job['estimate_duration'] = int((job['duration'] + (job['true_duration'] - job['duration']) * progress_ratio)  * coef * 1.1)

        for job in need_remove_jobs:
            self.running_jobs.remove(job)

            
    def flush_pending_jobs(self, prev_time, cur_time):
        self.logger.info('how many spot resource {} {}'.format(cur_time, self.spot_resource))

        in_running_jobs = list()
        for job in self.running_jobs:
            in_running_jobs.append(job['job_id'])

        # if False: # adde above line is 16.16
        # if True: 
        if self.lease_term_interval <= 30:
            unaccepted_slo_list = list()
            force_list = list()
            in_lease_remaining_time = self.cur_lease_index * self.lease_term_interval - cur_time

            for job in self.pending_jobs:
                true_remaining_time = job['true_duration'] - job['progress']
                if job['queue_priority'] == UNACCEPTED_SLO and true_remaining_time <= in_lease_remaining_time:
                    unaccepted_slo_list.append(job)
                    job['emergence'] = -true_remaining_time * job.required_gpu_num
                if job['queue_priority'] == ACCEPTED_SLO and job['force_guarantee'] == True:
                    force_list.append(job)

            unaccepted_slo_list.sort(key=lambda e: e.__getitem__('emergence'), reverse=True)
            # unaccepted_slo_list.sort(key=lambda e: e.required_gpu_num)
            spot_resource_num = self.spot_resource - sum([job.required_gpu_num for job in force_list])
            for job in unaccepted_slo_list:
                if job.required_gpu_num <= spot_resource_num:
                    spot_resource_num -= job.required_gpu_num
                    job['force_guarantee'] = True
                    if job['submit_time'] >= (self.cur_lease_index - 1) * self.lease_term_interval:
                        job['queue_priority'] = ACCEPTED_SLO
                    # job['queue_priority'] = ACCEPTED_SLO
                    job['require_mip_optimization'] = False

                
        for job in self.pending_jobs:
            assert job['occupy_lease'] == False
            self.update_job_time(job, cur_time, name='PENDING')
            expected_remaining_time = (job.uesr_specified_deadline_time - (cur_time - job['submit_time']))
            true_remaining_time = job['estimate_duration'] - job['progress'] # job['total_executed_time']
            if self.noise_diff <= 40:
                if job['queue_priority'] != BEST_EFFORT and job['queue_priority'] != LOWEST and (job.uesr_specified_deadline_time + job['submit_time']) < cur_time + job['true_duration'] - job['progress']:
                    job['queue_priority'] = LOWEST
            else:
                if job['queue_priority'] != BEST_EFFORT and job['queue_priority'] != LOWEST and (job.uesr_specified_deadline_time + job['submit_time']) < cur_time + job['duration'] - job['progress']:
                    job['queue_priority'] = LOWEST

                
            if job['queue_priority'] == BEST_EFFORT or job['queue_priority'] == UNACCEPTED_SLO:
                job['emergence'] = -true_remaining_time * job.required_gpu_num
            else:
                job['emergence'] =  -(job.uesr_specified_deadline_time + job['submit_time'] - cur_time - job['estimate_duration'] + job['progress']) * job.required_gpu_num
        
        
        remaining_gpu_num = self.CLUSTER.check_free_gpus()
        resuming_jobs = list()
        preempt_jobs = list()
        required_gpu_num = 0
        for job in  self.pending_jobs:
            if job['force_guarantee']:
                self.logger.info('display force guarantee running {} {} {}'.format(job['job_id'], job['estimate_duration'], job['expect_time']))
                resuming_jobs.append(job)
                required_gpu_num += job.required_gpu_num

        temp_resource = 0
        for job in self.running_jobs:
            if job['occupy_lease'] or job['force_guarantee']:
                temp_resource += job.required_gpu_num

        assert temp_resource == self.guarantee_resource, '{} {}'.format(temp_resource, self.guarantee_resource)
        assert required_gpu_num <= self.spot_resource

        # preempt job to release resource
        if required_gpu_num > remaining_gpu_num:
            for job in self.running_jobs:
                if job['occupy_lease'] == False and job['force_guarantee'] == 0:
                    preempt_jobs.append(job)

            for job in preempt_jobs:
                expected_remaining_time = (job.uesr_specified_deadline_time - (cur_time - job['submit_time']))
                true_remaining_time = job['estimate_duration'] - job['progress']
                if self.noise_diff <= 40:
                    if job['queue_priority'] != BEST_EFFORT and (job.uesr_specified_deadline_time + job['submit_time']) < cur_time + job['true_duration'] - job['progress']:
                        job['queue_priority'] = LOWEST
                else:
                    if job['queue_priority'] != BEST_EFFORT and (job.uesr_specified_deadline_time + job['submit_time']) < cur_time + job['duration'] - job['progress']:
                        job['queue_priority'] = LOWEST

                
                if job['queue_priority'] in [BEST_EFFORT, UNACCEPTED_SLO]:
                    job['emergence'] = -true_remaining_time * job.required_gpu_num
                else:
                    job['emergence'] = -(job.uesr_specified_deadline_time + job['submit_time'] - cur_time - job['estimate_duration'] + job['progress']) * job.required_gpu_num
        
            # preempt_jobs.sort(key=lambda e:(e.__getitem__('queue_priority'), e.__getitem__('emergence'), -e.__getitem__('submit_time'))) # large value means emergency
            preempt_jobs.sort(key=best_func)
            for job in preempt_jobs:
                if required_gpu_num <= remaining_gpu_num: break
                assert self.release_job_resource(job) == True
                job['status'] = 'PENDING'
                job['occupy_lease'] = False
                job['last_check_time'] = cur_time
                job['last_pending_time'] = 0

                # update lease
                # still_required_lease_count = math.ceil((job['estimate_duration'] - job['total_executed_time']) / self.lease_term_interval)
                still_required_lease_count = math.ceil(1.0 * (job['estimate_duration'] - job['progress']) / self.lease_term_interval)
                if still_required_lease_count != job['still_required_lease_count']:
                    if job['cache_solution'] is not None:
                        assert still_required_lease_count < job['still_required_lease_count']
                        assert sum(job['cache_solution']) == job['still_required_lease_count']
                        for idx in range(len(job['cache_solution'])):
                            if job['cache_solution'][idx] == 1:
                                job['cache_solution'][idx] = 0
                                if sum(job['cache_solution']) == still_required_lease_count:
                                    break
                            
                    job['still_required_lease_count'] = still_required_lease_count
                
                self.running_jobs.remove(job)
                self.pending_jobs.append(job) 
                remaining_gpu_num += job.required_gpu_num

        if remaining_gpu_num < required_gpu_num:
            import pdb; pdb.set_trace()
        # run resume job
        assert remaining_gpu_num >= required_gpu_num
        self.place_jobs(resuming_jobs, cur_time) # , force_consolidation=True)
        

        for job in resuming_jobs:
            assert job['status'] == 'RUNNING'
            assert job['topology'] is not None
            if job['status'] == 'RUNNING':
                job['last_check_time'] = cur_time
                job['last_pending_time'] = 0
                self.pending_jobs.remove(job)
                self.running_jobs.append(job)
                self.guarantee_resource += job.required_gpu_num
                self.spot_resource -= job.required_gpu_num
            else:
                job['force_guarantee'] = False
        
        self.pending_jobs.sort(key=lambda e:(e.__getitem__('queue_priority'), e.__getitem__('emergence'), e.__getitem__('submit_time')), reverse=True) # emergency jobs needs to be rescheduled 
        self.opportunistic_run(cur_time=cur_time)
        
        # run 2
        temp_resource = sum([job.required_gpu_num for job in self.running_jobs if job['occupy_lease'] or job['force_guarantee']])
        assert temp_resource == self.guarantee_resource
        for job in self.running_jobs:
            if job['job_id'] not in in_running_jobs:
                job['resume'] += 1
        
        for job in self.pending_jobs:
            if job['job_id'] in in_running_jobs:
                job['preempt'] += 1
        
    
    def move_to_pending(self, job):
        ''' job gets into the system: pending or running, and finally END'''
        job['last_check_time'] = job['submit_time']
        self.pending_jobs.append(job)
        job['status'] = 'PENDING'
        job['start_time'] = sys.maxsize
        job['emergence'] = sys.maxsize
        job['occupy_lease'] = False


    # event related 
    def execute_start_job(self, start_job, cur_time):
        self.move_to_pending(start_job)
        self.logger.info('---- job[%s], gpu [%d] is added  at time[%d]' % (str(start_job['job_id']), start_job.required_gpu_num, cur_time))


    def abstract_lease_list(self, job_list):
        # 1. init
        select_soft_list = list()
        select_value_list = list() 
        select_soft_id_list = list() 
        required_resource_list = list()
        required_lease_list = list()
        maximum_lease_list = list()
        in_block_list = list()

        # 2. job abstraction

        for job in self.pending_jobs + self.running_jobs:
            if job['queue_priority'] != ACCEPTED_SLO: continue 
            if job['require_mip_optimization'] == False: continue
            required_resource = convert_to_bin(job.required_gpu_num)
            required_lease = job['still_required_lease_count']
            # maximum_lease = compute_maximum_lease(job.uesr_specified_deadline_time + job.submit_time, self.lease_term_interval, self.cur_lease_index) 
            
            if required_lease > 0:
                for vid, (expect_time, expect_value) in enumerate(zip(job['expect_time_list'], job['expect_value_list'])):
                    maximum_lease = compute_maximum_lease(expect_time + job.submit_time, self.lease_term_interval, self.cur_lease_index) 
                    required_resource_list.append(required_resource)
                    required_lease_list.append(required_lease)
                    maximum_lease_list.append(maximum_lease)
                    in_block_list.append(job)


                    if job['cache_solution'] is not None:
                        if job['cache_expect_vid'] < 0 or job['cache_expect_vid'] >= len(job['expect_time_list']):
                            import pdb; pdb.set_trace() 
                        if job['cache_expect_vid'] == vid:
                            select_soft_list.append(1)
                            if not DEBUG_PLACEMENT:
                                if maximum_lease < required_lease:
                                    import pdb; pdb.set_trace() 
                                assert maximum_lease >= required_lease, 'maximum_lease {} should exceed required_lease {}, job_id {}, required_gpu {}, duration {}'.format(maximum_lease, required_lease, job['job_id'], job.required_gpu_num, job['estimate_duration'])
                        else:
                            select_soft_list.append(0)
                    else:
                        select_soft_list.append(-1)

                    select_value_list.append(expect_value)
                    select_soft_id_list.append(vid)

        return select_soft_list, select_value_list, select_soft_id_list, required_resource_list, required_lease_list, maximum_lease_list, in_block_list
        

    def abstract_lease_with_cache_solution(self, job_list):
        # 1. init
        select_soft_list = list()
        select_value_list = list() 
        select_soft_id_list = list() 
        required_resource_list = list()
        required_lease_list = list()
        maximum_lease_list = list()
        in_block_list = list()

        no_cache_soft_list = list() 
        no_cache_value_list = list() 
        no_cache_soft_id_list = list() 
        no_cache_required_resource_list = list()
        no_cache_required_lease_list = list()
        no_cache_maximum_lease_list = list()
        no_cache_in_block_list = list()
        
        # 2. job abstraction
        
        for job in self.pending_jobs + self.running_jobs:
            if job['queue_priority'] != ACCEPTED_SLO: continue 
            if job['require_mip_optimization'] == False: continue
            required_resource = convert_to_bin(job.required_gpu_num)
            required_lease = job['still_required_lease_count']
            # maximum_lease = compute_maximum_lease(job.uesr_specified_deadline_time + job.submit_time, self.lease_term_interval, self.cur_lease_index) 
            if required_lease > 0:
                for vid, (expect_time, expect_value) in enumerate(zip(job['expect_time_list'], job['expect_value_list'])):
                    maximum_lease = compute_maximum_lease(expect_time + job.submit_time, self.lease_term_interval, self.cur_lease_index) 
                    
                    required_resource_list.append(required_resource)
                    required_lease_list.append(required_lease)
                    maximum_lease_list.append(maximum_lease)
                    in_block_list.append(job)
                    if job['cache_solution'] is not None:
                        if job['cache_expect_vid'] < 0 or job['cache_expect_vid'] >= len(job['expect_time_list']):
                            import pdb; pdb.set_trace() 

                        if job['cache_expect_vid'] == vid:
                            select_soft_list.append(1)
                            if not DEBUG_PLACEMENT:
                                if maximum_lease < required_lease:
                                    import pdb; pdb.set_trace()

                                assert maximum_lease >= required_lease, 'maximum_lease {} should exceed required_lease {}, job_id {}, required_gpu {}, duration {}'.format(maximum_lease, required_lease, job['job_id'], job.required_gpu_num, job['duration'])
                        else:
                            select_soft_list.append(0)
                    else:
                        select_soft_list.append(-1)
                    select_value_list.append(expect_value)
                    select_soft_id_list.append(vid)

        
        cache_solution =  [self.resource_by_lease[i] for i in range(max(maximum_lease_list))] if len(maximum_lease_list) > 0 else list()
        
        
        for select_value, select_soft_id, select_soft, required_resource, required_lease, maximum_lease, job  in \
            zip(select_value_list, select_soft_id_list, select_soft_list, required_resource_list, required_lease_list, maximum_lease_list, in_block_list):
            if job['cache_solution'] is None:
                no_cache_required_resource_list.append(required_resource)
                no_cache_required_lease_list.append(required_lease)
                no_cache_maximum_lease_list.append(maximum_lease)
                no_cache_in_block_list.append(job)
                no_cache_soft_list.append(select_soft)
                no_cache_value_list.append(select_value)
                no_cache_soft_id_list.append(select_soft_id)
                assert select_soft == -1
            else:
                if select_soft_id != job['cache_expect_vid']:
                    continue 

                if len(job['cache_solution']) != maximum_lease:
                    import pdb; pdb.set_trace()
                    
                for idx, occupy in enumerate(job['cache_solution']):
                    cache_solution[idx] -= int(occupy * job.required_gpu_num)
                    assert cache_solution[idx] >= 0
                    
        return no_cache_soft_list, no_cache_value_list, no_cache_soft_id_list, \
            no_cache_required_resource_list, no_cache_required_lease_list, no_cache_maximum_lease_list, no_cache_in_block_list, cache_solution
     

    def flush_jobs(self, prev_time, cur_time, status):
        if status == 'EVENT':
            self.flush_event_jobs_with_cache_solution(prev_time, cur_time)

        elif status == 'RUNNING':
            self.flush_running_jobs(prev_time, cur_time)
        
        elif status == 'PENDING':
            self.flush_pending_jobs(prev_time, cur_time)
        else:
            raise NotImplementedError
    
    
    def flush_lease_jobs(self, prev_time, cur_time):
        hour = self.cur_lease_index * self.lease_term_interval // 60 % 24
        # number_of_accepted = 0
        # for job in self.pending_jobs + self.running_jobs:
        #     if job['queue_priority'] == UNACCEPTED_SLO:
        #         number_of_accepted += 1
        if self.adaptive:
            self.mip_objective = 'maximize'
            if self.number_of_accepted > 0:
                self.mip_objective = 'minimize'
        
        
        # TODO combine IPL programming
        in_running_jobs = list()
        for job in self.running_jobs:
            in_running_jobs.append(job['job_id'])
        
        # debug = False
        # for job in self.pending_jobs + self.running_jobs:
        #     if job['job_id'] == 9571:

        #         for vid, (expect_time, expect_value) in enumerate(zip(job['expect_time_list'], job['expect_value_list'])):
        #             maximum_lease = compute_maximum_lease(expect_time + job.submit_time, self.lease_term_interval, self.cur_lease_index) 
        #             required_lease = job['still_required_lease_count']
        #             self.logger.info('debug maximum lease {} // required_lease {}'.format(maximum_lease, required_lease))
        #             if required_lease == 79:
        #                 debug = True

        # 1. clear all jobs
        for job in self.running_jobs:
            assert self.release_job_resource(job) == True 

            job['status'] = 'PENDING'
            job['occupy_lease'] = False
            job['last_check_time'] = cur_time
            job['last_pending_time'] = 0
            if job['queue_priority'] == ACCEPTED_SLO:
                # update lease
                # still_required_lease_count = math.ceil((job['estimate_duration'] - job['total_executed_time']) / self.lease_term_interval)
                still_required_lease_count = math.ceil((job['estimate_duration'] - job['progress']) / self.lease_term_interval)
                if still_required_lease_count < job['still_required_lease_count']:
                    remove_count = job['still_required_lease_count'] - still_required_lease_count 
                    job['still_required_lease_count'] = still_required_lease_count
                    if job['cache_solution'] is not None: 
                        for idx, occupy in enumerate(job['cache_solution']):
                            if job['cache_solution'][idx] == 1 and remove_count > 0:
                                job['cache_solution'][idx] = 0
                                remove_count -= 1
            
            
            self.pending_jobs.append(job)
            job['force_guarantee'] = 0 # TODO 
        
        self.running_jobs.clear()
                
        # # 2. compute prioirity 
        # for job in self.pending_jobs:
        #     expected_remaining_time = (job.uesr_specified_deadline_time - (cur_time - job['submit_time']))
        #     true_remaining_time = job['duration'] - job['total_executed_time']
        #     attianed_service = 1. * true_remaining_time / expected_remaining_time
        #     job['priority'] = attianed_service
                
        # 3. select jobs to assign lease
        if True:
            should_run_jobs = list()
            def lease_info_collection():
                # select job 
                # total_resource_num = self.CLUSTER.check_total_gpus()
                select_soft_list, select_value_list, select_soft_id_list, required_resource_list, required_lease_list, maximum_lease_list, in_block_list = \
                    self.abstract_lease_list(self.pending_jobs + self.running_jobs)

                remaining_resource_list = [self.resource_by_lease[i] for i in range(max(maximum_lease_list + [1]))]
                out_block_list = list()

                if len(required_lease_list) > MAX_SEARCH_JOB:
                    lease_info_list = [(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in zip(required_resource_list, required_lease_list, maximum_lease_list, in_block_list, select_soft_list, select_value_list, select_soft_id_list)]
                    lease_info_list.sort(key=lambda e: (-e[0], -e[2], e[3]['job_id'], e[-1]))
                    remove_count = len(required_lease_list) - MAX_SEARCH_JOB
                    
                    required_resource_list, required_lease_list, maximum_lease_list, in_block_list = list(), list(), list(), list()
                    select_soft_list, select_value_list, select_soft_id_list = list(), list(), list()
                    visit_list = list()
                    for required_source, required_lease, maximum_lease, job, select_soft, select_value, select_soft_id in lease_info_list:
                        if job['cache_solution'] is not None and (remove_count > 0 or job['job_id'] in visit_list or required_source >= RESOURCE_LIMIT):
                            if job['job_id'] not in visit_list:
                                visit_list.append(job['job_id'])
                            if job['cache_expect_vid'] == select_soft_id:
                                out_block_list.append(job)
                                for idx, occupy in enumerate(job['cache_solution']):
                                    remaining_resource_list[idx] -= int(occupy * required_source)
                            remove_count -= 1
                        else:
                            if job['job_id'] in visit_list:
                                continue 
                            required_resource_list.append(required_source)
                            required_lease_list.append(required_lease)
                            maximum_lease_list.append(maximum_lease)
                            in_block_list.append(job)
                            select_soft_list.append(select_soft)
                            select_value_list.append(select_value)
                            select_soft_id_list.append(select_soft_id)
                        
                    
                    lease_info_list = [(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in zip(required_resource_list, required_lease_list, maximum_lease_list, in_block_list, select_soft_list, select_value_list, select_soft_id_list)]
                    lease_info_list.sort(key=lambda e: (e[3]['job_id'], e[-1]))
                    required_resource_list, required_lease_list, maximum_lease_list, in_block_list = list(), list(), list(), list()
                    select_soft_list, select_value_list, select_soft_id_list = list(), list(), list()

                    for required_source, required_lease, maximum_lease, job, select_soft, select_value, select_soft_id in lease_info_list:
                        required_resource_list.append(required_source)
                        required_lease_list.append(required_lease)
                        maximum_lease_list.append(maximum_lease)
                        in_block_list.append(job)
                        select_soft_list.append(select_soft)
                        select_value_list.append(select_value)
                        select_soft_id_list.append(select_soft_id)
                        

                return select_soft_list, select_value_list, select_soft_id_list, required_resource_list, required_lease_list, maximum_lease_list, in_block_list, remaining_resource_list, out_block_list
            

            select_soft_list, select_value_list, select_soft_id_list, required_resource_list, required_lease_list, maximum_lease_list, in_block_list, remaining_resource_list, out_block_list = lease_info_collection()
            # assert len(required_resource_list) == len(self.pending_jobs), '{} / {}'.format(len(required_resource_list), len(self.pending_jobs))
            should_run_jobs = list() # TODO [job for job in self.pending_jobs + self.running_jobs if job not in need_optimize_jobs]
            
            
            if len(required_lease_list) > 0:
                self.logger.info('Job Dim {}, Lease Dim {}'.format(len(required_resource_list), max(maximum_lease_list)))
                try:
                    solution_matrix, soft_matrix = self.mip_solver.job_selection(select_soft_list, select_value_list, select_soft_id_list, required_resource_list, required_lease_list, maximum_lease_list, remaining_resource_list, self.mip_objective)

                    for idx, job in enumerate(in_block_list):
                        if soft_matrix[idx] == 0: continue 

                        if solution_matrix[idx][0] > 0:
                            should_run_jobs.append(job)

                        maximum_lease = compute_maximum_lease(job['expect_time_list'][select_soft_id_list[idx]] + job.submit_time, self.lease_term_interval, self.cur_lease_index) 
                        
                        job['cache_solution'] = solution_matrix[idx][1:]
                        job['cache_expect_vid'] = select_soft_id_list[idx]

                    self.logger.info("what in out_block_list")
                    for job in out_block_list:
                        if job['queue_priority'] == ACCEPTED_SLO:
                            if job['cache_solution'][0] > 0:
                                should_run_jobs.append(job)
                                self.logger.info("run should_run_jobs")
                            job['cache_solution'] = job['cache_solution'][1:]
                except:
                    self.logger.info('walk fast forward way')
                    should_run_jobs = list()
                    soft_list, value_list, soft_id_list, required_resource_list, required_lease_list, maximum_lease_list, in_block_list, global_cache_solution = \
                        self.abstract_lease_with_cache_solution(self.pending_jobs + self.running_jobs)
                    
                    feasible, solution_matrix, soft_matrix = self.mip_solver.batch_fast_job_selection(soft_list, value_list, soft_id_list, required_resource_list, \
                                    required_lease_list, maximum_lease_list, in_block_list, global_cache_solution, self.resource_by_lease)

                    if not feasible:
                        required_resource_list, required_lease_list, maximum_lease_list, in_block_list, remaining_resource_list, out_block_list = lease_info_collection()
                        solution_matrix, soft_matrix = self.mip_solver.job_selection(required_resource_list, required_lease_list, maximum_lease_list, remaining_resource_list, self.mip_objective, max_seconds=60)
                        if solution_matrix[0][0] is None:
                            import pdb; pdb.set_trace()
                        for idx, job in enumerate(in_block_list):
                            if soft_matrix[idx] == 0: continue 
                            if solution_matrix[idx][0] > 0:
                                should_run_jobs.append(job)
                            job['cache_solution'] = solution_matrix[idx][1:]
                            job['cache_expect_vid'] = soft_id_list[idx]
                        for idx, job in enumerate(in_block_list):
                            if job['cache_expect_vid'] == -1 or job['cache_expect_vid'] >= len(job['expect_value_list']):
                                import pdb; pdb.set_trace() 
                        for job in out_block_list:
                            if job['queue_priority'] == ACCEPTED_SLO:
                                if job['cache_solution'][0] > 0:
                                    should_run_jobs.append(job)
                                job['cache_solution'] = job['cache_solution'][1:]
                        
                        self.logger.info('retry once again job selection')
                    else:
                        for idx, job in enumerate(in_block_list):
                            if soft_matrix[idx] == 1:
                                job['cache_solution'] = solution_matrix[idx]
                                job['cache_expect_vid'] = soft_id_list[idx]
                                
                        for job in self.pending_jobs + self.running_jobs:
                            if job['queue_priority'] == ACCEPTED_SLO:
                                if job['cache_solution'] is None:
                                    job['queue_priority'] = UNACCEPTED_SLO
                                    continue 

                                if job['cache_solution'][0] > 0:
                                    should_run_jobs.append(job)
                                job['cache_solution'] = job['cache_solution'][1:]
                        self.logger.info('fast job selection')
            else:
                for job in out_block_list:
                    if job['queue_priority'] == ACCEPTED_SLO:
                        if job['cache_solution'][0] > 0:
                            should_run_jobs.append(job)
                        job['cache_solution'] = job['cache_solution'][1:]

            self.guarantee_resource = sum([job.required_gpu_num for job in should_run_jobs])
            self.spot_resource = self.CLUSTER.check_total_gpus() - self.guarantee_resource

            self.logger.info('guarantee resource is {}'.format(self.guarantee_resource))
            self.logger.info('should run jobs {}'.format([job['job_id'] for job in should_run_jobs]))

        else:
            self.pending_jobs.sort(key=lambda e: (e.__getitem__('queue_priority'), e.__getitem__('emergence'), e.__getitem__('submit_time')), reverse=True)
            self.pending_jobs.sort(key=best_func, reverse=True)
            should_run_jobs = list()
            total_gpus = self.CLUSTER.check_total_gpus()
            for job in self.pending_jobs:
                if job.required_gpu_num <= total_gpus:
                    total_gpus -= job.required_gpu_num 
                    should_run_jobs.append(job)
        

        # 4. allocate resource
        self.place_jobs(should_run_jobs, cur_time)

        # 5. update lease number
        for job in should_run_jobs:
            if DEBUG_PLACEMENT:
                if job['status'] != 'RUNNING':
                    self.guarantee_resource -= job.required_gpu_num
                    continue 
            assert job['status'] == 'RUNNING'
            job['occupy_lease'] = True
            job['still_required_lease_count'] -= 1
            self.running_jobs.append(job)
            if job not in self.pending_jobs:
                import pdb; pdb.set_trace()
            self.pending_jobs.remove(job)
            # still_required_lease_count = math.ceil((job['duration'] - job['total_executed_time']) / self.lease_term_interval)
            # print(still_required_lease_count, job['still_required_lease_count'], job['job_id'], job['submit_time'])
            # assert still_required_lease_count - 1 == job['still_required_lease_count'], '{}'.format(job['duration'] - job['total_executed_time'])
        
            
        # global increase lease count
        self.cur_lease_index += 1
        for job in self.running_jobs:
            if job['job_id'] not in in_running_jobs:
                job['resume'] += 1
        
        for job in self.pending_jobs:
            if job['job_id'] in in_running_jobs:
                job['preempt'] += 1

        self.logger.info('number of not accepted jobs {} at hour {}, spot_resource {}'.format(self.number_of_accepted, hour, self.spot_resource))
        self.number_of_accepted = 0
        for job in self.pending_jobs + self.running_jobs:
            if job['queue_priority'] == UNACCEPTED_SLO:
                self.number_of_accepted += 1
            

    def finish_all_jobs(self, ):
        return len(self.running_jobs) + len(self.event_jobs) + len(self.pending_jobs) == 0

    
    def schedule_summary(self, ):
        if not self.disable_turn_off:
            self.name = self.name + '-aggressive' if self.aggressive else self.name + '-conserative'
            
        self.JOBS.job_list = [job for job in self.JOBS.job_list if job['status'] == 'END']
        for job in self.JOBS.job_list:
            job['gt_miss_ddl'] = 0 if job['gt_expect_time'] + job['submit_time'] >= job['end_time'] else 1


            if 'best_effort' in job and job['best_effort'] == 1: 
                job['gt_miss_ddl'] = 0
                job['miss_ddl'] = 0
                
            if 'best_effort' not in job or job['best_effort'] == 0:
                job['miss_ddl'] = 1
                for vid, (expect_time, expect_value) in enumerate(zip(job['expect_time_list'], job['expect_value_list'])):
                    if expect_time + job['submit_time'] >= job['end_time']:
                        job['miss_ddl'] = 1 - expect_value * 1.0 / MAX_EXPECT_VALUE
                        break
                
                job['expect_time'] = job.uesr_specified_deadline_time
                # job['miss_ddl'] = 0 if job['expect_time'] + job['submit_time'] >= job['end_time'] else 1
                if job['miss_ddl']: self.logger.info('job id {}, expect time {}, submit time {}, end time {}'.format(job['job_id'], job['expect_time'], job['submit_time'], job['end_time']))
            job['expect_time_list'] =  '-'.join([str(item) for item in job['expect_time_list']])
            job['expect_value_list'] = '-'.join([str(item) for item in job['expect_value_list']])


        assert all([job['status'] == 'END' for job in self.JOBS.job_list])
        attribute_list = ['job_id', 'pending_time', 'total_executed_time', 'progress', 'submit_time', 'end_time', 'preempt', 'resume',  'num_gpu', 'expect_time', 'expect_time_list', 'expect_value_list', 'miss_ddl', 'gt_miss_ddl', 'gt_expect_time', 'best_effort', 'duration', 'queue_priority']
        with open(os.path.join(self.save_dir, self.name + '.csv'), 'w') as f:
            print(",".join(attribute_list), file=f)
            for job in self.JOBS.job_list:
                for attribute in attribute_list:
                    print("{}".format(job[attribute]), file=f) if attribute == attribute_list[-1] else print("{}".format(job[attribute]), end=',', file=f)
        
        attribute_list = ['time','full_resource', 'free_resource', 'pending_num', 'running_num', 'submit_num', 'ce']
        with open(os.path.join(self.save_dir, self.name + '_resource.csv'), 'w') as f:
            print(",".join(attribute_list), file=f)
            for cur_time, full_resource, free_resource, pending_num, running_num, submit_num, efficiency in zip(self.time_list, self.full_resource_list, self.free_resource_list, self.pending_job_num_list, self.running_job_num_list, self.submit_job_num_list, self.effiency_list):
                print('{},{},{},{},{},{},{}'.format(cur_time,full_resource, free_resource, pending_num, running_num, submit_num, efficiency), file=f)
        if len(self.batch_place_job_num_list) > 0:
            self.batch_place_job_num_list.sort()
            with open(os.path.join(self.save_dir, self.name + '_batch_placement_info.csv'), 'w') as f:
                print('time,job_num,gpu_num', file=f)
                for cur_time, job_num, gpu_num in self.batch_place_job_num_list:
                    print('{},{},{}'.format(cur_time, job_num, gpu_num), file=f)
        filename = os.path.join(self.save_dir, self.name + 'latency_info.npy')
        np.save(filename, self.latency_list)

    def get_efficiency(self, job_dict):
        global_batch_size = str(job_dict['batch_size'])
        job_dict['model']['name'] = job_dict['model_name']
        num = str(job_dict['num_gpu'])
        tpt = float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][num])
        tpt_1 = float(THROUGHPUTS[job_dict['model']['name']][global_batch_size]['1'])
        return tpt / tpt_1


    def resource_summary(self, cur_time):
        self.full_resource_list.append(self.CLUSTER.check_total_gpus())
        self.free_resource_list.append(self.CLUSTER.check_free_gpus())
        self.pending_job_num_list.append(len(self.pending_jobs))
        self.running_job_num_list.append(len(self.running_jobs))
        efficiency = 0
        for job in self.running_jobs:
            if job['status'] != 'RUNNING':
                continue
            if job['num_gpu'] == 0 or job['placements'] is None:
                continue
            efficiency += self.get_efficiency(job)
        efficiency /= self.CLUSTER.num_gpu
        self.effiency_list.append(efficiency)
        self.time_list.append(cur_time)
        # TODO: record log.
        if self.simulation:
            return
        global MASTER_PORT, RESTART_THRESHOLD
        self.global_lock.acquire()
        self.this_round_running_jobs = dict()
        # step 1: judge if there is need to reschedule (change in placement, job finish, new job)
        return_flag = True # default: do not need to reschedule
        for job in self.running_jobs:
            if job['status'] != 'RUNNING':
                continue
            if job['num_gpu'] == 0 or job['placements'] is None:
                if job['job_idx'] in last_round_running_jobs:
                    return_flag = False
                    break
                continue
            if job['job_idx'] not in self.last_round_running_jobs:
                return_flag = False
                break
        for job_idx in self.last_round_running_jobs:
            if search_dict_list(self.running_jobs, 'job_idx', job_idx) is None:
                return_flag = False
                break
        if return_flag:
            self.fast_forward_permission = True
            self.global_lock.release()
            return

        restart_trainers = (self.schedule_count % RESTART_THRESHOLD == 0) and self.schedule_count != 0
        # restart if jobs didn't restart successfully last round
        #if not restart_trainers:
        #    restart_trainers = self.job_to_be_killed
        if restart_trainers:
            self.scheduler_rpc_client.schedule('RE')

        self.job_stable = dict()
        self.commands = []
        cmd = 'R'
        actual_time = math.ceil(time.time())
        self.gpu_allocations = [[0 for gpu in range(self.CLUSTER.num_gpu_p_node)] for _ in range(self.CLUSTER.num_node)]
        # step 2: record all commands to run
        for job in self.running_jobs:
            if job['num_gpu'] >= self.CLUSTER.num_gpu_p_node:
                for i in range(len(job['topology'].node_group)):
                    compressed_gpu_list = 0
                    for j in job['topology'].gpu_list:
                        if j.node_id() != job['topology'].node_group[i].node_id:
                            continue
                        self.gpu_allocations[job['topology'].node_group[i].node_id][j.id] = 1
                        compressed_gpu_list += (1 << j.id)
                    MASTER_PORT += 1
                    command = ' '.join([cmd, str(job['topology'].node_group[i].node_id), job['model_name'], 
                        str(job['batch_size']), str(job['job_idx']), str(min(job['num_gpu'], self.CLUSTER.num_gpu_p_node)), str(len(job['topology'].node_group)), 
                        str(i), '127.0.0.1', str(MASTER_PORT), str(compressed_gpu_list), str(1000), str(actual_time)]) # use 1000 iterations as a workaround.
                    print(command)

                    if job['job_idx'] not in self.this_round_running_jobs:
                        self.this_round_running_jobs[job['job_idx']] = {'worker_id':[]}
                    self.this_round_running_jobs[job['job_idx']]['worker_id'].append(job['topology'].node_group[i].node_id)

                    if job['job_idx'] not in self.job_stable:
                        self.job_stable[job['job_idx']] = 0
                    self.fast_forward_permission = False
                    self.commands.append(command)
            else:
                node_id = job['topology'].node_group[0].node_id
                allocated_gpu = 0
                compressed_gpu_list = 0
                for i in range(len(self.gpu_allocations[node_id])):
                    if self.gpu_allocations[node_id][i] == 1:
                        continue
                    allocated_gpu += 1
                    self.gpu_allocations[node_id][i] = 1
                    compressed_gpu_list += (1 << i)
                    if allocated_gpu == job['num_gpu']:
                        break
                MASTER_PORT += 1
                command = ' '.join([cmd, str(node_id), job['model_name'], str(job['batch_size']), str(job['job_idx']), 
                str(min(job['num_gpu'], self.CLUSTER.num_gpu_p_node)), str(len(job['topology'].node_group)), '0', '127.0.0.1', str(MASTER_PORT), str(compressed_gpu_list), 
                str(1000), str(actual_time)]) # use 1000 iterations as a workaround.
                print(command)

                tmp_dict = dict()
                tmp_dict['worker_id'] = [node_id]
                tmp_dict['num_gpu'] = job['num_gpu']
                tmp_dict['compressed_gpu_list'] = compressed_gpu_list
                self.this_round_running_jobs[job['job_idx']] = tmp_dict
                if job['job_idx'] not in self.job_stable:
                    self.job_stable[job['job_idx']] = 0
                self.fast_forward_permission = False
                self.commands.append(command)


        # step 3: send commands to kill all jobs and run all jobs
        self.global_ready_lock.acquire()
        if not restart_trainers:
            for job in self.last_round_running_jobs:
                for each_worker in self.last_round_running_jobs[job]['worker_id']:
                    command = 'K ' + str(each_worker) + ' ' + str(job)
                    self.scheduler_rpc_client.schedule(command)
                    self.job_to_be_killed = True
        else:
            self.job_to_be_killed = False
    
        if not self.job_to_be_killed:
            # run all commands
            for command in self.commands:
                self.scheduler_rpc_client.schedule(command)
            self.scheduler_rpc_client.schedule('F')
            self.scheduler_rpc_client.schedule('T')
            self.last_round_gpu_allocations = self.gpu_allocations
        self.fast_forward_permission = (len(self.commands) == 0)
        self.global_ready_lock.release()

        self.last_round_running_jobs = self.this_round_running_jobs
        self.schedule_count += 1
        self.global_lock.release()
    

    def turn_off_resource(self, ):
        turn_off_interval =  self.turn_off_interval # (self.turn_on_resource_time - self.turn_off_resource_time) // self.lease_term_interval
        total_resource_num = self.FULL_CLUSTER.check_total_gpus()
        resource_list = [total_resource_num for _ in range(turn_off_interval)]
        self.logger.info('total_resource_num -- {}'.format(total_resource_num))
        for job in self.pending_jobs + self.running_jobs:
            cache_solution = job['cache_solution']
            if cache_solution is not None:
                if self.aggressive:
                    for i in range(min(len(cache_solution), turn_off_interval)):
                        resource_list[i] -= cache_solution[i] * job.required_gpu_num 
                else:
                    if sum(cache_solution) >= turn_off_interval:
                        for i in range(turn_off_interval):
                            resource_list[i] -= job.required_gpu_num
                    else:
                        for i in range(min(len(cache_solution), turn_off_interval)):
                            resource_list[i] -= cache_solution[i] * job.required_gpu_num
            else:
                maximum_lease = compute_maximum_lease(job.uesr_specified_deadline_time + job.submit_time, self.lease_term_interval, self.cur_lease_index) 
                for i in range(min(turn_off_interval, maximum_lease)):
                    resource_list[i] -= job.required_gpu_num
        if min(resource_list) <= 0: return  
        
        remove_resource_num = min(min(resource_list), total_resource_num - self.CLUSTER.num_gpu_p_node * MIN_NODE_NUM)  # TODO 
        
        for idx, remove_resource_num in enumerate(resource_list):
            remove_node_num = min(remove_resource_num // self.CLUSTER.num_gpu_p_node, total_resource_num // self.CLUSTER.num_gpu_p_node - MIN_NODE_NUM)
            if remove_node_num > 0:
                assert self.resource_by_lease[idx] == total_resource_num
                self.resource_by_lease[idx] = total_resource_num - remove_node_num * self.CLUSTER.num_gpu_p_node
        self.full_resource_state = False
        self.logger.info('resource information -- {}'.format(self.resource_by_lease[:10]))


    def turn_on_resource(self, ):
        for switch in self.FULL_CLUSTER.switch_list:
            for node in switch.node_list:
                node.enable_resource = True
        self.full_resource_state = True
    
    def turn_off_resource_by_number(self, keep_resource_num):
        for switch in self.FULL_CLUSTER.switch_list:
            for node in switch.node_list:
                node.enable_resource = True
        total_resource_num = self.FULL_CLUSTER.check_total_gpus()
        remove_resource_num = total_resource_num - keep_resource_num
        for switch in self.FULL_CLUSTER.switch_list:
            for node in switch.node_list:
                total_resource_num = node.check_total_gpus()
                if remove_resource_num >= total_resource_num:
                    remove_resource_num -= total_resource_num
                    node.enable_resource = False
        assert remove_resource_num == 0, 'remove all resource'
        assert self.CLUSTER.check_total_gpus() == self.resource_by_lease[0], '{} / {} resource shoud same'.format(self.CLUSTER.check_total_gpus(), self.resource_by_lease[0])

    def filling_jobs(self, prev_time, cur_time):
        free_gpu_num = self.CLUSTER.check_free_gpus()
        if free_gpu_num > 0:
            self.pending_jobs.sort(key=lambda x: (x.__getitem__('queue_priority'), x.__getitem__('emergence')))
            should_run_jobs = list()
            for job in self.pending_jobs:
                if job.required_gpu_num <= free_gpu_num:
                    free_gpu_num -= job.required_gpu_num
                    should_run_jobs.append(job)

            self.place_jobs(should_run_jobs, cur_time=cur_time)
            should_remove_jobs = list()
            for job in should_run_jobs:
                if job['status'] == 'RUNNING':
                    should_remove_jobs.append(job)
                
            for job in should_remove_jobs:
                self.running_jobs.append(job)
                self.pending_jobs.remove(job)


    def run(self, ):
        cur_time = 0
        while not self.finish_all_jobs():
            prev_time = max(0, cur_time - self.check_time_interval)
            if self.disable_turn_off == False:
                if cur_time % (self.hour_metric * self.hour_of_day) == self.turn_off_resource_time:
                    self.turn_off_resource()
                
                if cur_time % (self.hour_metric * self.hour_of_day) == self.turn_on_resource_time:
                    self.turn_on_resource()

                if cur_time % self.lease_term_interval == 0 and not self.full_resource_state:
                    self.turn_off_resource_by_number(self.resource_by_lease[0])

            # self.logger.info('0'); self.debug_check(cur_time)
            # 1. run jobs and release resource from ending jobs
            self.flush_jobs(prev_time, cur_time, status='RUNNING')
            
            

             # 3. allocate release for resource
            if cur_time % self.lease_term_interval == 0:
                if not self.full_resource_state and self.disable_turn_off == False:
                    self.turn_off_resource_by_number(self.resource_by_lease[0])
                
                start_time = time.time()
                self.flush_lease_jobs(prev_time, cur_time)
                self.latency_list.append(time.time() - start_time)
                self.next_lease_time = self.lease_term_interval + self.check_time_interval
                self.resource_by_lease = self.resource_by_lease[1:]

            # self.logger.info('1'); self.debug_check(cur_time)
            # 2. receive the new coming jobs
            if cur_time % BEST_SLOT == 0:
                self.flush_jobs(cur_time - BEST_SLOT, cur_time, status='EVENT') # changed order
            
            
            # self.logger.info('3'); self.debug_check(cur_time)

            # 4. opportunistic allocate resource for pending jobs
            if cur_time % BEST_SLOT == 0:
                self.flush_jobs(cur_time, cur_time, status='PENDING')
            # self.logger.info('4'); self.debug_check(cur_time)
            if cur_time % BEST_EFFORT != 0:
                self.filling_jobs(prev_time, cur_time)

            self.this_round_begin_time = time.time()
            self.resource_summary(cur_time)
            if not self.simulation:
                old_cur_time = copy.deepcopy(cur_time)
                while (not self.fast_forward_permission) and (cur_time < old_cur_time + 60):
                    time.sleep(1)
                    cur_time += 1
                if not self.fast_forward_permission:
                    #update_overhead() # todo: overhead
                    print("ATTENTION! not all jobs ready")

                # 5. update time info
                cur_time = max(old_cur_time + self.check_time_interval, cur_time)
            else:
                # 5. update time info
                cur_time += self.check_time_interval


            self.next_lease_time -= self.check_time_interval
            
        self.schedule_summary()
    
    
    def best_effort_job(self, job):
        return 'best_effort' in job and job['best_effort'] == 1

    def process_unaccepted_jobs(self, job, required_resource_list, required_lease_list, maximum_lease_list, soft_list, value_list, soft_id_list, in_block_list, cache_solution_length, global_cache_solution):
        if not self.best_effort_job(job):
            infeasible_lease = required_lease_list[-1]
            if required_lease_list[-1] > maximum_lease_list[-1]: maximum_lease_list[-1] = required_lease_list[-1]
            left = maximum_lease_list[-1]
            right = max(max(maximum_lease_list), len(global_cache_solution)) + job['still_required_lease_count'] + 1
            maximum_lease_list[-1] = right 
            feasible, existing_solution = self.mip_solver.batch_fast_check_if_packable(soft_list, value_list, soft_id_list, required_resource_list, \
                    required_lease_list, maximum_lease_list, in_block_list,  global_cache_solution, self.resource_by_lease)

            while left < right:  # TODO, meet in the middle
                maximum_lease_list[-1] = left +  (right - left ) // 2
                feasible, existing_solution = self.mip_solver.batch_fast_check_if_packable(soft_list, value_list, soft_id_list, required_resource_list, \
                        required_lease_list, maximum_lease_list, in_block_list, global_cache_solution, self.resource_by_lease)

                if feasible:
                    right = maximum_lease_list[-1]
                else:
                    left = maximum_lease_list[-1] + 1
                maximum_lease_list[-1] = right 
        

            self.logger.info('no existing solution and malicious user behavior, job_id {} required_lease {}, maximum_lease {}, infeasible_lease {}'.format(job['job_id'], required_lease_list[-1], maximum_lease_list[-1], infeasible_lease))
            self.logger.info('job info {}'.format(job))
            uesr_specified_deadline_time = (self.lease_term_interval * (maximum_lease_list[-1] + self.cur_lease_index)) - job['submit_time'] 
            if uesr_specified_deadline_time <= int(job['expect_time_list'][-1] * TOLERATE_RATIO): 
                return True
            elif uesr_specified_deadline_time - job['expect_time_list'][-1]  <= (TOLERATE_RATIO - 1) * 60:
                job['expect_time_list'][-1] = uesr_specified_deadline_time
                return True
            job['queue_priority'] = UNACCEPTED_SLO

        required_resource_list = required_resource_list[:cache_solution_length]
        required_lease_list = required_lease_list[:cache_solution_length]
        maximum_lease_list = maximum_lease_list[:cache_solution_length]
        soft_list = soft_list[:cache_solution_length]
        value_list = value_list[:cache_solution_length]
        soft_id_list = soft_id_list[:cache_solution_length]
        in_block_list = in_block_list[:cache_solution_length]
        return False


    def flush_event_jobs_with_cache_solution(self, prev_time, cur_time):
        event_list = list()
        for event in self.event_jobs:
            if event['time'] <= cur_time:
                assert event['time'] >= prev_time
                event_list.append(event)
        
        if len(event_list) == 0:
            self.submit_job_num_list.append(0)
            return 
        

        total_resource_num = self.CLUSTER.check_total_gpus()
        runnable_jobs = self.pending_jobs + self.running_jobs

        soft_list, value_list, soft_id_list, required_resource_list, required_lease_list, maximum_lease_list, in_block_list, global_cache_solution = \
            self.abstract_lease_with_cache_solution(runnable_jobs)
        feasible, existing_solution = self.mip_solver.batch_fast_check_if_packable(soft_list, value_list, soft_id_list, required_resource_list, \
                                    required_lease_list, maximum_lease_list, in_block_list, global_cache_solution, total_resource_num)
        if not feasible:
            self.logger.info('double check once')
            # TODO 
            solution_matrix, soft_matrix = self.mip_solver.job_selection(soft_list, value_list, soft_id_list, required_resource_list, required_lease_list, maximum_lease_list, global_cache_solution, objective=self.mip_objective)
            if solution_matrix[0][0] is None:
                import pdb; pdb.set_trace()

            assert solution_matrix[0][0] is not None, 'should exisiting solution'
            for idx, job in enumerate(in_block_list):
                if soft_matrix[idx] == 1:
                    job['cache_solution'] = solution_matrix[idx]
                    job['cache_expect_vid'] = soft_id_list[idx]


            soft_list, value_list, soft_id_list, required_resource_list, required_lease_list, maximum_lease_list, in_block_list, global_cache_solution = \
                self.abstract_lease_with_cache_solution(runnable_jobs)
            feasible, existing_solution = self.mip_solver.batch_fast_check_if_packable(soft_list, value_list, soft_id_list, required_resource_list, \
                                        required_lease_list, maximum_lease_list, in_block_list, global_cache_solution, total_resource_num)


        assert feasible == True, 'should existing feasible solution at time {}'.format(cur_time)
        # check whether new coming jobs are feasible
        in_lease_remaining_time = self.cur_lease_index * self.lease_term_interval - cur_time
        event_guarantee_resource = 0

        submit_job_num = 0
        for event in event_list:
            for start_job in event['start_jobs']:
                if self.disable_noise_tolerance == False:
                    start_job['true_duration'] = start_job['duration']
                    scale = -1
                    while scale >= 2 or scale <= 0.5:
                        scale = (1+np.random.normal(0, self.noise_diff/100))
                        if int(start_job['duration'] * scale)  > start_job['expect_time']:
                            continue 
                    self.logger.info('random scale == {}'.format(scale))
                    start_job['duration'] = int(start_job['duration'] * scale)
                else:
                    start_job['true_duration'] = start_job['duration']


                cache_solution_length = len(soft_list)

                submit_job_num += 1
                if 'best_effort' not in start_job: start_job['best_effort'] = 0
                start_job['queue_priority'] = ACCEPTED_SLO
                start_job['gt_expect_time'] = start_job.uesr_specified_deadline_time
                self.logger.info('starting add job, solution info {}'.format(( existing_solution is not None and len(existing_solution) > 0)))
                coef = 1 + (1 + math.log2(1.0 * start_job.required_gpu_num)) * (RESUME_OVER_HEAD+CHECKPOINT_OVER_HEAD)  / self.lease_term_interval
                start_job['estimate_duration'] = int(start_job['duration'] * coef * 1.1)
                start_job['emergence'] = sys.maxsize
                start_job['obtain_lease_count'] = 0
                start_job['still_required_lease_count'] = math.ceil(start_job['estimate_duration'] / self.lease_term_interval)
                start_job['cache_expect_vid'] = -1

                for vid, (expect_time, expect_value) in enumerate(zip(start_job['expect_time_list'], start_job['expect_value_list'])):
                    maximum_lease = compute_maximum_lease(expect_time + start_job.submit_time, self.lease_term_interval, self.cur_lease_index) 
                    required_resource_list.append(convert_to_bin(start_job.required_gpu_num))
                    required_lease_list.append(start_job['still_required_lease_count'])
                    maximum_lease_list.append(maximum_lease)
                    soft_list.append(-1)
                    value_list.append(expect_value)
                    soft_id_list.append(vid)
                    in_block_list.append(start_job)

                start_job['require_mip_optimization'] = True
                start_job['force_guarantee'] = 0
                
                if start_job['best_effort'] == 1:
                    start_job['require_mip_optimization'] = 0
                    start_job['force_guarantee'] = 0
                    start_job['queue_priority'] = BEST_EFFORT

                if start_job['require_mip_optimization']:
                    if True: 
                        if required_lease_list[-1] <= maximum_lease_list[-1]:
                            feasible, existing_solution = self.mip_solver.batch_fast_check_if_packable(soft_list, value_list, soft_id_list, required_resource_list, \
                                    required_lease_list, maximum_lease_list, in_block_list, global_cache_solution, self.resource_by_lease)
                            self.logger.info('pass_fast_check {}, existing solution info {}'.format(feasible, ( existing_solution is not None and len(existing_solution) > 0)))
                        else:
                            feasible, existing_solution = False, None
                        
                        accepted = False
                        if not feasible or start_job['best_effort'] == 1 : # and self.disable_force_guarantee == False:  
                            accepted = self.process_unaccepted_jobs(start_job, required_resource_list, required_lease_list, maximum_lease_list, soft_list, value_list, soft_id_list, in_block_list, cache_solution_length, global_cache_solution)

                        if feasible or accepted:           
                            start_job['server_specified_deadline_lease_count'] = maximum_lease_list[-1]
                            start_job.uesr_specified_deadline_time = self.lease_term_interval * (maximum_lease_list[-1] + self.cur_lease_index) - start_job['submit_time']
                            if start_job['gt_expect_time'] < start_job.uesr_specified_deadline_time:
                                pass
                            feasible, existing_solution = self.mip_solver.batch_fast_check_if_packable(soft_list, value_list, soft_id_list, required_resource_list, \
                                            required_lease_list, maximum_lease_list, in_block_list, global_cache_solution, self.resource_by_lease)
                            if not feasible:
                                import pdb; pdb.set_trace()
                        elif start_job['force_guarantee'] == 1:
                            start_job['force_guarantee'] = 0
                            event_guarantee_resource -= convert_to_bin(start_job.required_gpu_num)

                else:
                    required_resource_list = required_resource_list[:cache_solution_length]
                    required_lease_list = required_lease_list[:cache_solution_length]
                    maximum_lease_list = maximum_lease_list[:cache_solution_length]
                    soft_list = soft_list[:cache_solution_length]
                    value_list = value_list[:cache_solution_length]
                    soft_id_list = soft_id_list[:cache_solution_length]
                    in_block_list = in_block_list[:cache_solution_length]


                
                start_job['occupy_lease'] = False
                self.execute_start_job(start_job, cur_time)

                start_job['last_check_time'] = cur_time
                start_job['pending_time'] = cur_time - start_job['submit_time']
                if 'cache_solution' not in start_job: start_job['cache_solution'] = None
                start_job['access_noise_tolerance'] = False
                start_job['early_termination'] = False
                if start_job['queue_priority'] == UNACCEPTED_SLO:
                    self.number_of_accepted += 1
            
            self.event_jobs.remove(event)
        self.submit_job_num_list.append(submit_job_num)

