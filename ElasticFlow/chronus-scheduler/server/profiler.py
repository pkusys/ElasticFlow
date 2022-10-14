import os, sys
import copy
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..')
from .cluster import _Cluster
from utils.util import allocate_set, search_dict_list
from alg import PlaceMentFactory
import math

LIMIT_DURATION=2 * 60
LIMIT_RESOURCE=28
DELAY_TIME=60

class _Profiler(object):
    def __init__(self, proflie_time_interval=5, **kwargs):
        self.cluster = _Cluster()
        # self.cluster.init_infra(
        #     num_switch=1, 
        #     num_node_p_switch=4,
        #     num_gpu_p_node=8,
        #     num_cpu_p_node=64,
        #     mem_p_node=256
        # )
        self.pending_list = list()
        self.running_list = list()
        self.ending_list = list()
        self.job_name_list = list() 
        self.job_name_cluster = dict()
        self.user_submit_info = dict()
        self.proflie_time_interval = proflie_time_interval
        
    def build_cluster(self, node_num):
        self.cluster.init_infra(
            num_switch=1, 
            num_node_p_switch=node_num,
            num_gpu_p_node=8,
            num_cpu_p_node=64,
            mem_p_node=256
        )
        self.PM = PlaceMentFactory(cluster=self.cluster, name='policy', model_info={})
        global LIMIT_RESOURCE
        LIMIT_RESOURCE = node_num * 8 - 4 


    def submit_job(self, job):
        if not self.dry_run:
            job['profile_finish_time'] = job['submit_time']
            if job['num_gpu'] > 8:
                return True
            
            policy_list = allocate_set(job['num_gpu'])
            for policy in policy_list:
                job_with_policy = dict()
                job_with_policy['job'] = job
                job_with_policy['profile_submit_time'] = job['submit_time']
                job_with_policy['policy'] = copy.deepcopy(policy)
                self.pending_list.append(job_with_policy)
            return 


        job['profile_finish_time'] = job['submit_time']

        if self.profile_method == 'Clustering' and len(str(job['job_name'])) >= 10 :
            if job['job_name'] not in self.job_name_list:
                self.job_name_list.append(job['job_name'])
                self.job_name_cluster[job['job_name']] = [job]
            else:
                for prev_job in self.job_name_cluster[job['job_name']]:
                    if job['num_gpu'] == prev_job['num_gpu']: #  and abs(job['submit_time'] - prev_job['submit_time']) < self.proflie_time_interval:
                        return
                self.job_name_cluster[job['job_name']].append(job)

            
        if job['num_gpu'] > 8 and self.dry_run == False:
            return True
        

        if job['num_gpu'] > 8:
            job['num_gpu'] = 8
            policy_list = [[4, 4]]
        elif job['duration'] <= self.duration_limit: 
            policy_list = [[job['num_gpu']]]
        else:
            # policy_list = allocate_set(job['num_gpu'])
            policy_list = [[job['num_gpu']]]
        
        for policy in policy_list:
            job_with_policy = dict()
            job_with_policy['job'] = job
            job_with_policy['profile_submit_time'] = job['submit_time']
            job_with_policy['policy'] = copy.deepcopy(policy)
            self.pending_list.append(job_with_policy)
    

    def finish_all_jobs(self, ):
        return len(self.pending_list) + len(self.running_list) == 0


    def fake_simualate_speed(self, job, policy, job_info):
        if len(policy) == 1:
            return 1.0
        if len(job_info) > 0:
            if len(policy) == 2:
                return job_info['2']
            else:
                if '4' in job_info:
                    return job_info['4']
                else:
                    return job_info['2']
        
        return 5.0 / job['num_gpu'] + 2.0 / len(policy) + 1.0 / (1 + max(policy) - min(policy))
    

    def init_run_param(self, **kwargs):
        self.dry_run = kwargs.get('dry_run', False)
        self.duration_limit = kwargs.get('duration_limit', 5)
        self.profile_method = kwargs.get('profile_method', 'NO')
        self.save_dir = kwargs.get('save_dir', './')


    def summary(self,):
        attribute_list = ['job_id',  'early_exit', 'in_profiler_time', 'submit_time', 'finish_time', 'num_gpu', 'job_name', 'best_effort']
        job_info_dict = dict()
        job_id_list = list()
        for job_with_policy in self.ending_list: 
            job = job_with_policy['job']
            job_id = job['job_id']
            if job_id not in job_id_list: 
                job_id_list.append(job_id)
            submit_time = job_with_policy['profile_submit_time']
            finish_time = job_with_policy['profile_finish_time']
            if job_id not in job_info_dict:
                job_info_dict[job_id] = dict() 
                job_info_dict[job_id]['submit_time'] = submit_time
                job_info_dict[job_id]['finish_time'] = finish_time
            job_info_dict[job_id]['finish_time'] = max(job_info_dict[job_id]['finish_time'], finish_time)
            job_info_dict[job_id]['in_profiler_time'] = job_info_dict[job_id]['finish_time'] - job_info_dict[job_id]['submit_time']
            if job['duration'] <= self.duration_limit:
                job_info_dict[job_id]['early_exit'] = 1
            else:
                job_info_dict[job_id]['early_exit'] = 0
            job_info_dict[job_id]['num_gpu'] = job['num_gpu']
            if 'job_name' not in job:
                job['job_name'] = job['user'].name

            job_info_dict[job_id]['job_name'] = job['job_name']
            job_info_dict[job_id]['best_effort'] = job['best_effort']
            

        job_id_list.sort()
        self.profiler_name = '{}/profile_info_{}.csv'.format(self.save_dir, self.profile_method)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        with open(os.path.join(self.profiler_name), 'w') as f:
            header = ','.join(attribute_list)
            print(header, file=f)
            for job_id in job_id_list: 
                info_list = [str(job_id)]
                info_dict = job_info_dict[job_id]
                for attribute in attribute_list[1:]:
                    info_list.append(str(info_dict[attribute]))
                print(','.join(info_list), file=f)
        self.profiler_name = '{}/profile_info_{}_capacity.csv'.format(self.save_dir, self.profile_method)
        with open(os.path.join(self.profiler_name), 'w') as f:
            header = ','.join(['time','capacity'])
            print(header, file=f)
            for idx, gpu in enumerate(self.dynamic_capacity):
                print(','.join([str(idx), str(self.dynamic_capacity[idx])]), file=f)


        

    def dynamic_run(self, job_list, model_info, **kwargs):
        self.init_run_param(**kwargs)
        cur_time = 0 # job_list[0]['submit_time']
        job_list.sort(key=lambda e: e.__getitem__('submit_time'))
        node_list = kwargs.get('submission_dist', None)
        
        for job in job_list:
            self.submit_job(job) 
        

        if self.dry_run:
            self.dynamic_capacity = [0 for i in range(job_list[-1]['submit_time'] // 3600 + 10)]
            for job_with_policy in self.pending_list:
                job = job_with_policy['job']
                gpu = sum(job_with_policy['policy'])
                index = job['submit_time'] // 3600
                self.dynamic_capacity[index] += gpu


            self.pending_list.sort(key=lambda e: e.__getitem__('profile_submit_time'))
            # trigger mode, to accelerate 
            prev_hour = -1
            gpu_limit = 0 
            all_gpu_num = 0

            while not self.finish_all_jobs():
                if cur_time % 10000 == 0 and len(self.pending_list) > 0:
                    print(cur_time, len(self.pending_list), self.pending_list[0]['profile_submit_time'], self.pending_list[0]['job'].required_gpu_num)

                cur_hour = cur_time // 3600
                if cur_hour != prev_hour:
                    if cur_hour < len(node_list):
                        cur_gpu_limit = int(node_list[cur_hour] * 8)
                    else:
                        cur_gpu_limit = 2 * 8
                    add_gpu = cur_gpu_limit - gpu_limit

                    gpu_limit = cur_gpu_limit
                    prev_hour = cur_hour 
                    all_gpu_num = max(0, add_gpu + all_gpu_num)


                # flushing ending
                for job_with_policy in self.running_list:
                    # print(job_with_policy['profile_finish_time'], cur_time)
                    if job_with_policy['profile_finish_time'] <= cur_time:
                        # assert self.cluster.release_job_resource(job_with_policy['job']) == True
                        all_gpu_num += sum(job_with_policy['policy'])
                        all_gpu_num = min(all_gpu_num, gpu_limit)
                        self.ending_list.append(job_with_policy)
                        self.running_list.remove(job_with_policy)
                        job = job_with_policy['job']
                        job_info = {}
                        if job.required_gpu_num == 1:
                            job_info = {}
                        elif job['model']['name'] in model_info:
                            if str(job.required_gpu_num) not in model_info[job['model']['name']]:
                                model_info[job['model']['name']][str(job.required_gpu_num)] = {}
                            job_info = model_info[job['model']['name']][str(job.required_gpu_num)]
                        job_with_policy['speed'] = self.fake_simualate_speed(job_with_policy['job'], job_with_policy['policy'], job_info)
                        #job_with_policy['speed'] = 1
                        if self.profile_method == 'admission_control':
                            usr_name = job['user'].name
                            self.user_submit_info[usr_name].remove(job)
                # flushing pending
                remove_list = list()
                for job_with_policy in self.pending_list:
                    if job_with_policy['profile_submit_time'] <= cur_time:
                        if self.profile_method == 'admission_control':
                            job = job_with_policy['job']
                            usr_name = job['user'].name
                            if usr_name not in self.user_submit_info:
                                self.user_submit_info[usr_name] = list() 
                            if len(self.user_submit_info[usr_name]) == 0:
                                occupy_gpu = 0
                            else: 
                                occupy_gpu = sum([min(8, prev_job['num_gpu']) for prev_job in self.user_submit_info[usr_name]])
                            if occupy_gpu >= gpu_limit - 4:
                                job_with_policy['profile_submit_time'] += DELAY_TIME
                                if 'delay' not in job_with_policy:
                                    job_with_policy['delay'] = 0
                                
                                if job_with_policy['delay'] == 0:
                                    job_with_policy['profile_submit_time'] += DELAY_TIME
                                job_with_policy['delay'] += 1
                                # print('occupy_gpu', occupy_gpu)
                                # print('job_id', job_with_policy['job']['job_id'])
                                continue 

                        # if self.PM.place_jobs(job_with_policy['job'], job_with_policy['policy']): # TODO
                        if sum(job_with_policy['policy']) <= all_gpu_num:
                            all_gpu_num -= sum(job_with_policy['policy'])
                            if job_with_policy['job']['duration'] <= self.duration_limit:
                                job_with_policy['profile_finish_time'] = cur_time + job_with_policy['job']['duration'] 
                            else:
                                job_with_policy['profile_finish_time'] = cur_time +  min([LIMIT_DURATION, job_with_policy['job']['num_gpu'] * 30])
                            if self.profile_method == 'admission_control':
                                self.user_submit_info[usr_name].append(job)

                            self.running_list.append(job_with_policy)
                            remove_list.append(job_with_policy)

                for job_with_policy in remove_list:
                    self.pending_list.remove(job_with_policy)

                cur_time += self.proflie_time_interval
                if  len(self.pending_list) > 0 and self.pending_list[0]['profile_submit_time'] - cur_time > LIMIT_DURATION  * 10:
                    self.pending_list.sort(key=lambda e: e.__getitem__('profile_submit_time'))
                    cur_time = self.pending_list[0]['profile_submit_time'] // self.proflie_time_interval * self.proflie_time_interval - self.proflie_time_interval
            self.summary() 


    def run(self, job_list, model_info, **kwargs):
        self.init_run_param(**kwargs)
        cur_time = 0 # job_list[0]['submit_time']
        job_list.sort(key=lambda e: e.__getitem__('submit_time'))
        
        for job in job_list:
            self.submit_job(job)
        

        if self.dry_run:
            self.dynamic_capacity = [0 for i in range(job_list[-1]['submit_time'] // 3600 + 10)]
            for job_with_policy in self.pending_list:
                job = job_with_policy['job']
                gpu = sum(job_with_policy['policy'])
                index = job['submit_time'] // 3600
                self.dynamic_capacity[index] += gpu


            self.pending_list.sort(key=lambda e: e.__getitem__('profile_submit_time'))
            # trigger mode, to accelerate 
            while not self.finish_all_jobs():
                if cur_time % 10000 == 0 and len(self.pending_list) > 0:
                    print(cur_time, len(self.pending_list), self.pending_list[0]['profile_submit_time'], self.pending_list[0]['job'].required_gpu_num)
                # flushing ending
                for job_with_policy in self.running_list:
                    # print(job_with_policy['profile_finish_time'], cur_time)
                    if job_with_policy['profile_finish_time'] <= cur_time:
                        assert self.cluster.release_job_resource(job_with_policy['job']) == True
                        self.ending_list.append(job_with_policy)
                        self.running_list.remove(job_with_policy)
                        job = job_with_policy['job']
                        job_info = {}
                        if job.required_gpu_num == 1:
                            job_info = {}
                        elif job['model']['name'] in model_info:
                            if str(job.required_gpu_num) not in model_info[job['model']['name']]:
                                model_info[job['model']['name']][str(job.required_gpu_num)] = {}
                            job_info = model_info[job['model']['name']][str(job.required_gpu_num)]
                        job_with_policy['speed'] = self.fake_simualate_speed(job_with_policy['job'], job_with_policy['policy'], job_info)
                        if self.profile_method == 'admission_control':
                            usr_name = job['user'].name
                            self.user_submit_info[usr_name].remove(job)
                # flushing pending
                remove_list = list()
                for job_with_policy in self.pending_list:
                    if job_with_policy['profile_submit_time'] <= cur_time:
                        if self.profile_method == 'admission_control':
                            job = job_with_policy['job']
                            usr_name = job['user'].name
                            if usr_name not in self.user_submit_info:
                                self.user_submit_info[usr_name] = list() 
                            if len(self.user_submit_info[usr_name]) == 0:
                                occupy_gpu = 0
                            else: 
                                occupy_gpu = sum([min(8, prev_job['num_gpu']) for prev_job in self.user_submit_info[usr_name]])
                            if occupy_gpu >= LIMIT_RESOURCE:
                                job_with_policy['profile_submit_time'] += DELAY_TIME
                                print('occupy_gpu', occupy_gpu)
                                continue 

                        if self.PM.place_jobs(job_with_policy['job'], job_with_policy['policy']): # TODO
                            if job_with_policy['job']['duration'] <= self.duration_limit:
                                job_with_policy['profile_finish_time'] = cur_time + job_with_policy['job']['duration'] 
                            else:
                                job_with_policy['profile_finish_time'] = cur_time +  min([LIMIT_DURATION, job_with_policy['job']['num_gpu'] * 30])
                            if self.profile_method == 'admission_control':
                                self.user_submit_info[usr_name].append(job)

                            self.running_list.append(job_with_policy)
                            remove_list.append(job_with_policy)

                for job_with_policy in remove_list:
                    self.pending_list.remove(job_with_policy)

                cur_time += self.proflie_time_interval
                if  len(self.pending_list) > 0 and self.pending_list[0]['profile_submit_time'] - cur_time > LIMIT_DURATION  * 10:
                    self.pending_list.sort(key=lambda e: e.__getitem__('profile_submit_time'))
                    cur_time = self.pending_list[0]['profile_submit_time'] // self.proflie_time_interval * self.proflie_time_interval - self.proflie_time_interval
            self.summary() 
            return 
        else:
            def convert_to_bin(num_gpu):
                if num_gpu in [3]:
                    return 4
                elif num_gpu in [5, 6, 7]:
                    return 8 
                else:
                    return num_gpu
            for job_with_policy in self.pending_list:
                self.ending_list.append(job_with_policy)
                job = job_with_policy['job']
                if job.required_gpu_num == 1:
                    job_info = {}
                elif job['model']['name'] in model_info:
                    job_info = model_info[job['model']['name']][str(convert_to_bin(job.required_gpu_num))]
                else:
                    job_info= {}
                job_with_policy['speed'] = self.fake_simualate_speed(job_with_policy['job'], job_with_policy['policy'], job_info)
                #job_with_policy['speed'] = 1
                job_with_policy['profile_finish_time'] = 0
            self.pending_list.clear()

        profile_info_list = list()
        for idx, job_with_policy in enumerate(self.ending_list):
            job = job_with_policy['job']
            policy = job_with_policy['policy']
            speed = job_with_policy['speed']
            job['profile_finish_time'] = max(job['profile_finish_time'], job_with_policy['profile_finish_time'])
            profile_info = None
            if len(profile_info_list) > 0:
                profile_info = profile_info_list[-1]
                if profile_info['job'] != job:
                    profile_info = None

            if profile_info is None:
                profile_info = dict({'info': [(policy, speed)], 'job': job})
                profile_info_list.append(profile_info)
            else:
                profile_info['info'].append((policy, speed))
        
        for profile_info in profile_info_list:
            job = profile_info.pop('job')
            job.estimate_reward(profile_info['info'])
            


ProfileManager = _Profiler()


_allowed_symbols = [
    'ProfileManager'
]
            




