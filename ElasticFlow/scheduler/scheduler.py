from __future__ import print_function
from collections import OrderedDict
import copy
import csv
import cvxpy as cp
import re
import sys
import types
import time
import threading
import math
#parse args
import argparse
import copy
import os

import utils
import flags
import jobs
import cluster
import log
#import lp
import profiles
from runtime.rpc import scheduler_client
from runtime.rpc import scheduler_server

# import hosts
# import placement_scheme as scheme
# import cmd

#parse input arguments
flags.DEFINE_string('trace_file', 'tf_job.csv',
                '''Provide TF job trace file (*.csv, *.txt).
                    *.csv file, use \',\' as delimiter; *.txt file, user \' \' as deliminter. 
                    Default file is tf_job.csv ''')
flags.DEFINE_string('log_path', 'result-' + time.strftime("%Y%m%d-%H-%M-%S", time.localtime()),
                '''Simulation output folder, including cluster/node/gpu usage trace, pending job_queue info.
                Default folder is result-[time]''')
flags.DEFINE_string('scheme', 'yarn',
                '''
                Job placement scheme:
                0.count, just resource counting, without assignment (which gpu, which cpu)
                1.yarn, ms yarn
                2.random
                3.crandom (consolidate + random)
                4.greedy
                5.balance
                6.cbalance (consolidate + balance)
                Default is yarn''')
flags.DEFINE_string('schedule', 'fifo',
                '''
                Job schedule scheme:
                1.fifo
                2.fjf, fit job first( in fifo order)
                3.sjf, smallest job first
                4.lpjf, longest pending job first
                5.shortest, shortest-remaining-time job first
                6.shortest-gpu, shortest-remaining-gputime job first 
                7.dlas, discretized las 
                8.dlas-gpu, dlas using gpu time
                Default is fifo''')
flags.DEFINE_integer('num_switch', 1, 
                '''Part of cluster spec: the number of switches in this cluster, default is 1''')
flags.DEFINE_integer('num_node_p_switch', 32, 
                '''Part of cluster spec: the number of nodes under a single switch, default is 32''')
flags.DEFINE_integer('num_gpu_p_node', 8, 
                '''Part of cluster spec: the number of gpus on each node, default is 8''')
flags.DEFINE_integer('num_cpu_p_node', 64,
                '''Part of cluster spec: the number of cpus on each node, default is 64''')
flags.DEFINE_integer('mem_p_node', 256,
                '''Part of cluster spec: memory capacity on each node, default is 128''')
flags.DEFINE_integer('scheduling_slot', 60,
                '''The least re-scheduling time slot for ef and edf''')
flags.DEFINE_integer('restart_threshold', 100,
                '''restart trainers after a while''')
flags.DEFINE_string('cluster_spec', None,
                '''Part of cluster spec: cluster infra spec file, 
                this file will overwrite the specs from num_switch, num_node_p_switch, and num_gpu_p_node
                Spec format:
                    num_switch,num_node_p_switch,num_gpu_p_node
                    int,int,int''')

flags.DEFINE_boolean('print', False, 
                '''Enable print out information, default is False''')
flags.DEFINE_boolean('flush_stdout', True, 
                '''Flush stdout, default is True''')
flags.DEFINE_boolean('simulation', True, 
                '''whether the scheduler is for simulation or phisical cluster experiments, default is True''')
flags.DEFINE_boolean('fastforward', True, 
                '''Whether to fastforward the cluster experiments process, default is True''')
flags.DEFINE_version('0.1')
flags.DEFINE_boolean('early_stop', False, 
                '''Whether to stop a job if a target metric is reached''')
flags.DEFINE_string('gpu_type', 'A100', 
                '''The GPU to run on. It should match the trace provided.''')
flags.DEFINE_boolean('plot_figure', True, 
                '''Whether to write log files and plot figures afterwards''')


FLAGS = flags.FLAGS

#prepare JOBS list
JOBS = jobs.JOBS

#get host info
CLUSTER = cluster.CLUSTER

#get LOG object
LOG = log.LOG

#pre-run throughput information
THROUGHPUTS = profiles.THROUGHPUTS

job_stable = dict()
fast_forward_permission = False

MASTER_PORT = 22224
last_round_running_jobs, this_round_running_jobs = dict(), dict()
trainers_to_kill = {}
this_round_begin_time = None
last_round_gpu_allocations, gpu_allocations = None, None
job_to_be_killed = False
# for dlas-gpu cluster experiments
run_jobs = None

global_lock = threading.Lock()
global_ready_lock = threading.Lock()
commands = []

schedule_count = 0

def report_ready_callback(trainer_id):
    """Callback for tainers reporting ready. For overhead estimation and for debug.
    @param trainer_id: The id of the ready trainer.
    """
    print("received report ready request of trainer_id", trainer_id)
    global trainers_to_kill, global_ready_lock, this_round_begin_time
    global job_stable, commands, job_to_be_killed, last_round_gpu_allocations
    global gpu_allocations
    global_ready_lock.acquire()
    """for eachjob in trainers_to_kill:
        if trainer_id in trainers_to_kill[eachjob]:
            trainers_to_kill[eachjob].remove(trainer_id)
    for eachjob in trainers_to_kill:
        if len(trainers_to_kill[eachjob]) > 0:
            global_ready_lock.release()
            return"""
    last_round_gpu_allocations[trainer_id // CLUSTER.num_gpu_p_node][trainer_id % CLUSTER.num_gpu_p_node] = 0
    for node in last_round_gpu_allocations:
        for gpu in node:
            if gpu == 1:
                global_ready_lock.release()
                return
    for command in commands:
        scheduler_rpc_client.schedule(command)
    scheduler_rpc_client.schedule('F')
    scheduler_rpc_client.schedule('T')
    # all jobs have been killed. no running jobs in cluster
    job_to_be_killed = False
    last_round_gpu_allocations = gpu_allocations
    global_ready_lock.release()


def report_stable_callback(job_id):
    """Callback for tainers reporting stable status and prepare for fast forward
    @param job_id: The id of the stable job(s).
    """
    print("received fastforward request of job", job_id)
    receive_time = time.time()
    global job_stable, fast_forward_permission, this_round_begin_time, global_lock
    global_lock.acquire()
    if job_id not in job_stable:
        print("job", job_id, "requested fast forward before scaling")
    if job_stable[job_id] != 0:
        print("unexpected request from job", job_id, job_stable)
    assert job_id in job_stable and job_stable[job_id] == 0
    job_stable[job_id] = 1
    """if FLAGS.schedule == 'dlas-gpu':
        # workaround
        job = utils.search_dict_list(JOBS.runnable_jobs, 'job_idx', job_id)
        job['overhead'] = math.floor(receive_time - this_round_begin_time)
        print("job", job_id, "overhead", job['overhead'])
        for each_job in JOBS.runnable_jobs:
            if each_job['status'] != 'RUNNING':
                continue
            if each_job['num_gpu'] == 0 or each_job['placements'] is None or len(each_job['placements']) == 0:
                continue
            if each_job['job_idx'] not in job_stable:
                if each_job['job_idx'] in this_round_running_jobs:
                    each_job['overhead'] = 0
                    continue
                global_lock.release()
                return
            if job_stable[each_job['job_idx']] == 0:
                global_lock.release()
                return"""
    if FLAGS.schedule == 'dlas-gpu':
        job = utils.search_dict_list(JOBS.runnable_jobs, 'job_idx', job_id)
        job['overhead'] = math.floor(receive_time - this_round_begin_time)
        print("job", job_id, "overhead", job['overhead'])
        if job['overhead'] > 20:
            job['overhead'] = 20
        for each_job in JOBS.runnable_jobs:
            if each_job['status'] != 'RUNNING':
                continue
            if each_job['num_gpu'] == 0 or each_job['node_set'] is None:
                continue
            if each_job['job_idx'] not in job_stable:
                if each_job['job_idx'] in this_round_running_jobs:
                    each_job['overhead'] = 0
                    continue
                global_lock.release()
                return
            if job_stable[each_job['job_idx']] == 0:
                global_lock.release()
                return
    else:
        job = utils.search_dict_list(JOBS.running_jobs, 'job_idx', job_id)
        job['overhead'] = math.floor(receive_time - this_round_begin_time)
        print("job", job_id, "overhead", job['overhead'])
        if job['overhead'] > 20:
            job['overhead'] = 20
        for each_job in JOBS.running_jobs:
            if each_job['num_gpu'] == 0 or each_job['node_set'] is None:
                continue
            if each_job['job_idx'] not in job_stable:
                if each_job['job_idx'] in this_round_running_jobs:
                    each_job['overhead'] = 0
                    continue
                global_lock.release()
                return
            if job_stable[each_job['job_idx']] == 0:
                global_lock.release()
                return
    fast_forward_permission = True
    global_lock.release()
    print("ALL JOBS READY")


def parse_job_file(trace_file):
    #check trace_file is *.csv
    fd = open(trace_file, 'r')
    deli = ','
    if ((trace_file.find('.csv') == (len(trace_file) - 4))):
        deli = ','
    elif ((trace_file.find('.txt') == (len(trace_file) - 4))):
        deli = ' '

    reader = csv.DictReader(fd, delimiter = deli) 
    ''' Add job from job trace file'''
    keys = reader.fieldnames
    utils.print_fn('--------------------------------- Read TF jobs from: %s ---------------------------------' % trace_file) 
    utils.print_fn('    we get the following fields:\n        %s' % keys)
    job_idx = 0
    for row in reader:
        #add job into JOBS
        JOBS.add_job(row)
        # JOBS.read_job_info(job_idx, 'num_gpu')
        job_idx += 1

    assert job_idx == len(JOBS.job_list) 
    assert JOBS.num_job == len(JOBS.job_list) 
    # JOBS.print_all_job_size_info()
    JOBS.sort_all_jobs()
    # print(lp.prepare_job_info(JOBS.job_list[0]))
    utils.print_fn('---------------------------------- Get %d TF jobs in total ----------------------------------' % job_idx)
    # JOBS.read_all_jobs()
    fd.close()

def parse_cluster_spec():
    global last_round_gpu_allocations
    if FLAGS.cluster_spec:
        print(FLAGS.cluster_spec)
        spec_file = FLAGS.cluster_spec
        fd = open(spec_file, 'r')
        deli = ','
        if ((spec_file.find('.csv') == (len(spec_file) - 4))):
            deli = ','
        elif ((spec_file.find('.txt') == (len(spec_file) - 4))):
            deli = ' '
        reader = csv.DictReader(fd, delimiter = deli) 
        keys = reader.fieldnames
        utils.print_fn(keys)
        if 'num_switch' not in keys:
            return
        if 'num_node_p_switch' not in keys:
            return
        if 'num_gpu_p_node' not in keys:
            return
        if 'num_cpu_p_node' not in keys:
            return
        if 'mem_p_node' not in keys:
            return
        
        ''' there should be only one line remaining'''
        assert reader.line_num == 1

        ''' get cluster spec '''
        for row in reader:
            # utils.print_fn('num_switch %s' % row['num_switch'])
            FLAGS.num_switch = int(row['num_switch'])
            FLAGS.num_node_p_switch = int(row['num_node_p_switch'])
            FLAGS.num_gpu_p_node = int(row['num_gpu_p_node'])
            FLAGS.num_cpu_p_node = int(row['num_cpu_p_node'])
            FLAGS.mem_p_node = int(row['mem_p_node'])
        fd.close()

    utils.print_fn("num_switch: %d" % FLAGS.num_switch)
    utils.print_fn("num_node_p_switch: %d" % FLAGS.num_node_p_switch)
    utils.print_fn("num_gpu_p_node: %d" % FLAGS.num_gpu_p_node)
    utils.print_fn("num_cpu_p_node: %d" % FLAGS.num_cpu_p_node)
    utils.print_fn("mem_p_node: %d" % FLAGS.mem_p_node)

    '''init infra'''
    CLUSTER.init_infra()
    # utils.print_fn(lp.prepare_cluster_info())
    last_round_gpu_allocations = [[0 for gpu in range(CLUSTER.num_gpu_p_node)] for _ in range(CLUSTER.num_node)]
    utils.print_fn('--------------------------------- End of cluster spec ---------------------------------')
    return 


'''
Allocate job resource
'''
def try_get_job_res(job):
    '''
    select placement scheme
    '''
    if FLAGS.scheme == 'elastic':
        ret = CLUSTER.elastic_placement(job)
    elif FLAGS.scheme == 'yarn':
        ret = CLUSTER.ms_yarn_placement(job)
    elif FLAGS.scheme == 'balance':
        ret = lp.placement(job)
        # ret = lp.min_new_job(job)
    elif FLAGS.scheme == 'random':
        ret = CLUSTER.random_placement(job)
    elif FLAGS.scheme == 'crandom':
        ret = CLUSTER.consolidate_random_placement(job)
    elif FLAGS.scheme == 'greedy':
        ret = CLUSTER.greedy_placement(job)
    elif FLAGS.scheme == 'gandiva':
        ret = CLUSTER.gandiva_placement(job)
    elif FLAGS.scheme == 'count':
        ret = CLUSTER.none_placement(job)
    else:
        ret = CLUSTER.ms_yarn_placement(job)
    if ret == True:
        # job['status'] = 'RUNNING'
        pass
    return ret


#ef_allocation_heuristic
def ef_sim_allocation(job_dict, start_time, assign_gpu=False, assigned_gpus=-1, simulation=False, future_free_gpus=None):
    event_time = start_time
    return_value = True
    global_batch_size = str(job_dict['batch_size'])
    if 'iter_left' not in job_dict:
        job_dict['iter_left'] = job_dict['iteration']
    iter_left = job_dict['iter_left']
    if iter_left <= 0:
        print(job_dict)
    assert iter_left > 0

    new_allocations = {event_time:0}
    if future_free_gpus is None:
        future_free_gpus = CLUSTER.future_free_gpus
    new_future_free_gpus = copy.deepcopy(future_free_gpus)

    aligned_ddl = utils.align_to_time_slot(start_time, job_dict['ddl'], FLAGS.scheduling_slot)
    # add ddl into future free gpus
    if aligned_ddl not in new_future_free_gpus and aligned_ddl >= start_time:
        for each_event_time in new_future_free_gpus:
            if each_event_time < aligned_ddl:
                last_event_gpu = new_future_free_gpus[each_event_time]
            else:
                break
        new_future_free_gpus[aligned_ddl] = last_event_gpu
        new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
    

    available_gpu = new_future_free_gpus[start_time]
    if assign_gpu:
        quota = assigned_gpus
        assert quota <= available_gpu
        found_future_time = False
        for each_event_time in future_free_gpus:
            future_event_time = each_event_time
            if future_event_time < event_time:
                available_gpu = future_free_gpus[future_event_time]
                del new_future_free_gpus[future_event_time]
                continue
            elif future_event_time == event_time:
                available_gpu = future_free_gpus[future_event_time]
                continue
            found_future_time = True
            break
        if not found_future_time:
            future_event_time = aligned_ddl
        new_allocations[event_time] = quota
        estimated_real_end_time = math.ceil(event_time + iter_left / float(
            THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(quota)]))
        estimated_end_time = utils.align_to_time_slot(start_time, estimated_real_end_time, FLAGS.scheduling_slot)
        if future_event_time >= estimated_end_time and job_dict['ddl'] >= estimated_real_end_time:
            for each_event_time in future_free_gpus:
                if each_event_time < event_time:
                    continue
                if each_event_time < estimated_end_time:
                    last_event_gpu = new_future_free_gpus[each_event_time]
                    new_future_free_gpus[each_event_time] -= quota
                    assert new_future_free_gpus[each_event_time] >= 0

            if event_time != estimated_end_time:
                new_future_free_gpus[event_time] = available_gpu - quota
            assert new_future_free_gpus[event_time] >= 0
            if estimated_end_time not in future_free_gpus:
                new_future_free_gpus[estimated_end_time] = last_event_gpu
                assert new_future_free_gpus[event_time] >= 0
            new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
                
            if simulation:
                job_dict['next_level_allocation'] = utils.merge_dict(new_allocations)
                job_dict['marginal_gputime'] = (utils.get_allocation_time(
                    new_allocations, estimated_end_time) - utils.get_allocation_time(
                    job_dict['allocations'], job_dict['end_time'])) / (job_dict['next_level_gpu'] - job_dict['num_gpu'])
                job_dict['next_level_future_gpus'] = utils.merge_dict(new_future_free_gpus)
                job_dict['next_level_endtime'] = estimated_end_time
            else:
                if assign_gpu:
                    job_dict['old_end_time'] = job_dict['end_time']
                CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
                job_dict['end_time'] = estimated_end_time
                job_dict['real_end_time'] = estimated_real_end_time
                job_dict['allocations'] = utils.merge_dict(new_allocations)
                
            return estimated_real_end_time <= job_dict['ddl']
        else:
            duration = future_event_time - event_time
            if future_event_time == aligned_ddl:
                duration = job_dict['ddl'] - event_time
            iterations = duration * float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                quota)])
            iter_left -= iterations
            if event_time in new_future_free_gpus:
                new_future_free_gpus[event_time] -= quota
            else:
                new_future_free_gpus[event_time] = last_event_gpu - new_allocations[event_time]
            assert new_future_free_gpus[event_time] >= 0
        event_time = future_event_time # assign GPU!
        new_allocations[event_time] = 0

    
    quota = utils.get_next_level(job_dict, gpu_num=0)
    last_quota, last_last_quota = 0, 0
    point = start_time - 1
    # allocate from 1 gpu only
    while quota <= CLUSTER.num_gpu and quota <= job_dict['max_gpu']:

        period_end = aligned_ddl
        tmp_iters = 0 # 
        for period_start in new_future_free_gpus:
            if period_start >= aligned_ddl:
                continue
            if period_start not in new_allocations:
                new_allocations[period_start] = 0
        for period_start in reversed(list(new_future_free_gpus.keys())):
            if period_start >= aligned_ddl:
                continue
            if period_start < event_time:
                break
            allocation = quota
            if new_future_free_gpus[period_start] < quota:
                allocation = min(quota, utils.get_last_level(
                    job_dict, gpu_num=1+new_future_free_gpus[period_start]))
            if allocation == 0:
                period_end = period_start
                continue
            duration = period_end - period_start
            if period_end == aligned_ddl:
                duration = job_dict['ddl'] - period_start
            tmp_iters += duration * float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                    allocation)])
            if tmp_iters >= iter_left:
                point = period_start
                # allocate from period start, and return!
                last_allocation_time = None
                last_allocation = None
                estimated_end_time = None
                for eachtime in new_future_free_gpus:
                    if eachtime > aligned_ddl:
                        break # >=
                    if eachtime < event_time:
                        continue
                    
                    if last_allocation_time is None:
                        last_allocation_time = eachtime
                        if eachtime >= period_start:
                            last_allocation = min(quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
                        else:
                            last_allocation = min(last_quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
                        continue
                    new_allocations[last_allocation_time] = last_allocation
                    new_future_free_gpus[last_allocation_time] -= new_allocations[last_allocation_time]
                    assert new_future_free_gpus[last_allocation_time] >= 0

                    if last_allocation == 0:
                        last_allocation_time = eachtime
                        if eachtime >= period_start:
                            last_allocation = min(quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
                        else:
                            last_allocation = min(last_quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
                        continue

                    duration = eachtime - last_allocation_time
                    if eachtime == aligned_ddl:
                        duration = job_dict['ddl'] - last_allocation_time
                    iterations = duration * float(
                        THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                        last_allocation)])
                    if last_allocation_time < period_start:
                        assert iterations < iter_left
                        iter_left -= iterations
                    else:
                        if iterations < iter_left:
                            iter_left -= iterations
                        else:
                            real_time_left = math.ceil(iter_left / float(
                                THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                                new_allocations[last_allocation_time])]))
                            time_left = utils.align_to_time_slot(0, real_time_left, FLAGS.scheduling_slot)
                            estimated_end_time = last_allocation_time + time_left
                            estimated_real_end_time = last_allocation_time + real_time_left
                            if estimated_end_time > eachtime:
                                print(estimated_end_time, eachtime, new_future_free_gpus)
                            assert estimated_end_time <= eachtime
                            break
                    last_allocation_time = eachtime
                    if eachtime >= period_start:
                        last_allocation = min(quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
                    else:
                        last_allocation = min(last_quota, utils.get_last_level(
                                job_dict, gpu_num=1+new_future_free_gpus[eachtime]))

                if estimated_end_time is None:
                    new_allocations[last_allocation_time] = last_allocation
                    new_future_free_gpus[last_allocation_time] -= new_allocations[last_allocation_time]
                    assert new_future_free_gpus[last_allocation_time] >= 0
                    real_time_left = math.ceil(iter_left / float(
                        THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                        new_allocations[last_allocation_time])]))
                    time_left = utils.align_to_time_slot(0, real_time_left, FLAGS.scheduling_slot)
                    estimated_real_end_time = last_allocation_time + real_time_left
                    estimated_end_time = last_allocation_time + time_left
                new_allocations = OrderedDict(sorted(new_allocations.items(), key=lambda t: t[0]))

                last_event_gpu = CLUSTER.num_gpu
                for each_event_time in future_free_gpus:
                    if each_event_time < event_time:
                        continue
                    if each_event_time < estimated_end_time:
                        last_event_gpu = future_free_gpus[each_event_time]
                            
                if estimated_end_time not in future_free_gpus:
                    new_future_free_gpus[estimated_end_time] = last_event_gpu
                    assert new_future_free_gpus[estimated_end_time] >= 0
                new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
                
                if simulation:
                    job_dict['next_level_allocation'] = utils.merge_dict(new_allocations)
                    job_dict['marginal_gputime'] = (utils.get_allocation_time(
                        new_allocations, estimated_end_time) - utils.get_allocation_time(
                        job_dict['allocations'], job_dict['end_time'])) / (job_dict['next_level_gpu'] - job_dict['num_gpu'])
                    job_dict['next_level_future_gpus'] = utils.merge_dict(new_future_free_gpus)
                    job_dict['next_level_endtime'] = estimated_end_time
                else:
                    if assign_gpu:
                        job_dict['old_end_time'] = job_dict['end_time']
                    CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
                    job_dict['end_time'] = estimated_end_time
                    job_dict['real_end_time'] = estimated_real_end_time
                    job_dict['allocations'] = utils.merge_dict(new_allocations)
            
                return estimated_real_end_time <= job_dict['ddl']

            period_end = period_start

        last_last_quota = last_quota
        last_quota = quota
        quota = utils.get_next_level(job_dict, gpu_num=quota)
        if quota == last_quota:
            break

    #if job_dict['job_idx'] == 
    # allocate from period start
    last_allocation_time = None
    last_allocation = None
    for eachtime in new_future_free_gpus:
        if eachtime > aligned_ddl:
             break
        if eachtime < event_time:
            continue
                    
        if last_allocation_time is None:
            last_allocation_time = eachtime
            if eachtime >= point:
                last_allocation = min(last_quota, utils.get_last_level(
                    job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
            else:
                last_allocation = min(last_last_quota, utils.get_last_level(
                    job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
            continue
        new_allocations[last_allocation_time] = last_allocation
        new_future_free_gpus[last_allocation_time] -= new_allocations[last_allocation_time]
        assert new_future_free_gpus[last_allocation_time] >= 0

        if last_allocation == 0:
            last_allocation_time = eachtime
            #last_allocation = min(last_quota, new_future_free_gpus[eachtime])
            #last_allocation = min(last_quota, 
            #    get_last_level(job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
            if eachtime >= point:
                last_allocation = min(last_quota, utils.get_last_level(
                    job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
            else:
                last_allocation = min(last_last_quota, utils.get_last_level(
                    job_dict, gpu_num=1+new_future_free_gpus[eachtime]))
            continue

        duration = eachtime - last_allocation_time
        if eachtime == aligned_ddl:
            duration = job_dict['ddl'] - last_allocation_time
        iterations = duration * float(
            THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
            last_allocation)])
        if iterations >= iter_left:
            print("job", job_dict['job_idx'])
        assert iterations < iter_left
        iter_left -= iterations
        
        last_allocation_time = eachtime
        last_allocation = min(last_quota, utils.get_last_level(
            job_dict, gpu_num=new_future_free_gpus[eachtime]+1))

    assert iter_left > 0

    # allocate from ddl
    last_allocation_time = max(aligned_ddl, event_time)
    for each_event_time in new_future_free_gpus:
        if each_event_time <= aligned_ddl:
            continue
        quota = min(utils.get_last_level(
            job_dict, gpu_num=new_future_free_gpus[last_allocation_time]+1), job_dict['max_gpu'])
        if quota == 0:
            new_allocations[last_allocation_time] = quota
            last_allocation_time = each_event_time
            continue
        new_allocations[last_allocation_time] = quota
        iterations = (each_event_time - last_allocation_time) * float(
            THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
            new_allocations[last_allocation_time])])
        
        if iterations >= iter_left:
            estimated_real_end_time = math.ceil(last_allocation_time + iter_left / float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(quota)]))
            estimated_end_time = utils.align_to_time_slot(start_time, estimated_real_end_time, FLAGS.scheduling_slot)
            assert estimated_end_time <= each_event_time
            for each_event_time1 in new_future_free_gpus:
                if each_event_time1 < aligned_ddl:
                    continue
                if each_event_time1 < estimated_end_time:
                    if each_event_time1 in future_free_gpus:
                        last_event_gpu = future_free_gpus[each_event_time1]
                if each_event_time1 >= last_allocation_time and each_event_time1 < estimated_end_time:
                    new_future_free_gpus[each_event_time1] -= quota
                    assert new_future_free_gpus[each_event_time1] >= 0
            

            if last_allocation_time != estimated_end_time:
                new_future_free_gpus[last_allocation_time] = last_event_gpu - quota
            assert new_future_free_gpus[last_allocation_time] >= 0
            
            if estimated_end_time not in future_free_gpus:
                new_future_free_gpus[estimated_end_time] = last_event_gpu
                assert new_future_free_gpus[estimated_end_time] >= 0
            new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
                
            if simulation:
                job_dict['next_level_allocation'] = utils.merge_dict(new_allocations)
                job_dict['marginal_gputime'] = (utils.get_allocation_time(
                    new_allocations, estimated_end_time) - utils.get_allocation_time(
                    job_dict['allocations'], job_dict['end_time'])) / (job_dict['next_level_gpu'] - job_dict['num_gpu'])
                job_dict['next_level_future_gpus'] = utils.merge_dict(new_future_free_gpus)
                job_dict['next_level_endtime'] = estimated_end_time
            else:
                if assign_gpu:
                    job_dict['old_end_time'] = job_dict['end_time']
                CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
                job_dict['end_time'] = estimated_end_time
                job_dict['real_end_time'] = estimated_real_end_time
                job_dict['allocations'] = utils.merge_dict(new_allocations)
            
            return estimated_real_end_time <= job_dict['ddl']
        else:
            iter_left -= iterations
            for each_event_time1 in future_free_gpus:
                if each_event_time1 < last_allocation_time:
                    last_event_gpu = new_future_free_gpus[each_event_time1]
            for each_event_time1 in new_future_free_gpus:
                if each_event_time1 >= last_allocation_time and each_event_time1 < each_event_time:
                    new_future_free_gpus[each_event_time1] -= quota
                    assert new_future_free_gpus[each_event_time1] >= 0

        last_allocation_time = each_event_time
    
    assert iter_left > 0
    available_gpu = CLUSTER.num_gpu
    quota = min(CLUSTER.num_gpu, job_dict['max_gpu'])
    if assign_gpu:
        if assigned_gpus > quota:
            print("assigned_gpus", assigned_gpus, "quota", quota)
            print("job", job_dict['job_idx'], "new_allocations", new_allocations)
        assert assigned_gpus <= quota
        quota = assigned_gpus
    new_allocations[last_allocation_time] = quota
    new_allocations = OrderedDict(sorted(new_allocations.items(), key=lambda t: t[0]))
    estimated_real_end_time = math.ceil(last_allocation_time + iter_left / float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(quota)]))
    estimated_end_time = utils.align_to_time_slot(start_time, estimated_real_end_time, FLAGS.scheduling_slot)
    assert estimated_end_time > job_dict['ddl']
    return_value = False
    new_future_free_gpus[last_allocation_time] = available_gpu - quota
    assert new_future_free_gpus[last_allocation_time] >= 0
    new_future_free_gpus[estimated_end_time] = CLUSTER.num_gpu
    new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
    if simulation:
        job_dict['next_level_allocation'] = utils.merge_dict(new_allocations)
        gpu_increased = job_dict['next_level_gpu'] - job_dict['num_gpu']
        if gpu_increased == 0:
            job_dict['marginal_gputime'] = -sys.maxsize#sys.maxsize
        else:
            job_dict['marginal_gputime'] = (utils.get_allocation_time(
                new_allocations, estimated_end_time) - utils.get_allocation_time(
                job_dict['allocations'], job_dict['end_time'])) / gpu_increased
        job_dict['next_level_future_gpus'] = utils.merge_dict(new_future_free_gpus)
        job_dict['next_level_endtime'] = estimated_end_time
    else:
        if assign_gpu:
            job_dict['old_end_time'] = job_dict['end_time']
        CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
        job_dict['end_time'] = estimated_end_time
        job_dict['real_end_time'] = estimated_real_end_time
        job_dict['allocations'] = utils.merge_dict(new_allocations)
            

    return estimated_real_end_time <= job_dict['ddl']
       

def edf_sim_allocation(job_dict, start_time, simulation=True, future_free_gpus=None):
    event_time = start_time
    return_value = True
    global_batch_size = str(job_dict['batch_size'])
    if 'iter_left' not in job_dict:
        job_dict['iter_left'] = job_dict['iteration']
    #print("time", start_time, "job", job_dict['job_idx'], CLUSTER.future_free_gpus, CLUSTER.free_gpu)
    iter_left = job_dict['iter_left']

    new_allocations = {event_time:0}
    if future_free_gpus is None:
        future_free_gpus = CLUSTER.future_free_gpus
    new_future_free_gpus = copy.deepcopy(future_free_gpus)

    available_gpu = new_future_free_gpus[start_time]
    #available_gpu = get_available_gpu(start_time)
    base_quota = min(job_dict['max_gpu'], CLUSTER.num_gpu)

    for future_event_time in future_free_gpus:
        if future_event_time < event_time:
            available_gpu = future_free_gpus[future_event_time]
            del new_future_free_gpus[future_event_time]
            continue
        elif future_event_time == event_time:
            available_gpu = future_free_gpus[future_event_time]
            continue
        # the least number of GPU to meet DDL requirements
        duration = future_event_time - event_time    
        #print("job", job_dict['job_id'], "future_event_time", future_event_time, "base_quota", base_quota)
        #print("***FREE GPU", available_gpu, "at ", event_time)
        
        if available_gpu >= base_quota and base_quota > 0:
            new_allocation = base_quota
            new_allocations[event_time] = new_allocation
            estimated_end_time = math.ceil(event_time + iter_left / float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(base_quota)]))
            last_event_gpu = CLUSTER.num_gpu
            # if future_event_time > job_dict['end_time']
            if future_event_time >= estimated_end_time:
                for each_event_time in future_free_gpus:
                    if each_event_time < event_time:
                        continue
                    if each_event_time < estimated_end_time:
                        last_event_gpu = new_future_free_gpus[each_event_time]
                        new_future_free_gpus[each_event_time] -= base_quota
                        assert new_future_free_gpus[each_event_time] >= 0

                if event_time != estimated_end_time:
                    new_future_free_gpus[event_time] = available_gpu - base_quota
                assert new_future_free_gpus[event_time] >= 0
                if estimated_end_time not in future_free_gpus:
                    #new_future_free_gpus[estimated_end_time] = last_event_gpu
                    new_future_free_gpus[utils.align_to_time_slot(start_time, 
                        estimated_end_time, FLAGS.scheduling_slot)] = last_event_gpu
                    assert new_future_free_gpus[event_time] >= 0
                new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
                
                CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
                job_dict['new_end_time'] = estimated_end_time
                job_dict['new_allocations'] = utils.merge_dict(new_allocations)
                    
                #print("event time:", event_time, "allocation:", job_dict['new_allocations'], "job",job_dict['job_id'], "end time", job_dict['new_end_time'])
                #print("future_free_gpus", new_future_free_gpus)
                if estimated_end_time > job_dict['ddl']:
                    #print("ERROR: Fail to meet deadline requirements!")
                    return_value = False
                return return_value
            else:
                #iterations = math.ceil(duration * float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                #    new_allocation)]))
                iterations = duration * float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(
                    new_allocation)])
                iter_left -= iterations
                if event_time in new_future_free_gpus:
                    new_future_free_gpus[event_time] -= new_allocation
                else:
                    new_future_free_gpus[event_time] = last_event_gpu - new_allocations[event_time]
                assert new_future_free_gpus[event_time] >= 0

        elif available_gpu > 0 and base_quota > 0:

            for throughput in THROUGHPUTS[job_dict['model']['name']][global_batch_size]:
                gpu_num = int(throughput)
                if gpu_num > available_gpu:
                    break
                #iterations = math.ceil(duration * float(
                #    THROUGHPUTS[job_dict['model']['name']][global_batch_size][throughput]))
                iterations = duration * float(
                    THROUGHPUTS[job_dict['model']['name']][global_batch_size][throughput])
                new_allocations[event_time] = gpu_num
            estimated_end_time = math.ceil(
                event_time + iter_left / float(
                    THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(new_allocations[event_time])]))
            last_event_gpu = CLUSTER.num_gpu
            # if future_event_time > job_dict['end_time']
            if future_event_time >= estimated_end_time:
                for each_event_time in future_free_gpus:
                    if each_event_time < event_time:
                        continue
                    if each_event_time < estimated_end_time:
                        last_event_gpu = new_future_free_gpus[each_event_time]
                        new_future_free_gpus[each_event_time] -= new_allocations[event_time]
                        assert new_future_free_gpus[each_event_time] >= 0

                new_future_free_gpus[event_time] = available_gpu - new_allocations[event_time]
                assert new_future_free_gpus[event_time] >= 0
                if estimated_end_time not in future_free_gpus:
                    #new_future_free_gpus[estimated_end_time] = last_event_gpu
                    new_future_free_gpus[utils.align_to_time_slot(start_time, 
                        estimated_end_time, FLAGS.scheduling_slot)] = last_event_gpu
                    assert new_future_free_gpus[event_time] >= 0
                new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
                
                CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
                job_dict['new_end_time'] = estimated_end_time
                job_dict['new_allocations'] = utils.merge_dict(new_allocations)
                    
                #print("event time:", event_time, "allocation:", job_dict['new_allocations'], "job",job_dict['job_id'], "end time", job_dict['new_end_time'])
                #print("future_free_gpus", new_future_free_gpus)
                if estimated_end_time > job_dict['ddl']:
                    #print("ERROR: Fail to meet deadline requirements!")
                    return_value = False
                return return_value
            iter_left -= iterations
            new_future_free_gpus[event_time] -= new_allocations[event_time]
            assert new_future_free_gpus[event_time] >= 0
        elif available_gpu == 0:
            new_allocations[event_time] = 0
        available_gpu = new_future_free_gpus[future_event_time]
        event_time = future_event_time
        if event_time > job_dict['ddl']:
            #print("ERROR: Fail to meet deadline requirements!")
            return_value = False
    
    # another round
    available_gpu = CLUSTER.num_gpu

    assert base_quota <= available_gpu
    if base_quota > available_gpu:
        print("base_quota > available_gpu")
        return False

    # event time >= max(future event time)
    new_allocations[event_time] = base_quota
    estimated_end_time = math.ceil(event_time + iter_left / float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][str(base_quota)]))
    if estimated_end_time > job_dict['ddl']:
        #print("ERROR: Fail to meet deadline requirements!")
        return_value = False
    new_future_free_gpus[event_time] = available_gpu - base_quota
    assert new_future_free_gpus[event_time] >= 0
    #new_future_free_gpus[estimated_end_time] = CLUSTER.num_gpu
    new_future_free_gpus[utils.align_to_time_slot(start_time, 
        estimated_end_time, FLAGS.scheduling_slot)] = CLUSTER.num_gpu
    new_future_free_gpus = OrderedDict(sorted(new_future_free_gpus.items(), key=lambda t: t[0]))
    CLUSTER.future_free_gpus = utils.merge_dict(new_future_free_gpus)
    job_dict['new_end_time'] = estimated_end_time
    job_dict['new_allocations'] = utils.merge_dict(new_allocations)
        

    #print("event time:", event_time, "allocation:", job_dict['new_allocations'], "job",job_dict['job_id'],"end time", job_dict['new_end_time'])
    #print("future_free_gpus", new_future_free_gpus)
    return return_value


def get_marginal(job, allocatable_gpus, cur_time):
    if job['next_level_gpu'] - job['num_gpu'] > allocatable_gpus or job['next_level_gpu'] == job['num_gpu']:
        job['marginal_gputime'] = -sys.maxsize
        return
    # simulate to re-allocate the job
    last_allocated_gpu = 0
    tmp_future_free_gpus = copy.deepcopy(CLUSTER.future_free_gpus)
    if job['end_time'] not in tmp_future_free_gpus:
        for each_event_time in tmp_future_free_gpus:
            if each_event_time < job['end_time']:
                last_event_gpu = tmp_future_free_gpus[each_event_time]
            else:
                break
        tmp_future_free_gpus[job['end_time']] = last_event_gpu
        tmp_future_free_gpus = OrderedDict(sorted(tmp_future_free_gpus.items(), key=lambda t: t[0]))
    for eachallocation in job['allocations']:
        if eachallocation not in tmp_future_free_gpus:
            tmp_future_free_gpus[eachallocation] = get_available_gpu(
                eachallocation, future_free_gpus=tmp_future_free_gpus)
            tmp_future_free_gpus = OrderedDict(sorted(tmp_future_free_gpus.items(), key=lambda t: t[0]))
        for future_event_time in tmp_future_free_gpus:
            if future_event_time < eachallocation:
                continue
            if future_event_time >= job['end_time']:
                break
            delta = job['allocations'][eachallocation] - last_allocated_gpu
            tmp_future_free_gpus[future_event_time] += delta
        
        last_allocated_gpu = job['allocations'][eachallocation]
    for future_event_time in tmp_future_free_gpus:
        if tmp_future_free_gpus[future_event_time] > CLUSTER.num_gpu:
            print("tmp_future_free_gpus",tmp_future_free_gpus)
            print("CLUSTER.future_free_gpus", CLUSTER.future_free_gpus)
        assert tmp_future_free_gpus[future_event_time] <= CLUSTER.num_gpu
    tmp_future_free_gpus = utils.merge_dict(tmp_future_free_gpus, start_time=cur_time)
    ef_sim_allocation(job, cur_time, assign_gpu=True, assigned_gpus=job['next_level_gpu'], 
        simulation=True, future_free_gpus=tmp_future_free_gpus)
    if job['num_gpu'] == 0:
        job['marginal_gputime'] = sys.maxsize
    del tmp_future_free_gpus

def allocate_free_gpus(cur_time):
    total_gpu_used = 0
    for job in JOBS.runnable_jobs:
        #marginal_throughput(job, cur_time)
        job['next_level_gpu'] = utils.get_next_level(job)
        total_gpu_used += job['num_gpu']
        print("job", job['job_id'], job['num_gpu'])
    allocatable_gpus = CLUSTER.num_gpu - total_gpu_used
    assert allocatable_gpus >= 0
    print()
    print(allocatable_gpus, "gpus left")
    print("CLUSTER.future_free_gpus", CLUSTER.future_free_gpus)
    if CLUSTER.future_free_gpus[cur_time] != allocatable_gpus:
        for r_job in JOBS.runnable_jobs:
            print("^^^", r_job['job_idx'], r_job['allocations'])
    assert CLUSTER.future_free_gpus[cur_time] == allocatable_gpus
    # sort and find the ones to allocate more GPUS, change end time of these jobs
    if allocatable_gpus == 0:
        return
    for job in JOBS.runnable_jobs:
        if job['next_level_gpu'] - job['num_gpu'] > allocatable_gpus:
            print("not enough GPU for job", job['job_id'], job['next_level_gpu'] - job['num_gpu'])
            #continue
        if job['next_level_gpu'] == job['num_gpu']:
            print("Cannot scale up for job", job['job_id'])
            #continue
        get_marginal(job, allocatable_gpus, cur_time)
        
    #JOBS.running_jobs.sort(key = lambda e:e.__getitem__('marginal_throughput'), reverse=True)
    JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('marginal_gputime'), reverse=True)
    
    allocated_last_round = True
    while allocatable_gpus > 0 and allocated_last_round:
        allocated_last_round = False
        for eachjob in JOBS.runnable_jobs:
            if eachjob['next_level_gpu'] - eachjob['num_gpu'] > allocatable_gpus:
                print("not enough GPU for job", eachjob['job_id'], eachjob['next_level_gpu'] - eachjob['num_gpu'])
                continue
            if eachjob['next_level_gpu'] == eachjob['num_gpu']:
                print("Cannot scale up for job", eachjob['job_id'])
                continue
            # allocate
            print("\nscale job[", eachjob['job_idx'], "] up to", eachjob['next_level_gpu'])
            old_gpu_num = eachjob['num_gpu']

            del CLUSTER.future_free_gpus
            CLUSTER.future_free_gpus = utils.merge_dict(eachjob['next_level_future_gpus'])
            print("CLUSTER.future_free_gpus", CLUSTER.future_free_gpus)
            eachjob['allocations'] = eachjob['next_level_allocation']
            eachjob['num_gpu'] = eachjob['next_level_gpu']
            eachjob['old_end_time'] = eachjob['end_time']
            eachjob['end_time'] = eachjob['next_level_endtime']
            if eachjob in JOBS.pending_jobs:
                JOBS.remove_from_pending(eachjob, cur_time)
            JOBS.change_job_end_event(eachjob)
            del eachjob['next_level_endtime'], eachjob['next_level_gpu'], eachjob['next_level_allocation']
            allocatable_gpus -= (eachjob['num_gpu'] - old_gpu_num)
            assert CLUSTER.future_free_gpus[cur_time] == allocatable_gpus
            allocated_last_round = True
            #marginal_throughput(eachjob, cur_time)
            eachjob['next_level_gpu'] = utils.get_next_level(eachjob)
            #if eachjob['next_level_gpu'] - eachjob['num_gpu'] > 0:
            #    get_marginal(eachjob, allocatable_gpus, cur_time)
            break
        for job in JOBS.runnable_jobs:
            """if job['next_level_gpu'] - job['num_gpu'] > allocatable_gpus:
                print("not enough GPU for job", job['job_id'], job['next_level_gpu'] - job['num_gpu'])
                continue"""
            if job['next_level_gpu'] - job['num_gpu'] > 0:
                get_marginal(job, allocatable_gpus, cur_time)

        #JOBS.running_jobs.sort(key = lambda e:e.__getitem__('marginal_throughput'), reverse=True)
        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('marginal_gputime'), reverse=True)

    print("after allocation, CLUSTER.future_free_gpus", CLUSTER.future_free_gpus)


def estimate_all_jobs(job_list, cur_time, record_old_end_time=True):
    del CLUSTER.future_free_gpus
    CLUSTER.future_free_gpus = {cur_time: CLUSTER.num_gpu}
    result = True
    job_list.sort(key = lambda e:e.__getitem__('ddl'))
    for job in job_list:
        if record_old_end_time:
            job['old_end_time'] = job['end_time']
        if 'node_set' in job:
            CLUSTER.release_job_res(job, end=False)
    for job in job_list:
        if not ef_sim_allocation(job, cur_time):
            result = False
    return result

def EDF_estimate_all_jobs(job_list, cur_time):
    del CLUSTER.future_free_gpus
    CLUSTER.future_free_gpus = {cur_time: CLUSTER.num_gpu}
    value = True
    job_list.sort(key = lambda e:e.__getitem__('ddl'))
    for eachjob in job_list:
        if not edf_sim_allocation(eachjob, cur_time):
            value = False
    return value


def get_available_gpu(event_time, future_free_gpus=None):
    if future_free_gpus is None:
        future_free_gpus = CLUSTER.future_free_gpus
    if event_time in future_free_gpus:
        return future_free_gpus[event_time]
    return_value = CLUSTER.num_gpu
    for each_event_time in future_free_gpus:
        if each_event_time <= event_time:
            return_value = future_free_gpus[each_event_time]
    return return_value


'''
Gandiva scheduler assumption
''' 
def gandiva_sim_jobs(gputime=False, solve_starvation=0):
    '''
    new jobs are added to the end of the ending queue
    but any fit job should be executed in fifo order
    '''
    global global_lock, this_round_begin_time, fast_forward_permission
    global schedule_count
    cur_time = JOBS.job_events[0]['time']
    node_release = False
    time_diff = 0
    last_reschedule_time = 0
    while (len(JOBS.job_events) + len(JOBS.pending_jobs) + len(JOBS.running_jobs))> 0:
        # if len(JOBS.job_events) == 0:
        #     break
        new_job_start = False
        CLUSTER.gandiva_node_set_adjust(cur_time, JOBS, LOG)
        print("%d-%d, %d, %d " % (cur_time, len(JOBS.job_events), len(JOBS.pending_jobs), len(JOBS.running_jobs)))
        #update job progress for end_jobs
        node_release = CLUSTER.time_slicing_execute(cur_time, JOBS, LOG, time_diff)
        for r_job in JOBS.runnable_jobs:
            r_job['overhead'] = 0

        #get new start job
        event = utils.search_dict_list(JOBS.job_events, 'time', cur_time)
        event_time = cur_time
        if event != None:
            #for new-start jobs, try to start
            for s_job in event['start_jobs']:
                ret = try_get_job_res(s_job)
                if ret == False:
                    JOBS.move_to_pending(s_job)
                else:
                    s_job['start_time'] = cur_time
                    JOBS.running_jobs.append(s_job)
                    if 'best_effort' not in s_job or int(s_job['best_effort']) != 1:
                        JOBS.num_accepted_job += 1
                    utils.print_fn('----job[%d] starts' % s_job['job_idx'])

            #remove time_event
            JOBS.job_events.remove(event)

        if node_release: 
            for p_job in JOBS.pending_jobs:
                ret = try_get_job_res(p_job)
                if ret == True:
                    JOBS.remove_from_pending(p_job, cur_time)
                    p_job['start_time'] = cur_time
                    #JOBS.running_jobs.append(p_job)
                    utils.print_fn('----job[%d] starts from pending' % p_job['job_idx'])
                    new_job_start = True

        node_release = False
        
        # add node_set information to job_dict
        for node_set in CLUSTER.node_g:
            for each_set in CLUSTER.node_g[node_set]:
                concurrency = 0
                for each_job in each_set['jobs']:
                    concurrency = concurrency + 1
                    if concurrency <= each_set['capacity']:
                        each_job['node_set'] = each_set
                    else:
                        each_job['node_set'] = None
        #if event != None or new_job_start:
        LOG.scheduling_result(event_time)

        # change from 10 to time_slot
        if len(JOBS.job_events) <= 0:
            cur_time = cur_time + FLAGS.scheduling_slot
            time_diff = FLAGS.scheduling_slot
        else:
            if FLAGS.simulation:
                restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
                schedule_count += 1
                for r_job in JOBS.running_jobs:
                    if r_job['num_gpu'] > 0:
                        r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart)
                    else:
                        r_job['overhead'] = 0
            else:
                global_lock.acquire()
                if not fast_forward_permission:
                    cur_time = last_reschedule_time
                    last_reschedule_time = event_time
                this_round_begin_time = math.ceil(time.time())
                global_lock.release()
                get_ef_input_no_overlap([], this_round_begin_time)
                JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
                #start time of next job
                #next_s_time = JOBS.job_events[0]['time']
                next_s_time = cur_time + FLAGS.scheduling_slot
                for each_event in JOBS.job_events:
                    if len(each_event['start_jobs']) == 0:
                        continue
                    next_s_time = max(each_event['time'], next_s_time)
                    break

                while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_s_time):
                    time.sleep(1)
                    cur_time += 1
                if not fast_forward_permission:
                    print("ATTENTION!!!, cur_time", cur_time)
                    update_overhead()

                for r_job in JOBS.running_jobs:
                    if r_job['job_idx'] not in job_stable:
                        if r_job['job_idx'] in this_round_running_jobs:
                            continue
                    r_job['old_end_time'] = r_job['end_time']
                    #if r_job['job_idx'] not in job_stable:
                    #    # not all jobs have finished scaling, but they have to be rescheduled
                    #    r_job['overhead'] = next_s_time - event_time
                    if r_job['job_idx'] in this_round_running_jobs:
                        r_job['end_time'] += r_job['overhead']

            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            next_e_time = JOBS.job_events[0]['time']
            
            while int(next_e_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                del JOBS.job_events[1]

            next_e_time = JOBS.job_events[0]['time']
            assert next_e_time >= cur_time
            time_diff = int(next_e_time - cur_time)
            cur_time = next_e_time
            LOG.checkpoint(event_time)

    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def one_queue_edf_sim_jobs():
    '''
    run jobs in edf order, without access control;
    jobs are sorted by ddl
    '''
    global this_round_begin_time, fast_forward_permission, global_lock
    global schedule_count
    idle_round = 0
    time_diff = 0
    cur_time = JOBS.job_events[0]['time']
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            if idle_round > 1:
                utils.print_fn("This cluster is not large enough to run the job")
                print(JOBS.pending_jobs)
                break
            idle_round += 1
        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        print()
        print(cur_time)
        JOBS.run_all_jobs(time_diff, cur_time)
        for r_job in JOBS.runnable_jobs:
            r_job['overhead'] = 0
        new_job_flag = False

        if len(JOBS.job_events) > 0:
            event = JOBS.job_events[0]
            event_time = event['time']
            # utils.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
            #for ending jobs, release gpu
            has_ejob = False
            for e_job in event['end_jobs']:
                if 'node_set' not in e_job:
                    print(e_job)
                #job completes
                CLUSTER.release_job_res(e_job)
                if e_job['end_time'] > e_job['ddl']:
                    utils.print_fn('----job[%d]\'s DDL request is not satisfied. Declined.' % e_job['job_idx'])
                    print(e_job['end_time'], "v.s.", e_job['ddl'])
                    JOBS.move_to_declined(e_job)
                    JOBS.num_accepted_job -= 1
                    #input()
                else:
                    print("ends at", cur_time, e_job['end_time'], "ddl", e_job['ddl'])
                JOBS.remove_running(e_job)
                LOG.job_complete(e_job, event_time)
                has_ejob = True

            #for new-start jobs, try to start
            available_gpu = CLUSTER.check_free_gpu()
            CLUSTER.future_free_gpus = {cur_time:CLUSTER.num_gpu}
            if len(event['start_jobs']) > 0:
                new_job_flag = True
                EDF_estimate_all_jobs(JOBS.runnable_jobs + event['start_jobs'], event_time)
                for s_job in event['start_jobs']:
                    JOBS.move_to_pending(s_job) #add into pending list
                    available_gpu -= s_job['new_allocations'][event_time]
        
        if not new_job_flag:
            EDF_estimate_all_jobs(JOBS.runnable_jobs, event_time)
        for r_job in JOBS.running_jobs:
            if r_job['num_gpu'] > 0:
                CLUSTER.release_job_res(r_job, end=False)
        run_jobs = []
        for r_job in JOBS.runnable_jobs:
            if 'new_allocations' not in r_job:
                print(r_job)
            r_job['allocations'] = r_job['new_allocations']
            del r_job['new_allocations']
            r_job['num_gpu'] = r_job['allocations'][event_time]
                
            r_job['old_end_time'] = r_job['end_time']
            r_job['end_time'] = r_job['new_end_time']
            del r_job['new_end_time']
            if r_job in JOBS.running_jobs:
                if r_job['num_gpu'] > 0:
                    #ret = try_get_job_res(r_job)
                    #assert ret
                    run_jobs.append(r_job)
                JOBS.change_job_end_event(r_job)
            else:
                assert r_job in JOBS.pending_jobs
                if r_job['num_gpu'] > 0:
                    JOBS.get_network_load(r_job)
                    #ret = try_get_job_res(r_job)
                    #assert ret
                    run_jobs.append(r_job)
                    JOBS.remove_from_pending(r_job, event_time)       
                    JOBS.add_job_end_event(r_job)
                    utils.print_fn('----job[%d] starts from pending' % r_job['job_idx'])
        run_jobs.sort(key = lambda e:e['num_gpu'], reverse=True)
        for r_job in run_jobs:
            ret = try_get_job_res(r_job)
            assert ret


        LOG.scheduling_result(event_time)
        JOBS.job_events.pop(0)

        #remove time_event
        if len(JOBS.job_events) > 0:
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            
        if len(JOBS.job_events) <= 0:
            time_diff = FLAGS.scheduling_slot
        else:
            if FLAGS.simulation:
                restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
                schedule_count += 1
                if restart:
                    LOG.cache = list()
                for r_job in JOBS.running_jobs:
                    if r_job['num_gpu'] > 0:
                        r_job['overhead'] = 0
                        #r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart, r_job['in_cache'])
                        r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] += r_job['overhead']
                        JOBS.change_job_end_event(r_job)
                    else:
                        r_job['overhead'] = 0
            else:
                global_lock.acquire()
                this_round_begin_time = math.ceil(time.time())
                global_lock.release()
                get_ef_input_no_overlap(event['end_jobs'], this_round_begin_time)
                JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
                #start time of next job
                next_s_time = cur_time + FLAGS.scheduling_slot
                for each_event in JOBS.job_events:
                    if len(each_event['start_jobs']) == 0:
                        continue
                    next_s_time = max(each_event['time'], next_s_time)
                    break
                while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_s_time):
                    time.sleep(1)
                    cur_time += 1
                if not fast_forward_permission:
                    print("ATTENTION!!!, cur_time", cur_time)
                    update_overhead()

                for r_job in JOBS.running_jobs:
                    if r_job['job_idx'] not in job_stable:
                        if r_job['job_idx'] in this_round_running_jobs:
                            continue
                    r_job['old_end_time'] = r_job['end_time']
                    #if r_job['job_idx'] not in job_stable:
                    #    # not all jobs have finished scaling, but they have to be rescheduled
                    #    r_job['overhead'] = next_s_time - event['time']
                    r_job['end_time'] += r_job['overhead']
                    JOBS.change_job_end_event(r_job)

            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            next_e_time = JOBS.job_events[0]['time']
            while int(next_e_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                JOBS.job_events[0]['end_jobs'].extend(
                    JOBS.job_events[1]['end_jobs'])
                del JOBS.job_events[1]

            next_e_time = JOBS.job_events[0]['time']
            time_diff = int(next_e_time - cur_time)
            cur_time = next_e_time
            LOG.checkpoint(event_time)
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def one_queue_edf_sim_jobs_access_control():
    '''
    run jobs in edf order, with access control;
    jobs are sorted by ddl
    '''
    global this_round_begin_time, fast_forward_permission, global_lock
    global schedule_count
    idle_round = 0
    time_diff = 0
    cur_time = JOBS.job_events[0]['time']
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            if idle_round > 1:
                utils.print_fn("This cluster is not large enough to run the job")
                print(JOBS.pending_jobs)
                break
            idle_round += 1
        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        print()
        print(cur_time)
        JOBS.run_all_jobs(time_diff, cur_time)
        for r_job in JOBS.runnable_jobs:
            r_job['overhead'] = 0
        for r_job in JOBS.runnable_jobs:
            r_job['old_end_time'] = r_job['end_time']
            if 'node_set' in r_job:
                CLUSTER.release_job_res(r_job, end=False)
            #else:
            #    assert r_job['num_gpu'] == 0
        if CLUSTER.check_free_gpu() != CLUSTER.num_gpu:
            for eachjob in JOBS.runnable_jobs:
                print(eachjob['job_idx'], eachjob['num_gpu'])
            print(CLUSTER.check_free_gpu(), CLUSTER.num_gpu)
        assert CLUSTER.check_free_gpu() == CLUSTER.num_gpu
        

        if len(JOBS.job_events) > 0:
            event = JOBS.job_events[0]
            event_time = event['time']
            # utils.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
            #for ending jobs, release gpu
            has_ejob = False
            for e_job in event['end_jobs']:
                #job completes
                print("job", e_job['job_idx'], "ends at", cur_time, e_job['end_time'], "ddl", e_job['ddl'])
                #CLUSTER.release_job_res(e_job)
                JOBS.remove_running(e_job)
                LOG.job_complete(e_job, event_time)
                #assert cur_time <= e_job['ddl']
                if e_job['end_time'] > e_job['ddl']:
                    JOBS.move_to_declined(e_job)
                    JOBS.num_accepted_job -= 1
                has_ejob = True

            #for new-start jobs, try to start
            available_gpu = CLUSTER.check_free_gpu()
            #CLUSTER.future_free_gpus = {cur_time:CLUSTER.num_gpu}
            new_job_flag = False
            if len(event['start_jobs']) > 1:
                print(event['start_jobs'])
                #input()
            for s_job in event['start_jobs']:
                if not EDF_estimate_all_jobs(JOBS.runnable_jobs + [s_job], event_time):
                    utils.print_fn('----job[%d]\'s DDL request cannot be satisfied. Declined.' % s_job['job_idx'])
                    JOBS.move_to_declined(s_job, remove_from_pending=False)
                    #input()
                else:
                    JOBS.move_to_pending(s_job) #add into pending list
                    new_job_flag = True
                    available_gpu -= s_job['new_allocations'][event_time]
                    assert s_job in JOBS.pending_jobs
                    for r_job in JOBS.running_jobs:

                        if 'new_allocations' not in r_job:
                            print(r_job)
                        r_job['allocations'] = r_job['new_allocations']
                        del r_job['new_allocations']
                        r_job['num_gpu'] = r_job['allocations'][event_time]
                
                        r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] = r_job['new_end_time']
                        del r_job['new_end_time']
                        assert r_job['end_time'] <= r_job['ddl']
                        if r_job['job_idx'] == 9:
                            print("!!!from", r_job['old_end_time'], "to", r_job['end_time'])
                            #input()
                        JOBS.change_job_end_event(r_job)
                    new_start_job = list()
                    for p_job in JOBS.pending_jobs:
                        if 'new_allocations' not in p_job:
                            print(p_job)
                        p_job['allocations'] = p_job['new_allocations']
                        del p_job['new_allocations']
                        p_job['num_gpu'] = p_job['allocations'][event_time]
                
                        p_job['old_end_time'] = p_job['end_time']
                        p_job['end_time'] = p_job['new_end_time']
                        if p_job['end_time'] > p_job['ddl']:
                            print(p_job)
                        assert p_job['end_time'] <= p_job['ddl']
                        del p_job['new_end_time']
                        if p_job['num_gpu'] > 0:
                            new_start_job.append(p_job)  
                    for ns_job in new_start_job:
                        JOBS.get_network_load(ns_job)
                        utils.print_fn('----job[%d] starts from pending' % ns_job['job_idx'])
                        JOBS.remove_from_pending(ns_job, event_time)       
                        JOBS.add_job_end_event(ns_job)
        
        if new_job_flag:
            for r_job in JOBS.runnable_jobs:
                if r_job in JOBS.running_jobs:
                    if r_job['num_gpu'] > 0:
                        ret = try_get_job_res(r_job)
                        assert ret
                else:
                    assert r_job in JOBS.pending_jobs
                    if r_job['num_gpu'] > 0:
                        ret = try_get_job_res(r_job)
                        assert ret
             
        else:
            EDF_estimate_all_jobs(JOBS.runnable_jobs, event_time)
            for r_job in JOBS.runnable_jobs:
                if cur_time > r_job['ddl']:
                    print(r_job)
                #assert cur_time <= r_job['ddl'] cannot be guaranteed because of overhead
                r_job['allocations'] = r_job['new_allocations']
                del r_job['new_allocations']
                r_job['num_gpu'] = r_job['allocations'][event_time]

                if r_job['num_gpu'] > 0:
                    ret = try_get_job_res(r_job)
                    assert ret
                    if r_job in JOBS.pending_jobs:
                        JOBS.get_network_load(r_job)
                        JOBS.remove_from_pending(r_job, event_time)       
                        JOBS.add_job_end_event(r_job)
                    
                        utils.print_fn('----job[%d] starts from pending' % r_job['job_idx'])
                        r_job['end_time'] = r_job['new_end_time']
                        del r_job['new_end_time']
                        #assert r_job['end_time'] <= r_job['ddl']
                        JOBS.change_job_end_event(r_job)
                    else:
                        #if len(event['start_jobs']) == 0:
                        #r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] = r_job['new_end_time']
                        del r_job['new_end_time']
                        #assert r_job['end_time'] <= r_job['ddl']
                        JOBS.change_job_end_event(r_job)

        LOG.scheduling_result(event_time)
        JOBS.job_events.pop(0)

        #remove time_event
        if len(JOBS.job_events) > 0:
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            
        if len(JOBS.job_events) <= 0:
            time_diff = 10
        else:
            if FLAGS.simulation:
                restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
                schedule_count += 1
                if restart:
                    LOG.cache = list()
                for r_job in JOBS.running_jobs:
                    if r_job['num_gpu'] > 0:
                        r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart, r_job['in_cache'])
                        r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] += r_job['overhead']
                        JOBS.change_job_end_event(r_job)
                    else:
                        r_job['overhead'] = 0
            else:
                global_lock.acquire()
                this_round_begin_time = math.ceil(time.time())
                global_lock.release()
                get_ef_input_no_overlap(event['end_jobs'], this_round_begin_time)
                JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
                #start time of next job
                next_s_time = cur_time + FLAGS.scheduling_slot
                for each_event in JOBS.job_events:
                    if len(each_event['start_jobs']) == 0:
                        continue
                    next_s_time = max(each_event['time'], next_s_time)
                    break
                while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_s_time):
                    time.sleep(1)
                    cur_time += 1
                if not fast_forward_permission:
                    print("ATTENTION!!!, cur_time", cur_time)
                    update_overhead()

                for r_job in JOBS.running_jobs:
                    if r_job['job_idx'] not in job_stable:
                        if r_job['job_idx'] in this_round_running_jobs:
                            continue
                    r_job['old_end_time'] = r_job['end_time']
                    #if r_job['job_idx'] not in job_stable:
                    #    # not all jobs have finished scaling, but they have to be rescheduled
                    #    r_job['overhead'] = next_s_time - event['time']
                    r_job['end_time'] += r_job['overhead']
                    if r_job['job_idx'] == 9:
                        print("from", r_job['old_end_time'], "to", r_job['end_time'])
                    JOBS.change_job_end_event(r_job)

            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            next_e_time = JOBS.job_events[0]['time']
            while int(next_e_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                JOBS.job_events[0]['end_jobs'].extend(
                    JOBS.job_events[1]['end_jobs'])
                del JOBS.job_events[1]

            next_e_time = JOBS.job_events[0]['time']
            time_diff = int(next_e_time - cur_time)
            cur_time = next_e_time

            LOG.checkpoint(event_time)
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def get_ef_input_no_overlap(end_jobs, actual_time):
    global job_stable, fast_forward_permission, MASTER_PORT, commands
    global last_round_running_jobs, this_round_running_jobs, trainers_to_kill
    global global_lock, global_ready_lock
    global job_to_be_killed, schedule_count
    global last_round_gpu_allocations, gpu_allocations
    global_lock.acquire()
    # if there is need to reschedule
    return_flag = True
    for job in JOBS.running_jobs:
        if job['num_gpu'] == 0 or job['node_set'] is None:
            if job['job_idx'] in last_round_running_jobs:
                return_flag = False
                break
            continue
        if job['num_gpu'] < CLUSTER.num_gpu_p_node:
            if job['job_idx'] in last_round_running_jobs:
                if 'num_gpu' in last_round_running_jobs[job['job_idx']] and last_round_running_jobs[job['job_idx']]['num_gpu'] == job['num_gpu']:
                    # do not need to update in this round
                    continue
                else:
                    return_flag = False 
                    break
            else:
                return_flag = False 
                break
        else:
            if job['job_idx'] in last_round_running_jobs:
                # same number of nodes
                if len(last_round_running_jobs[job['job_idx']]['worker_id']) == len(job['node_set']['nodes']):
                    continue
                else:
                    return_flag = False 
                    break
            else:
                return_flag = False 
                break

    if return_flag:
        fast_forward_permission = True
        global_lock.release()
        return

    restart_trainers = (schedule_count % FLAGS.restart_threshold == 0) and schedule_count != 0
    # restart if jobs didn't restart successfully last round
    if not restart_trainers:
        restart_trainers = job_to_be_killed
    del this_round_running_jobs
    this_round_running_jobs = dict()
    if restart_trainers:
        scheduler_rpc_client.schedule('RE')
    else:
        for e_job in end_jobs:
            assert e_job['job_idx'] in last_round_running_jobs
            for each_worker in last_round_running_jobs[e_job['job_idx']]['worker_id']:
                command = ' '.join(["K", 
                    str(each_worker), str(e_job['job_idx'])])
                print(command)
                scheduler_rpc_client.schedule(command)
                job_to_be_killed = True
            del last_round_running_jobs[e_job['job_idx']]
    
    gpu_allocations = [[0 for gpu in range(CLUSTER.num_gpu_p_node)] for _ in range(CLUSTER.num_node)]
    del job_stable
    job_stable = dict()
    commands = []

    # new jobs
    for job in JOBS.running_jobs:
        if job['num_gpu'] == 0 or job['node_set'] is None:
            continue
        cmd = 'R'
        if job['num_gpu'] >= CLUSTER.num_gpu_p_node:
            for i in range(len(job['node_set']['nodes'])):
                compressed_gpu_list = 0
                for j in range(len(gpu_allocations[job['node_set']['nodes'][i].id])):
                    gpu_allocations[job['node_set']['nodes'][i].id][j] = 1
                    compressed_gpu_list += (1 << j)
                MASTER_PORT += 1
                command = ' '.join([cmd, str(job['node_set']['nodes'][i].id), job['model_name'], 
                    str(job['batch_size']), str(job['job_idx']), str(min(job['num_gpu'], CLUSTER.num_gpu_p_node)), str(len(job['node_set']['nodes'])), 
                    str(i), '127.0.0.1', str(MASTER_PORT), str(compressed_gpu_list), str(int(job['iter_left'])), str(actual_time)])
                print(command)
                if job['job_idx'] not in this_round_running_jobs:
                    this_round_running_jobs[job['job_idx']] = {'worker_id':[]}
                this_round_running_jobs[job['job_idx']]['worker_id'].append(str(job['node_set']['nodes'][i].id))
                if job['job_idx'] not in job_stable:
                    job_stable[job['job_idx']] = 0
                fast_forward_permission = False
                #scheduler_rpc_client.schedule(command)
                commands.append(command)
        else:
            node_id = job['node_set']['nodes'][0].id
            if job['job_idx'] in this_round_running_jobs:
                continue

            allocated_gpu = 0
            compressed_gpu_list = 0
            for i in range(len(gpu_allocations[node_id])):
                if gpu_allocations[node_id][i] == 1:
                    continue
                allocated_gpu += 1
                gpu_allocations[node_id][i] = 1
                compressed_gpu_list += (1 << i)
                if allocated_gpu == job['num_gpu']:
                    break
            MASTER_PORT += 1
            command = ' '.join([cmd, str(node_id), job['model_name'], str(job['batch_size']), str(job['job_idx']), 
                str(min(job['num_gpu'], CLUSTER.num_gpu_p_node)), str(len(job['node_set']['nodes'])), '0', '127.0.0.1', str(MASTER_PORT), str(compressed_gpu_list), 
                str(int(job['iter_left'])), str(actual_time)])
            print(command)
            tmp_dict = dict()
            tmp_dict['worker_id'] = [str(node_id)]
            tmp_dict['num_gpu'] = job['num_gpu']
            tmp_dict['compressed_gpu_list'] = compressed_gpu_list
            this_round_running_jobs[job['job_idx']] = tmp_dict
            if job['job_idx'] not in job_stable:
                job_stable[job['job_idx']] = 0
            fast_forward_permission = False
            #scheduler_rpc_client.schedule(command)
            commands.append(command)
    #TODO: let master stop old jobs for on-the-fly elastic trainers
    
    global_ready_lock.acquire()
    if len(this_round_running_jobs) > 0:
        trainers_to_kill = {}
    for job in this_round_running_jobs:
        trainers_to_kill[job] = []
        if 'num_gpu' in this_round_running_jobs[job]:
            for each_gpu in utils.fetch_GPU_list_to_int(this_round_running_jobs[job]['compressed_gpu_list']):
                if last_round_gpu_allocations[int(this_round_running_jobs[job]['worker_id'][0])][each_gpu] == 0:
                    continue
                trainers_to_kill[job].append(utils.get_global_rank(
                    this_round_running_jobs[job]['worker_id'][0],
                    each_gpu, CLUSTER.num_gpu_p_node))
        else:   
            for each_worker in this_round_running_jobs[job]['worker_id']:
                for each_gpu in range(CLUSTER.num_gpu_p_node):
                    if last_round_gpu_allocations[int(each_worker)][each_gpu] == 0:
                        continue
                    trainers_to_kill[job].append(utils.get_global_rank(
                    each_worker, each_gpu, CLUSTER.num_gpu_p_node))
    
    print("$$$ in no overlap, trainers to kill", trainers_to_kill)
    print("$$$ last_round_running_jobs", last_round_running_jobs)
    print("$$$ this_round_running_jobs", this_round_running_jobs)

    if not restart_trainers:
        for job in last_round_running_jobs:
            for each_worker in last_round_running_jobs[job]['worker_id']:
                command = 'K ' + str(each_worker) + ' ' + str(job)
                scheduler_rpc_client.schedule(command)
                job_to_be_killed = True
    else:
        job_to_be_killed = False
    
    if not job_to_be_killed:
        # run all commands
        for command in commands:
            scheduler_rpc_client.schedule(command)
        scheduler_rpc_client.schedule('F')
        scheduler_rpc_client.schedule('T')
        last_round_gpu_allocations = gpu_allocations
    fast_forward_permission = (len(commands) == 0)

    global_ready_lock.release()
    del last_round_running_jobs
    last_round_running_jobs = this_round_running_jobs
    #last_round_gpu_allocations = gpu_allocations
    schedule_count += 1
    global_lock.release()


def update_overhead():
    global this_round_running_jobs, fast_forward_permission, global_lock
    global_lock.acquire()
    if fast_forward_permission:
        global_lock.release()
        return
    
    for each_job in JOBS.running_jobs:
        if each_job['num_gpu'] == 0:
            continue
        if each_job['job_idx'] not in job_stable:
            if each_job['job_idx'] in this_round_running_jobs:
                each_job['overhead'] = 0
                continue
            each_job['overhead'] = FLAGS.scheduling_slot
        if job_stable[each_job['job_idx']] == 0:
            each_job['overhead'] = FLAGS.scheduling_slot
    global_lock.release()


def ef_sim_jobs_access_control():
    '''
    run jobs with elasticity with access control;
    new jobs are added to the end of the pending queue;
    unsatisfiable jobs are declined.
    '''
    global this_round_begin_time, fast_forward_permission, global_lock
    global schedule_count
    cur_time = JOBS.job_events[0]['time']
    node_release = False
    time_diff = 0
    while (len(JOBS.job_events) + len(JOBS.pending_jobs) + len(JOBS.running_jobs))> 0:
        # if len(JOBS.job_events) == 0:
        #     break
        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        print()
        print("%d-%d, %d, %d " % (cur_time, len(JOBS.job_events), len(JOBS.pending_jobs), len(JOBS.running_jobs)))
        #update job progress for end_jobs
        JOBS.run_all_jobs(time_diff, cur_time)
        for r_job in JOBS.runnable_jobs:
            r_job['overhead'] = 0
        
        #get new start job
        event = utils.search_dict_list(JOBS.job_events, 'time', cur_time)
        if event != None:
            JOBS.job_events.remove(event)
            if len(JOBS.pending_jobs) == 0 and len(JOBS.running_jobs) == 0:
                if len(event['end_jobs']) == 0 and len(event['start_jobs']) == 0:
                    continue
            for e_job in event['end_jobs']:
                if 'node_set' in e_job:
                    #remove from migratable jobs, if it's there
                    JOBS.remove_migratable(e_job)
                    #job completes
                    JOBS.remove_running(e_job)
                    CLUSTER.release_job_res(e_job)
                    LOG.job_complete(e_job, cur_time)
                    print("ends at", cur_time, e_job['end_time'], "ddl", e_job['ddl'])
                    if e_job['real_end_time'] > e_job['ddl']:
                        print(e_job)
                        #assert e_job['end_time'] <= e_job['ddl']
                        JOBS.move_to_declined(e_job)
                        JOBS.num_accepted_job -= 1

            for r_job in JOBS.runnable_jobs:
                r_job['old_end_time'] = r_job['end_time']
                r_job['old_allocations'] = r_job['allocations']

            new_job = False
            if len(event['start_jobs']) > 0:
                CLUSTER.old_future_free_gpu = copy.deepcopy(CLUSTER.future_free_gpus)
                if not estimate_all_jobs(JOBS.runnable_jobs + event['start_jobs'], cur_time):
                    for s_job in event['start_jobs']:
                        if 'best_effort' in s_job and int(s_job['best_effort']) == 1:
                            JOBS.get_network_load(s_job)
                            JOBS.move_to_pending(s_job)
                        else:
                            utils.print_fn('----job[%d]\'s DDL request cannot be satisfied. Declined.' % s_job['job_idx'])
                            JOBS.move_to_declined(s_job)
                    if not new_job:
                        CLUSTER.future_free_gpus = CLUSTER.old_future_free_gpu
                        del CLUSTER.old_future_free_gpu
                    # only one new job at a time
                else:
                    new_job = True
                    for s_job in event['start_jobs']:
                        JOBS.get_network_load(s_job)
                        JOBS.move_to_pending(s_job)

            if not new_job:
                estimate_all_jobs(JOBS.runnable_jobs, cur_time, record_old_end_time=(len(event['start_jobs']) == 0))
            del CLUSTER.future_free_gpus
            CLUSTER.future_free_gpus = {cur_time: CLUSTER.num_gpu}
            JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('ddl'))
            for r_job in JOBS.runnable_jobs:
                r_job['num_gpu'] = r_job['allocations'][cur_time]
                JOBS.change_job_end_event(r_job)
                r_job['old_end_time'] = r_job['end_time']
                ef_sim_allocation(r_job, cur_time)
                if 'allocations' in r_job:
                    r_job['old_allocations'] = copy.deepcopy(r_job['allocations'])            
            
            #remove time_event


        if CLUSTER.check_free_gpu() > 0:
            #for pending jobs, try to start
            new_start_list = list()
            for p_job in JOBS.pending_jobs:
                pend_flag = False
                allocation_time = cur_time + 1
                for allocation_time in p_job['allocations']:
                    if p_job['allocations'][allocation_time] == 0:
                        continue
                    if allocation_time > cur_time:
                        pend_flag = True
                    break
                p_job['num_gpu'] = p_job['allocations'][cur_time]
                #if pend_flag:
                #    JOBS.add_job_scale_event(p_job, allocation_time)
                CLUSTER.free_gpu = CLUSTER.check_free_gpu()
                if p_job['num_gpu'] <= CLUSTER.free_gpu and not pend_flag:
                    new_start_list.append(p_job)
                    print("pending job", p_job['job_id'], p_job['allocations'])
            
            for ns_job in new_start_list:
                JOBS.remove_from_pending(ns_job, cur_time)
                JOBS.add_job_end_event(ns_job)
                ## add next arrangement event
                #if job is migratable, add into migratable job list
                JOBS.add_migratable(ns_job)
                # JOBS.read_job_info(p_job['job_idx'])
                utils.print_fn('----job[%d] starts from pending' % ns_job['job_idx'])
            
            for r_job in JOBS.runnable_jobs:
                print(r_job['job_idx'], r_job['allocations'], r_job['end_time'])
            # allocate free GPUs
            allocate_free_gpus(cur_time)
            #CLUSTER.status_update()

            first_scale_event_time = sys.maxsize
            JOBS.runnable_jobs.sort(key = lambda e:e['num_gpu'], reverse=True)
            for job in JOBS.runnable_jobs:
                # add scaling event for next allocation
                job['allocations'] = OrderedDict(sorted(job['allocations'].items(), key=lambda t: t[0]))
                for each_allocation_time in job['allocations']:
                    if each_allocation_time > cur_time:
                        #JOBS.add_job_scale_event(job, each_allocation_time)
                        if each_allocation_time < first_scale_event_time:
                            first_scale_event_time = each_allocation_time
                        break
                if job["num_gpu"] >= CLUSTER.num_gpu_p_node:
                    ret = try_get_job_res(job)
                    assert ret
                    if not ret:
                        print('ERROR when allocating for job[%d]' % job['job_idx'])
            for node_group in reversed(list(CLUSTER.node_g.keys())):
                if node_group >= CLUSTER.num_gpu_p_node:
                    continue
                for job in JOBS.runnable_jobs:
                    if job["num_gpu"] == node_group:
                        ret = try_get_job_res(job)
                        if not ret:
                            print('ERROR when allocating for job[%d]' % job['job_idx'])
                            print(CLUSTER.node_g)
                        assert ret
            
        LOG.scheduling_result(cur_time)
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        if len(JOBS.job_events) > 0:
            if first_scale_event_time < JOBS.job_events[0]['time']:
                JOBS.add_job_scale_event_new(first_scale_event_time)
                JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        else:
            JOBS.add_job_scale_event_new(first_scale_event_time)
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))

        #input()

        if len(JOBS.job_events) <= 0:
            cur_time = cur_time + 10
            time_diff = 10
        else:
            if FLAGS.simulation:
                restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
                schedule_count += 1
                if restart:
                    LOG.cache = list()
                for r_job in JOBS.runnable_jobs:
                    if r_job['num_gpu'] > 0:
                        r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart, r_job['in_cache'])
                        #r_job['overhead'] = 0
                        r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] += r_job['overhead']
                        JOBS.change_job_end_event(r_job)
                    else:
                        r_job['overhead'] = 0
            else:
                global_lock.acquire()
                this_round_begin_time = math.ceil(time.time())
                global_lock.release()
                get_ef_input_no_overlap(event['end_jobs'], this_round_begin_time)
                JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
                #start time of next job
                next_s_time = cur_time + FLAGS.scheduling_slot
                for each_event in JOBS.job_events:
                    if len(each_event['start_jobs']) == 0:
                        continue
                    next_s_time = max(each_event['time'], next_s_time)
                    break
                next_s_time = cur_time + FLAGS.scheduling_slot
                while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_s_time):
                    time.sleep(1)
                    cur_time += 1
                if not fast_forward_permission:
                    print("ATTENTION!!!, cur_time", cur_time)
                    # modify overhead
                    update_overhead()

                for r_job in JOBS.running_jobs:
                    if r_job['job_idx'] not in job_stable:
                        if r_job['job_idx'] in this_round_running_jobs:
                            continue
                    r_job['old_end_time'] = r_job['end_time']
                    #if r_job['job_idx'] not in job_stable:
                    #    # not all jobs have finished scaling, but they have to be rescheduled
                    #    r_job['overhead'] = next_s_time - event['time']
                    r_job['end_time'] += r_job['overhead']
                    JOBS.change_job_end_event(r_job)

            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            next_e_time = JOBS.job_events[0]['time']
            while int(next_e_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                JOBS.job_events[0]['end_jobs'].extend(
                    JOBS.job_events[1]['end_jobs'])
                del JOBS.job_events[1]
            next_e_time = JOBS.job_events[0]['time']
            time_diff = int(next_e_time - event['time'])
            if time_diff < 0:
                print("ATTENTION! time diff < 0", JOBS.job_events[0])
            cur_time = next_e_time
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def ef_sim_jobs():
    '''
    run jobs with elasticity without access control;
    new jobs are added to the end of the pending queue;
    unsatisfiable jobs are declined.
    '''
    global this_round_begin_time, fast_forward_permission, global_lock
    global schedule_count
    cur_time = JOBS.job_events[0]['time']
    node_release = False
    time_diff = 0
    while (len(JOBS.job_events) + len(JOBS.pending_jobs) + len(JOBS.running_jobs))> 0:
        # if len(JOBS.job_events) == 0:
        #     break
        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        print()
        print("%d-%d, %d, %d " % (cur_time, len(JOBS.job_events), len(JOBS.pending_jobs), len(JOBS.running_jobs)))
        #update job progress for end_jobs
        JOBS.run_all_jobs(time_diff, cur_time)
        for r_job in JOBS.runnable_jobs:
            r_job['overhead'] = 0
        
        #get new start job
        event = utils.search_dict_list(JOBS.job_events, 'time', cur_time)
        if event != None:
            for e_job in event['end_jobs']:
                #remove from migratable jobs, if it's there
                JOBS.remove_migratable(e_job)
                #job completes
                CLUSTER.release_job_res(e_job)
                if e_job['end_time'] > e_job['ddl']:
                    utils.print_fn('----job[%d]\'s DDL request is not satisfied. Declined.' % e_job['job_idx'])
                    print(e_job['end_time'], "v.s.", e_job['ddl'])
                    JOBS.move_to_declined(e_job)
                    JOBS.num_accepted_job -= 1
                else:
                    print("ends at", cur_time, e_job['end_time'], "ddl", e_job['ddl'])
                JOBS.remove_running(e_job)
                LOG.job_complete(e_job, cur_time)
                has_ejob = True
                #input()

            for r_job in JOBS.runnable_jobs:
                r_job['old_end_time'] = r_job['end_time']
                r_job['old_allocations'] = r_job['allocations']

            if len(event['start_jobs']) > 0:
                CLUSTER.old_future_free_gpu = copy.deepcopy(CLUSTER.future_free_gpus)
                for s_job in event['start_jobs']:
                    JOBS.move_to_pending(s_job)
            estimate_all_jobs(JOBS.runnable_jobs, cur_time)

            JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('ddl'))
            for r_job in JOBS.runnable_jobs:
                r_job['num_gpu'] = r_job['allocations'][cur_time]
                JOBS.change_job_end_event(r_job)
                r_job['old_end_time'] = r_job['end_time']
                if 'allocations' in r_job:
                    r_job['old_allocations'] = copy.deepcopy(r_job['allocations'])
            
            #remove time_event
            JOBS.job_events.remove(event)


        if CLUSTER.check_free_gpu() > 0:
            #for pending jobs, try to start
            new_start_list = list()
            for p_job in JOBS.pending_jobs:
                pend_flag = False
                allocation_time = cur_time + 1
                for allocation_time in p_job['allocations']:
                    if p_job['allocations'][allocation_time] == 0:
                        continue
                    if allocation_time > cur_time:
                        pend_flag = True
                    break
                p_job['num_gpu'] = p_job['allocations'][cur_time]
                if pend_flag:
                    JOBS.add_job_scale_event(p_job, allocation_time)
                CLUSTER.free_gpu = CLUSTER.check_free_gpu()
                if p_job['num_gpu'] <= CLUSTER.free_gpu and not pend_flag:
                    new_start_list.append(p_job)
                    print("pending job", p_job['job_id'], p_job['allocations'])
            
            for ns_job in new_start_list:
                JOBS.get_network_load(ns_job)
                JOBS.remove_from_pending(ns_job, cur_time)
                JOBS.add_job_end_event(ns_job)
                ## add next arrangement event
                #if job is migratable, add into migratable job list
                JOBS.add_migratable(ns_job)
                # JOBS.read_job_info(p_job['job_idx'])
                utils.print_fn('----job[%d] starts from pending' % ns_job['job_idx'])
            
            for r_job in JOBS.runnable_jobs:
                print(r_job['job_idx'], r_job['allocations'], r_job['end_time'])
            # allocate free GPUs
            allocate_free_gpus(cur_time)
            for job in JOBS.running_jobs:
                # add scaling event for next allocation
                job['allocations'] = OrderedDict(sorted(job['allocations'].items(), key=lambda t: t[0]))
                for each_allocation_time in job['allocations']:
                    if each_allocation_time > cur_time:
                        JOBS.add_job_scale_event(job, each_allocation_time)
                        break
                if job["num_gpu"] >= CLUSTER.num_gpu_p_node:
                    ret = try_get_job_res(job)
                    assert ret
                    if not ret:
                        print('ERROR when allocating for job[%d]' % job['job_idx'])
            for node_group in reversed(list(CLUSTER.node_g.keys())):
                if node_group >= CLUSTER.num_gpu_p_node:
                    continue
                for job in JOBS.running_jobs:
                    if job["num_gpu"] == node_group:
                        ret = try_get_job_res(job)
                        if not ret:
                            print('ERROR when allocating for job[%d]' % job['job_idx'])
                            print(CLUSTER.node_g)
                        assert ret
            
            LOG.scheduling_result(cur_time)
        
        #input()

        if len(JOBS.job_events) <= 0:
            cur_time = cur_time + 10
            time_diff = 10
        else:
            if FLAGS.simulation:
                restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
                schedule_count += 1
                if restart:
                    LOG.cache = list()
                for r_job in JOBS.running_jobs:
                    if r_job['num_gpu'] > 0:
                        r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart, r_job['in_cache'])
                        r_job['old_end_time'] = r_job['end_time']
                        r_job['end_time'] += r_job['overhead']
                        JOBS.change_job_end_event(r_job)
                    else:
                        r_job['overhead'] = 0
            else:
                global_lock.acquire()
                this_round_begin_time = math.ceil(time.time())
                global_lock.release()
                get_ef_input_no_overlap(event['end_jobs'], this_round_begin_time)
                JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
                #start time of next job
                next_s_time = cur_time + FLAGS.scheduling_slot
                for each_event in JOBS.job_events:
                    if len(each_event['start_jobs']) == 0:
                        continue
                    next_s_time = max(each_event['time'], next_s_time)
                    break
                while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_s_time):
                    time.sleep(1)
                    cur_time += 1
                if not fast_forward_permission:
                    update_overhead()

                for r_job in JOBS.running_jobs:
                    if r_job['job_idx'] not in job_stable:
                        if r_job['job_idx'] in this_round_running_jobs:
                            continue
                    r_job['old_end_time'] = r_job['end_time']
                    #if r_job['job_idx'] not in job_stable:
                    #    # not all jobs have finished scaling, but they have to be rescheduled
                    #    r_job['overhead'] = next_s_time - event['time']
                    r_job['end_time'] += r_job['overhead']
                    JOBS.change_job_end_event(r_job)
            
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            next_e_time = JOBS.job_events[0]['time']
            while int(next_e_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                JOBS.job_events[0]['end_jobs'].extend(
                    JOBS.job_events[1]['end_jobs'])
                del JOBS.job_events[1]
            next_e_time = JOBS.job_events[0]['time']
            time_diff = int(next_e_time - event['time'])
            if time_diff < 0:
                print("ATTENTION! time diff < 0", JOBS.job_events[0])
            assert time_diff >= 0
            cur_time = next_e_time
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def one_queue_fifo_sim_jobs():
    '''
    run jobs in fifo order;
    new jobs are added to the end of the pending queue
    '''
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            utils.print_fn("This cluster is not large enough to run the job")
            break

        event = JOBS.job_events[0]
        event_time = event['time']
        # utils.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        has_ejob = False
        for e_job in event['end_jobs']:
            #remove from migratable jobs, if it's there
            # JOBS.remote_migratable(e_job)

            #job completes
            CLUSTER.release_job_res(e_job)
            # CLUSTER.release_gpus(e_job)
            LOG.job_complete(e_job, event_time)
            has_ejob = True


        #for new-start jobs, try to start
        for s_job in event['start_jobs']:
            #add into pending list
            JOBS.move_to_pending(s_job)


        if CLUSTER.check_free_gpu() > 0:
            #for pending jobs, try to start
            new_start_list = list()
            for p_job in JOBS.pending_jobs:
                # ret = CLUSTER.alloc_gpus(p_job)
                ret = try_get_job_res(p_job)
                if ret == True:
                    ''' if remove_from_pending, then will miss the next p_job in the list '''
                    new_start_list.append(p_job)
                    #if job is migratable, add into migratable job list
                    # JOBS.add_migratable(p_job)
                    # JOBS.remove_from_pending(p_job, event_time)
                    # JOBS.add_job_end_event(p_job)
                    # utils.print_fn('----job[%d] starts from pending' % p_job['job_idx'])
                    # JOBS.read_job_info(p_job['job_idx'])
                else:
                    break
            for ns_job in new_start_list:
                JOBS.remove_from_pending(ns_job, event_time)
                JOBS.add_job_end_event(ns_job)
                utils.print_fn('----job[%d] starts from pending' % ns_job['job_idx'])


        #sort pending jobs based on the num_gpu
        #JOBS.pending_jobs.sort(key = lambda e:e.__getitem__('num_gpu'))

        #remove time_event
        JOBS.job_events.pop(0)
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        # JOBS.print_job_events()

        LOG.checkpoint(event_time)


def themis_sim_jobs():
    '''
    themis finish-time fairness
    '''
    global this_round_begin_time, global_lock, fast_forward_permission
    num_steps_remaining_prev_iteration, isolated_throughputs_prev_iteration = {}, {}
    cumulative_isolated_time = {} 
    cur_time = JOBS.job_events[0]['time']
    schedule_count = 0
    old_running_jobs = []
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs)) > 0:
        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        running_jobs_list = []
        for r_job in JOBS.runnable_jobs:
            if 'node_set' in r_job:
                CLUSTER.release_job_res(r_job, end=False)
        #jobs.run_all_jobs
        #cur_time = JOBS.job_events[0]['time'] #todo: judge
        print(cur_time, len(JOBS.job_events), "events left,", len(JOBS.runnable_jobs), "runnable jobs left")
        #if len(JOBS.job_events) == 0:
        #    input()
        event = utils.search_dict_list(JOBS.job_events, 'time', cur_time)
        if event != None:
            event = JOBS.job_events[0]
            event_time = event['time']
            # end job
            for e_job in event['end_jobs']:
                if e_job['end_time'] > e_job['ddl']:
                    utils.print_fn('----job[%d]\'s DDL request is not satisfied. Declined.' % e_job['job_idx'])
                    print(e_job['end_time'], "v.s.", e_job['ddl'])
                    JOBS.move_to_declined(e_job)
                    #JOBS.num_accepted_job -= 1
                else:
                    print("ends at", cur_time, e_job['end_time'], "ddl", e_job['ddl'])
                    if 'best_effort' not in e_job or int(e_job['best_effort']) != 1:
                        JOBS.num_accepted_job += 1
                JOBS.remove_running(e_job)
                LOG.job_complete(e_job, e_job['end_time'])

            # end job, release all resources CLUSTER.release_job_res(e_job)
            for s_job in event['start_jobs']:
                #add into pending list
                JOBS.move_to_pending(s_job)
        if len(JOBS.runnable_jobs) > 0:
            scale_factors_array = utils.scale_factors_array(JOBS.runnable_jobs)
            isolated_throughputs = utils.get_isolated_throughputs(JOBS.runnable_jobs)
            x = cp.Variable(len(JOBS.runnable_jobs)) 
            #avg_share = math.ceil(CLUSTER.num_gpu / len(JOBS.runnable_jobs))
            job_idx = 0
            expected_time_fractions = []
            for r_job in JOBS.runnable_jobs:
                assert r_job['iter_left'] > 0
                if r_job['job_idx'] not in cumulative_isolated_time:
                    cumulative_isolated_time[r_job['job_idx']] = 0
                if r_job['job_idx'] in num_steps_remaining_prev_iteration:
                    cumulative_isolated_time[r_job['job_idx']] += (
                        num_steps_remaining_prev_iteration[r_job['job_idx']] -
                        r_job['iter_left']) / \
                        isolated_throughputs_prev_iteration[r_job['job_idx']]
                throughput = THROUGHPUTS[r_job['model']['name']][str(
                    r_job['batch_size'])][str(r_job['num_gpu'])]
                allocation_throughput = throughput * x[job_idx]
                expected_time_isolated = cumulative_isolated_time[r_job['job_idx']] + \
                (r_job['iter_left'] / isolated_throughputs[job_idx])
                expected_time_allocation = (event_time - r_job['submit_time']) + \
                    (r_job['iter_left'] * cp.inv_pos(allocation_throughput))
                num_steps_remaining_prev_iteration[r_job['job_idx']] = r_job['iter_left']
                expected_time_fraction = expected_time_allocation / expected_time_isolated
                #print("expected_time_allocation, expected_time_isolated", expected_time_allocation, expected_time_isolated)
                expected_time_fractions.append(expected_time_fraction)
                isolated_throughputs_prev_iteration[r_job['job_idx']] = isolated_throughputs[job_idx]
                job_idx += 1

            if len(expected_time_fractions) == 1:
                objective = cp.Minimize(expected_time_fractions[0])
            else:
                objective = cp.Minimize(cp.maximum(*expected_time_fractions))

            # Make sure that the allocation can fit in the cluster.
            constraints = utils.get_base_constraints(x, scale_factors_array)
            cvxprob = cp.Problem(objective, constraints)
            result = cvxprob.solve(solver='ECOS')

            if cvxprob.status != "optimal":
                print('WARNING: Allocation returned by policy not optimal!')
                
                
            print(x.value)
            # reset time so far
            """worker_time_so_far = 0
            for r_job in JOBS.runnable_jobs:
                r_job['job_time_so_far'] = FLAGS.scheduling_slot / 2.0
                worker_time_so_far += r_job['job_time_so_far']"""

            # update priorities
            #fractions = {}
            for i, r_job in enumerate(JOBS.runnable_jobs):
                """if worker_time_so_far == 0.0:
                    fraction = 0.0
                else:
                    fraction = r_job['job_time_so_far'] / worker_time_so_far
                fractions[r_job['job_idx']] = fraction
                new_priority = x.value[i] * 1e9
                if fractions[r_job['job_idx']] > 0.0:
                    new_priority = x.value[i] / fractions[r_job['job_idx']]"""
                r_job['priority'] = x.value[i] * 1e9
                if 'rounds_received' not in r_job:
                    r_job['rounds_received'] = 0
                if r_job['rounds_received'] > 0:
                    r_job['priority'] = x.value[i] / r_job['rounds_received']
                r_job['x'] = x.value[i]

            JOBS.runnable_jobs.sort(key=lambda e:(e.__getitem__('priority'), e.__getitem__('x')), reverse=True)
            JOBS.running_jobs = list()
            free_gpus = CLUSTER.num_gpu
            for r_job in JOBS.runnable_jobs:
                if free_gpus <= 0:
                    break
                assert free_gpus > 0
                if r_job['num_gpu'] <= free_gpus:
                    JOBS.running_jobs.append(r_job)
                    free_gpus -= r_job['num_gpu']
            # allocate
            JOBS.running_jobs.sort(key=lambda e:(e.__getitem__('num_gpu')), reverse=True)
            for r_job in JOBS.running_jobs:
                ret = try_get_job_res(r_job)
                if not ret:
                    print('ERROR when allocating for job[%d]' % r_job['job_idx'])
                    print(CLUSTER.node_g)
                assert ret
                running_jobs_list.append(r_job['job_idx'])
        LOG.scheduling_result(cur_time)

        if len(JOBS.job_events) > 0 and cur_time == JOBS.job_events[0]['time']:
            #remove time_event
            JOBS.job_events.pop(0)
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        # JOBS.print_job_events()

        # run all jobs
        next_e_time = cur_time + FLAGS.scheduling_slot # lease time
        time_diff = FLAGS.scheduling_slot
        
        if len(JOBS.job_events) > 0:
            next_s_time = JOBS.job_events[0]['time']
            while int(next_s_time - cur_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = cur_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - cur_time > FLAGS.scheduling_slot:
                    next_s_time = JOBS.job_events[0]['time']
                    break
                next_s_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                JOBS.job_events[0]['end_jobs'].extend(
                    JOBS.job_events[1]['end_jobs'])
                del JOBS.job_events[1]
            if int(next_s_time - cur_time) < FLAGS.scheduling_slot:
                assert len(JOBS.job_events) == 1
                JOBS.job_events[0]['time'] = next_e_time
            
        end_jobs = []
        reschedule_flag = False
        for job in old_running_jobs:
            if job not in running_jobs_list:
                reschedule_flag = True
                break
        for job in running_jobs_list:
            if job not in old_running_jobs:
                reschedule_flag = True
                break
        if FLAGS.simulation:
            restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
            schedule_count += 1
            
        elif reschedule_flag:
            global_lock.acquire()
            this_round_begin_time = math.ceil(time.time())
            global_lock.release()
            if event is None:
                get_ef_input_no_overlap([], this_round_begin_time)
            else:
                get_ef_input_no_overlap(event['end_jobs'], this_round_begin_time)####
            JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
            while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_e_time):
                time.sleep(1)
                cur_time += 1
            if not fast_forward_permission:
                update_overhead()

            for r_job in JOBS.running_jobs:
                if 'node_set' not in r_job:
                    continue
                #if r_job['job_idx'] not in job_stable:
                #    # not all jobs have finished scaling, but they have to be rescheduled
                #    r_job['overhead'] = next_e_time - event['time']
        for r_job in JOBS.runnable_jobs:
            if 'node_set' not in r_job:
                continue
            if reschedule_flag == False:
                r_job['overhead'] = 0
            elif 'overhead' not in r_job:
                r_job['overhead'] = utils.estimate_overhead(r_job['num_gpu'], restart, r_job['in_cache']) # only for simulation
            if 'remaining_time' not in r_job:
                r_job['remaining_time'] = r_job['duration']
            end_time = cur_time + r_job['overhead'] + r_job['remaining_time']
            r_job['remaining_time'] -= (time_diff - r_job['overhead'])
            r_job['remaining_time'] = max(0, r_job['remaining_time'])
            r_job['iter_left'] -= (time_diff - r_job['overhead']) * float(THROUGHPUTS[r_job['model']['name']][str(
                r_job['batch_size'])][str(r_job['num_gpu'])])
            if r_job['iter_left'] <= 0:
                r_job['remaining_time'] = 0
                end_time = next_e_time
            if end_time <= next_e_time:
                # find all jobs that will end before next scheduling event
                end_jobs.append(r_job)
                r_job['end_time'] = end_time
        if len(JOBS.job_events) > 0:
            if JOBS.job_events[0]['time'] == next_e_time:
                JOBS.job_events[0]['end_jobs'].extend(end_jobs)
            elif len(end_jobs) > 0:
                JOBS.job_events.append({'time':next_e_time,'start_jobs':[], 'end_jobs':end_jobs})
        elif len(end_jobs) > 0:
            JOBS.job_events.append({'time':next_e_time,'start_jobs':[], 'end_jobs':end_jobs})
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))

        cur_time = next_e_time
        old_running_jobs = running_jobs_list

        LOG.checkpoint(event_time)
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')

def get_tiresias_input(end_jobs, actual_time):
    global job_stable, fast_forward_permission, MASTER_PORT, commands
    global last_round_running_jobs, this_round_running_jobs, trainers_to_kill
    global global_lock, global_ready_lock
    global job_to_be_killed, schedule_count
    global last_round_gpu_allocations, gpu_allocations
    global_lock.acquire()
    # if there is need to reschedule
    return_flag = True
    for job in JOBS.runnable_jobs:
        if job['status'] != 'RUNNING':
            continue
        if job['num_gpu'] == 0 or job['node_set'] is None:
            if job['job_idx'] in last_round_running_jobs:
                return_flag = False
                break
            continue
        if job['num_gpu'] < CLUSTER.num_gpu_p_node:
            if job['job_idx'] in last_round_running_jobs:
                if 'num_gpu' in last_round_running_jobs[job['job_idx']] and last_round_running_jobs[job['job_idx']]['num_gpu'] == job['num_gpu']:
                    # do not need to update in this round
                    continue
                else:
                    return_flag = False 
                    break
            else:
                return_flag = False 
                break
        else:
            if job['job_idx'] in last_round_running_jobs:
                # same number of nodes
                if len(last_round_running_jobs[job['job_idx']]['worker_id']) == len(job['node_set']['nodes']):
                    continue
                else:
                    return_flag = False 
                    break
            else:
                return_flag = False 
                break

    if return_flag:
        fast_forward_permission = True
        global_lock.release()
        return

    restart_trainers = (schedule_count % FLAGS.restart_threshold == 0) and schedule_count != 0
    # restart if jobs didn't restart successfully last round
    if not restart_trainers:
        restart_trainers = job_to_be_killed
    del this_round_running_jobs
    this_round_running_jobs = dict()
    if restart_trainers:
        scheduler_rpc_client.schedule('RE')
    else:
        for e_job in end_jobs:
            assert e_job['job_idx'] in last_round_running_jobs
            for each_worker in last_round_running_jobs[e_job['job_idx']]['worker_id']:
                command = ' '.join(["K", 
                    str(each_worker), str(e_job['job_idx'])])
                print(command)
                scheduler_rpc_client.schedule(command)
                job_to_be_killed = True
            del last_round_running_jobs[e_job['job_idx']]
    
    gpu_allocations = [[0 for gpu in range(CLUSTER.num_gpu_p_node)] for _ in range(CLUSTER.num_node)]
    del job_stable
    job_stable = dict()
    commands = []

    # new jobs
    for job in JOBS.runnable_jobs:
        if job['status'] != 'RUNNING':
            continue
        if job['num_gpu'] == 0 or job['node_set'] is None:
            continue
        cmd = 'R'
        if job['num_gpu'] >= CLUSTER.num_gpu_p_node:
            for i in range(len(job['node_set']['nodes'])):
                compressed_gpu_list = 0
                for j in range(len(gpu_allocations[job['node_set']['nodes'][i].id])):
                    gpu_allocations[job['node_set']['nodes'][i].id][j] = 1
                    compressed_gpu_list += (1 << j)
                MASTER_PORT += 1
                command = ' '.join([cmd, str(job['node_set']['nodes'][i].id), job['model_name'], 
                    str(job['batch_size']), str(job['job_idx']), str(min(job['num_gpu'], CLUSTER.num_gpu_p_node)), str(len(job['node_set']['nodes'])), 
                    str(i), '127.0.0.1', str(MASTER_PORT), str(compressed_gpu_list), str(int(job['iter_left'])), str(actual_time)])
                print(command)
                if job['job_idx'] not in this_round_running_jobs:
                    this_round_running_jobs[job['job_idx']] = {'worker_id':[]}
                this_round_running_jobs[job['job_idx']]['worker_id'].append(str(job['node_set']['nodes'][i].id))
                if job['job_idx'] not in job_stable:
                    job_stable[job['job_idx']] = 0
                fast_forward_permission = False
                #scheduler_rpc_client.schedule(command)
                commands.append(command)
        else:
            node_id = job['node_set']['nodes'][0].id
            if job['job_idx'] in this_round_running_jobs:
                continue

            allocated_gpu = 0
            compressed_gpu_list = 0
            for i in range(len(gpu_allocations[node_id])):
                if gpu_allocations[node_id][i] == 1:
                    continue
                allocated_gpu += 1
                gpu_allocations[node_id][i] = 1
                compressed_gpu_list += (1 << i)
                if allocated_gpu == job['num_gpu']:
                    break
            MASTER_PORT += 1
            command = ' '.join([cmd, str(node_id), job['model_name'], str(job['batch_size']), str(job['job_idx']), 
                str(min(job['num_gpu'], CLUSTER.num_gpu_p_node)), str(len(job['node_set']['nodes'])), '0', '127.0.0.1', str(MASTER_PORT), str(compressed_gpu_list), 
                str(int(job['iter_left'])), str(actual_time)])
            print(command)
            tmp_dict = dict()
            tmp_dict['worker_id'] = [str(node_id)]
            tmp_dict['num_gpu'] = job['num_gpu']
            tmp_dict['compressed_gpu_list'] = compressed_gpu_list
            this_round_running_jobs[job['job_idx']] = tmp_dict
            if job['job_idx'] not in job_stable:
                job_stable[job['job_idx']] = 0
            fast_forward_permission = False
            #scheduler_rpc_client.schedule(command)
            commands.append(command)
    #TODO: let master stop old jobs for on-the-fly elastic trainers
    
    global_ready_lock.acquire()
    if len(this_round_running_jobs) > 0:
        trainers_to_kill = {}
    for job in this_round_running_jobs:
        trainers_to_kill[job] = []
        if 'num_gpu' in this_round_running_jobs[job]:
            for each_gpu in utils.fetch_GPU_list_to_int(this_round_running_jobs[job]['compressed_gpu_list']):
                if last_round_gpu_allocations[int(this_round_running_jobs[job]['worker_id'][0])][each_gpu] == 0:
                    continue
                trainers_to_kill[job].append(utils.get_global_rank(
                    this_round_running_jobs[job]['worker_id'][0],
                    each_gpu, CLUSTER.num_gpu_p_node))
        else:   
            for each_worker in this_round_running_jobs[job]['worker_id']:
                for each_gpu in range(CLUSTER.num_gpu_p_node):
                    if last_round_gpu_allocations[int(each_worker)][each_gpu] == 0:
                        continue
                    trainers_to_kill[job].append(utils.get_global_rank(
                    each_worker, each_gpu, CLUSTER.num_gpu_p_node))
    
    print("$$$ in no overlap, trainers to kill", trainers_to_kill)
    print("$$$ last_round_running_jobs", last_round_running_jobs)
    print("$$$ this_round_running_jobs", this_round_running_jobs)

    if not restart_trainers:
        for job in last_round_running_jobs:
            for each_worker in last_round_running_jobs[job]['worker_id']:
                command = 'K ' + str(each_worker) + ' ' + str(job)
                scheduler_rpc_client.schedule(command)
                job_to_be_killed = True
    else:
        job_to_be_killed = False
    
    if not job_to_be_killed:
        # run all commands
        for command in commands:
            scheduler_rpc_client.schedule(command)
        scheduler_rpc_client.schedule('F')
        scheduler_rpc_client.schedule('T')
        last_round_gpu_allocations = gpu_allocations
    fast_forward_permission = (len(commands) == 0)

    global_ready_lock.release()
    del last_round_running_jobs
    last_round_running_jobs = this_round_running_jobs
    #last_round_gpu_allocations = gpu_allocations
    schedule_count += 1
    global_lock.release()

def dlas_sim_jobs(gputime=False, solve_starvation=0):
    '''
    Job's executed time -- priority queue
    Q0:[0, 30min)
    Q1:[30min,1h)
    Q2:[1h, 2h)
    Q3:[2h, 00)
    in each queue, jobs are scheduled in fit-first with FIFO
    how to avoid starvation?
    TODO:  2. add move_back for avoiding starvation
    '''
    global this_round_begin_time, fast_forward_permission, global_lock, run_jobs
    global schedule_count
    end_events = list()
    next_job_jump = sys.maxsize
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            utils.print_fn("This cluster is not large enough to run the job")
            #print(JOBS.runnable_jobs)
            for each in JOBS.runnable_jobs:
                print(each['job_id'], each['num_gpu'], each['status'])
            break

        #decide which is the next event: start or end  ?
        start_event = None
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']

        end_event = None
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']
        
        event_time = sys.maxsize
        event = dict()
        event['time'] = sys.maxsize
        if end_time < start_time:
            event_time = end_time
            event = end_event
        elif end_time > start_time:        
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #decide if job_jump first or (start/end) first
        if event_time > next_job_jump:
            event_time = next_job_jump
            event = dict()

        print("accepted jobs:", JOBS.num_accepted_job)
        print("declined jobs:", JOBS.num_declined_job)
        print()
        print(event_time, event)
        cur_time = event_time

        # utils.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                if e_job['end_time'] > e_job['ddl']:
                    utils.print_fn('----job[%d]\'s DDL request is not satisfied. Declined.' % e_job['job_idx'])
                    print(e_job['end_time'], "v.s.", e_job['ddl'])
                    JOBS.move_to_declined(e_job)
                    #input()
                else:
                    if 'best_effort' not in e_job or int(e_job['best_effort']) != 1:
                        JOBS.num_accepted_job += 1
                    print("ends at", event_time, e_job['end_time'], "ddl", e_job['ddl'])
                CLUSTER.release_job_res(e_job)
                LOG.job_complete(e_job, e_job['end_time'])
                # utils.print_fn('---- job[%d] is completed' % e_job['job_idx'])
                JOBS.runnable_jobs.remove(e_job)
                JOBS.queues[e_job['q_id']].remove(e_job)

        #for new jobs, append to runnable jobs with pending status
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                JOBS.move_to_runnable(s_job)
                s_job['q_id'] = 0 #any new start job should be in Q0
                JOBS.queues[0].append(s_job)
                utils.print_fn('---- job[%d] is added' % s_job['job_idx'])
            #pop start event
            JOBS.job_events.pop(0)

        #update executed_time
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                if 'overhead' not in rjob:
                    rjob['overhead'] = 0
                tmp = max(int(event_time - rjob['last_check_time']) - rjob['overhead'], 0)
                rjob['total_executed_time'] = int(rjob['total_executed_time'] + tmp)
                rjob['executed_time'] = int(rjob['executed_time'] + tmp) # decide job priority queue
                rjob['last_check_time'] = event_time
                if rjob['overhead'] != 0:
                    rjob['overhead'] = 0

                #check demotion
                j_gt = 0
                if gputime:
                    j_gt = int(rjob['executed_time'] * rjob['num_gpu'])
                else:
                    j_gt = int(rjob['executed_time'])
                cur_qid = rjob['q_id']
                if cur_qid < int(JOBS.num_queue - 1): #not for the last queue 
                    if j_gt >= JOBS.queue_limit[cur_qid]:
                        rjob['q_id'] = int(cur_qid + 1)
                        JOBS.queues[rjob['q_id']].append(rjob)
                        JOBS.queues[cur_qid].remove(rjob)
                        print("job %d demote to Q%d" % (rjob['job_idx'], rjob['q_id']))

                if FLAGS.schedule == 'dlas-gpu-gittins': 
                    # rjob['rank'] = cal_r_gittins_index(JOBS.job_dist_data, j_gt)
                    rjob['rank'] = get_gittins_index(j_gt)

            elif 'PENDING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time']) 
                rjob['last_check_time'] = event_time
                rjob['pending_time'] = int(rjob['pending_time'] + tmp) #this is the total pending_time
                if rjob['executed_time'] > 0: # if not started yet, job is always in Q0 and no need to push_back
                    rjob['last_pending_time'] = int(rjob['last_pending_time'] + tmp) #this is the total pending_time
                #Q0 job no need to push_back, and must be a runned 
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] > 0 and rjob['executed_time'] > 0:
                    if rjob['last_pending_time'] >= int(rjob['executed_time'] * solve_starvation):
                        rjob['executed_time'] = 0
                        rjob['last_pending_time'] = 0
                        JOBS.queues[0].append(rjob)
                        JOBS.queues[rjob['q_id']].remove(rjob)
                        rjob['q_id'] = 0
                        rjob['promote'] = int(rjob['promote'] + 1)

                if FLAGS.schedule == 'dlas-gpu-gittins': 
                    if gputime:
                        j_gt = int(rjob['executed_time'] * rjob['num_gpu'])
                    else:
                        j_gt = int(rjob['executed_time'])
                    # rjob['rank'] = cal_r_gittins_index(JOBS.job_dist_data, j_gt)
                    rjob['rank'] = get_gittins_index(j_gt)

            elif 'END' == rjob['status']: # won't happen
                JOBS.runnable_jobs.remove(rjob)
                # utils.print_fn('---- job[%d] completed' % rjob['job_idx'])
                pass

        #push job to their new queue
        # JOBS.update_priority_queues(gputime)

        ''' schedule jobs in each queue '''
        #empty_cluster resource
        #CLUSTER.empty_infra()
        CLUSTER.free_gpu = CLUSTER.num_gpu
        # for "count" placement
        run_jobs = list()
        preempt_jobs = list()

        # if FLAGS.schedule == 'dlas-gpu-gittins': 
        #     q = JOBS.queues[0]
        #     q.sort(key = lambda e:(e.__getitem__('rank'), e.__getitem__('r_submit_time')), reverse=True)

        if FLAGS.simulation:
            restart = schedule_count > 0 and schedule_count % FLAGS.scheduling_slot == 0
            schedule_count += 1
        for queue in JOBS.queues:
            if FLAGS.schedule == 'dlas-gpu-gittins': 
                queue.sort(key = lambda e:(e.__getitem__('rank'), e.__getitem__('r_submit_time')), reverse=True)
            for job in queue:
                ## make sure that all jobs to run can be allocated
                #ret = CLUSTER.release_job_res(job, end=False)
                #assert ret
                if CLUSTER.free_gpu >= job['num_gpu']:
                    #should run
                    if job['status'] == 'PENDING':              
                        #not running
                        run_jobs.append(job)
                        if FLAGS.simulation:
                            job['overhead'] = utils.estimate_overhead(job['num_gpu'], restart)
                        
                    CLUSTER.free_gpu = int(CLUSTER.free_gpu - job['num_gpu'])
                else:
                    #should NOT run
                    if job['status'] == 'RUNNING':                   
                        #running
                        preempt_jobs.append(job)
                    continue
        for job in JOBS.runnable_jobs:
            if 'node_set' in job:
                # make sure that all jobs to run can be allocated
                ret = CLUSTER.release_job_res(job, end=False)
                assert ret
        
        for job in preempt_jobs:
            job['status'] = 'PENDING'
            # if job['q_id'] == 0:
            #     job['preempt'] = int(job['preempt'] + 1)
            job['preempt'] = int(job['preempt'] + 1)
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            if job['start_time'] == sys.maxsize:
                job['start_time'] = event_time
            #JOBS.get_network_load(job)
            #ret = try_get_job_res(job)
            #assert ret
        JOBS.runnable_jobs.sort(key = lambda e:(e.__getitem__('num_gpu')), reverse=True)
        for job in JOBS.runnable_jobs:
            if job['status'] == 'RUNNING':
                JOBS.get_network_load(job)
                ret = try_get_job_res(job)
                if not ret:
                    print(CLUSTER.node_g)
                    print(job)
                assert ret

        #sort based on the job start time
        for queue in JOBS.queues:
            #job there are many students            
            pending_job = list()
            for job in queue: 
                # if sys.maxsize == job['start_time'] and job['status'] == 'PENDING':
                if job['status'] == 'PENDING':
                    pending_job.append(job)
                    # print(job['job_idx'])
            for job in pending_job: 
                queue.remove(job)
            queue.extend(pending_job)

        #update end events and sort, and get the most recent one
        del end_events[:]
        min_end_time = sys.maxsize
        tmp_end_event = dict()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                remaining_time = rjob['duration'] - rjob['total_executed_time']
                if FLAGS.simulation:
                    if restart:
                        rjob['overhead'] = utils.estimate_overhead(rjob['num_gpu'], restart)
                        remaining_time += rjob['overhead']
                end_time = int(event_time + remaining_time)
                end_event_time = max(event_time + FLAGS.scheduling_slot, end_time)
                if end_event_time < min_end_time:
                    tmp_end_event['time'] = end_event_time
                    tmp_end_event['end_jobs'] = list()
                    tmp_end_event['end_jobs'].append(rjob)
                    min_end_time = end_event_time
                    rjob['end_time'] = end_time
                elif min_end_time == end_event_time:
                    rjob['end_time'] = end_time
                    tmp_end_event['end_jobs'].append(rjob)
        if min_end_time < sys.maxsize:
            end_events.append(tmp_end_event)

        # what's the closest queue_jump (demotion, and promotion) among all the jobs
        next_job_jump = sys.maxsize
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                qid = rjob['q_id']
                if qid < int(JOBS.num_queue - 1):
                    if gputime:
                        jump_time = int(math.ceil((JOBS.queue_limit[qid] - rjob['executed_time'])/rjob['num_gpu']) + event_time)
                    else:
                        jump_time = int(JOBS.queue_limit[qid] - rjob['executed_time'] + event_time)
                    if jump_time < next_job_jump:
                        next_job_jump = jump_time

            elif 'PENDING' == rjob['status']: # when pending job will be push back to Q0
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] and rjob['executed_time'] > 0:
                    diff_time = int(rjob['executed_time'] * solve_starvation - rjob['last_pending_time'])
                    if diff_time > 0:
                        jump_time = int(diff_time + event_time)
                        if jump_time < next_job_jump:
                            next_job_jump = jump_time

        LOG.checkpoint(event_time)
        LOG.scheduling_tiresias_result(event_time)
        
        if not FLAGS.simulation:
            #decide which is the next event: start or end  ?
            start_time = sys.maxsize
            if len(JOBS.job_events) > 0:
                start_event = JOBS.job_events[0]
                start_time = start_event['time']
            end_event = None
            end_time = sys.maxsize
            if len(end_events) > 0:
                end_event = end_events[0]
                end_time = end_event['time']
        
            next_event_time = sys.maxsize
            if end_time < start_time:
                next_event_time = end_time
            elif end_time > start_time:        
                next_event_time = start_time
            elif end_time == start_time and end_time != sys.maxsize:
                next_event_time = start_time
            #decide if job_jump first or (start/end) first
            if event_time > next_job_jump:
                next_event_time = next_job_jump
            if next_event_time - cur_time < FLAGS.scheduling_slot:
                next_event_time = cur_time + FLAGS.scheduling_slot

            global_lock.acquire()
            this_round_begin_time = math.ceil(time.time())
            global_lock.release()
            if 'end_jobs' in event:
                get_tiresias_input(event['end_jobs'], this_round_begin_time)
            else:
                get_tiresias_input([], this_round_begin_time)
            while (not FLAGS.fastforward or not fast_forward_permission) and (cur_time < next_event_time):
                time.sleep(1)
                cur_time += 1
            if not fast_forward_permission:
                update_overhead()
                print("ATTENTION! not all jobs ready")

        # wait and record overhead.


        # if event time > start_time or end_time: modify start time and end time
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        if next_job_jump - event_time < FLAGS.scheduling_slot:
            next_job_jump = event_time + FLAGS.scheduling_slot
        if len(JOBS.job_events) > 0:
            next_e_time = JOBS.job_events[0]['time']
            while int(next_e_time - event_time) <= FLAGS.scheduling_slot and len(JOBS.job_events) > 1:
                JOBS.job_events[0]['time'] = event_time + FLAGS.scheduling_slot
                if JOBS.job_events[1]['time'] - event_time > FLAGS.scheduling_slot:
                    break
                next_e_time = JOBS.job_events[1]['time']
                JOBS.job_events[0]['start_jobs'].extend(
                    JOBS.job_events[1]['start_jobs'])
                del JOBS.job_events[1]
            end_time = sys.maxsize
        assert len(end_events) == 1 or len(end_events) == 0
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']
            if len(end_events) == 1 and end_time < event_time + FLAGS.scheduling_slot:
                end_events[0]['time'] = event_time + FLAGS.scheduling_slot
    if not FLAGS.simulation:
        scheduler_rpc_client.schedule('S')


def get_gittins_index(a):
    job_info = JOBS.job_dist_data
    if a > job_info['data'][-2]:
        return 0
    idx = next(x[0] for x in enumerate(job_info['data']) if x[1] > a)
    return job_info['gittins'][idx]


def cal_r_gittins_index(job_data, a):
    '''
    a means attained-service to that job
    gittins_index = P/E
    r_gi = E/P
    '''
    ut_delta = JOBS.gittins_delta

    data = job_data['data']
    if a > (job_data['data'][-1] - 1):
        return 0.0
    else:
        idx = next(x[0] for x in enumerate(data) if x[1] > a)

    next_a = a + ut_delta
    if next_a > (job_data['data'][-1] - 1):
        idx_delta = job_data['num'] - 1
    else:
        idx_delta = next(x[0] for x in enumerate(data) if x[1] > next_a)
    # print(idx, idx_delta)

    p = round(((idx_delta - idx) * 1.0) / (job_data['num'] - idx), 5)

    e_sum = sum(data[idx : idx_delta]) + (ut_delta * (job_data['num'] - idx_delta))
    e = round(e_sum / (job_data['num'] - idx), 5)

    # rank of gittins index = 1/gi
    # r_gi = round(e / p, 4)
    r_gi = round(p * 1000000 / e, 4)

    # print(idx, idx_delta, p, e_sum, e, r_gi)
    return r_gi


def parse_job_dist():
    job_dist_file = os.path.join(os.getcwd(), 'yarn-gput1000.csv')
    fd = open(job_dist_file, 'r')
    reader = csv.DictReader(fd, delimiter = ',') 
    durations = list()
    for row in reader:
        durations.append(int(row['duration']))
    fd.close()
    total_len = len(durations)
    durations.sort()
    print("  %s samples are learned" % total_len)

    job_dict = dict()
    job_dict['num'] = total_len
    job_dict['data'] = durations

    gi = list()
    for v in job_dict['data']:
        gi.append(cal_r_gittins_index(job_dict, int(v-1)))

    # print(gi)
    job_dict['data'].append(sys.maxsize)
    gi.append(0.0)
    job_dict['gittins'] = gi

    return job_dict


def main():

    if FLAGS.schedule == 'multi-dlas-gpu': 
        if FLAGS.scheme != 'count':
            utils.print_fn("In Main, multi-dlas-gpu without count")
            exit()
    if FLAGS.gpu_type == 'A100':
        throughput_path = "./throughputs_A100/"
    else:
        throughput_path = "./throughputs_T4/"
    for throughput_file in os.listdir(throughput_path):
        profiles.parse_throughput_file(throughput_path + throughput_file)
    ''' Parse input'''
    parse_job_file(FLAGS.trace_file)
    parse_cluster_spec()

    ''' prepare logging '''
    LOG.init_log()

    # lp.placement(JOBS.job_list[0])
    ''' Prepare jobs'''
    JOBS.prepare_job_start_events()

    if FLAGS.schedule == 'edf':
        #JOBS.job_dist_data = parse_job_dist()
        CLUSTER.init_gandiva_nodes()
        one_queue_edf_sim_jobs()
    elif FLAGS.schedule == 'ef-accessctrl':
        CLUSTER.init_gandiva_nodes()
        ef_sim_jobs_access_control()
    elif FLAGS.schedule == 'ef':
        CLUSTER.init_gandiva_nodes()
        ef_sim_jobs()
    elif FLAGS.schedule == 'edf-accessctrl':
        CLUSTER.init_gandiva_nodes()
        one_queue_edf_sim_jobs_access_control()
    elif FLAGS.schedule == 'fifo':
        one_queue_fifo_sim_jobs()
    elif FLAGS.schedule == 'dlas':
        JOBS.job_dist_data = parse_job_dist()
        dlas_sim_jobs()
    elif FLAGS.schedule == 'dlas-gpu':
        CLUSTER.init_gandiva_nodes()
        dlas_sim_jobs(True)
    elif FLAGS.schedule == 'themis':
        CLUSTER.init_gandiva_nodes()
        themis_sim_jobs()
    elif FLAGS.schedule == 'gandiva':
        CLUSTER.init_gandiva_nodes()
        gandiva_sim_jobs(True, 1000)
    elif FLAGS.schedule == 'gpu-demands':
        sim_gpu_demands()
    else:
        one_queue_fifo_sim_jobs()
    print("accepted jobs:", JOBS.num_accepted_job)
    print("declined jobs:", JOBS.num_declined_job)
    # record time ratio, cluster size, trace_file, schedule, placement
    LOG.log_final_result(JOBS.num_accepted_job, JOBS.num_declined_job)


if __name__ == '__main__':
    # print('Hello world %d' % 2)
    if not FLAGS.simulation:
        # RPC client to master
        scheduler_rpc_client = scheduler_client.SchedulerRpcClient('127.0.0.1', 6888)
        # run master rpc server in the background
        scheduler_server_port = 6890
        callbacks = {
            'ReportStable' : report_stable_callback,
            'ReportReady' : report_ready_callback,
        }
        server_thread = threading.Thread(target=scheduler_server.serve, 
            args=(scheduler_server_port, callbacks))
        server_thread.setDaemon(True)
        server_thread.start()
    main()