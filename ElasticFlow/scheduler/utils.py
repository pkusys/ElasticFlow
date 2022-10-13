from __future__ import print_function
import cvxpy as cp
import math
import socket
import sys
import subprocess
import cluster
import flags 
import profiles
import numpy as np
FLAGS = flags.FLAGS
#pre-run throughput information
THROUGHPUTS = profiles.THROUGHPUTS
CLUSTER = cluster.CLUSTER

def get_host_ip():
    """get the host ip elegantly
    https://www.chenyudong.com/archives/python-get-local-ip-graceful.html
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def print_fn(log):
    if FLAGS.print:
        print(log)
        if FLAGS.flush_stdout:
            sys.stdout.flush()


def mkdir(folder_path):
    cmd = 'mkdir -p ' + folder_path
    ret = subprocess.check_call(cmd, shell=True)
    print_fn(ret)


def search_dict_list(dict_list, key, value):
    '''
    Search the targeted <key, value> in the dict_list
    Return:
        list entry, or just None 
    '''
    for e in dict_list:
        # if e.has_key(key) == True:
        if key in e:
            if e[key] == value:
                return e

    return None

def merge_dict(dictionary, start_time=None):
    last_value = -1
    new_allocations = dict()
    for key in dictionary:
        if dictionary[key] == last_value:
            continue
        new_allocations[key] = dictionary[key]
        last_value = dictionary[key]
    return new_allocations


def get_global_rank(worker_id, local_rank, gpu_p_node):
    return int(gpu_p_node) * int(worker_id) + int(local_rank)


def get_allocation_time(allocations, end_time, free_gpus=0, alpha=1):
    last_time = None
    last_gpu = None
    gpu_time = 0
    for each_allocation in allocations:
        if last_time is not None:
            gpu_time += last_gpu * (pow(each_allocation - last_time, alpha))
        last_time = each_allocation
        last_gpu = allocations[each_allocation] + free_gpus
    gpu_time += last_gpu * (pow(end_time - last_time, alpha))
    return gpu_time


def get_next_level(job_dict, gpu_num=None):
    if gpu_num is None:
        gpu_num = job_dict['num_gpu']
    global_batch_size = str(job_dict['batch_size'])
    found_flag = False
    if gpu_num == 0:
        found_flag = True
    new_gpu_num = gpu_num
    for num in THROUGHPUTS[job_dict['model']['name']][global_batch_size]:
        if int(num) > job_dict['max_gpu']:
            break
        if int(num) == gpu_num:
            found_flag = True
            continue
        if found_flag:
            new_throughput = float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][num])
            new_gpu_num = int(num)
            break
    return new_gpu_num


def get_last_level(job_dict, gpu_num=None):
    assert gpu_num != 0
    if gpu_num is None:
        gpu_num = job_dict['num_gpu']
    global_batch_size = str(job_dict['batch_size'])
    new_gpu_num = 0
    for num in THROUGHPUTS[job_dict['model']['name']][global_batch_size]:
        if int(num) > job_dict['max_gpu'] or int(num) >= gpu_num:
            break
        new_gpu_num = int(num)
    return new_gpu_num


def align_to_time_slot(start_time, event_time, time_unit):
    slot_num = (event_time - start_time) // time_unit
    if (event_time - start_time) % time_unit > 0:
        slot_num += 1
    return start_time + slot_num * time_unit


# deprecated
def estimate_base_quota(event_time, job_dict, iter_left):
    global_batch_size = str(job_dict['batch_size'])
    for throughput in THROUGHPUTS[job_dict['model']['name']][global_batch_size]:
        gpu_num = int(throughput)
        if gpu_num > job_dict['max_gpu']:
            break
        execution_time = iter_left / float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][throughput])
        if execution_time + event_time <= job_dict['ddl']:
            return gpu_num
    return CLUSTER.num_gpu + 1

def fetch_GPU_list_to_int(compressed_list):
        """ decode the gpu_list from the compressed integer
        @param compressed_list -> uint32 : i-th bit equals 1 means the 2^i-th gpu is used, and equals 0 otherwise
        @return decoded_list -> list of int : the gpu list that this job uses on this specific node (the gpu index is sorted increasingly)
        """
        decoded_list = []
        for i in range(32):
            if ((1 << i) & compressed_list) != 0:
                decoded_list.append(i)
        return decoded_list

def estimate_overhead(num_gpu, restart=False, cache=False):
    overhead = 2
    #return overhead
    if CLUSTER.num_gpu >= 256:
        overhead = 7
    #return overhead
    if cache:
        return overhead
    if restart:
        overhead += 27
        if CLUSTER.num_gpu >= 256:
            overhead += 20
    overhead += (math.log(num_gpu, 2) + 1) * 2
    return overhead

def ranks_to_str(rank_list):
    """ decode the rank_list from the compressed string
    @param rank_list -> uint32 : i-th char equals '1' means the 2^i-th rank is used, and equals '0' otherwise
    @return decoded_str -> str : the str that represent the rank list
    """
    decoded_list = []
    for _ in range(CLUSTER.num_gpu):
        decoded_list.append('0')
    for rank in rank_list:
        decoded_list[rank] = '1'
    return ''.join(decoded_list)

def get_nearest_share(job, gpu_num):
    valid_list = list(THROUGHPUTS[job_dict['model']['name']].keys())
    # max min
    if gpu_num in valid_list:
        return gpu_num
    found_flag = 0
    for nearest_share in valid_list:
        if nearest_share < job['min_gpu']:
            continue
        if nearest_share > job['max_gpu']:
            break
        if nearest_share > gpu_num:
            break
        found_flag = 1
    assert found_flag != 0
    return nearest_share

def get_base_constraints(x, scale_factors_array):
    """Return base constraints."""
    return [
        x >= 0,
        x <= 1,
        #cp.sum(x, axis=0) <= CLUSTER.num_gpu,
        cp.sum(cp.multiply(
                scale_factors_array, x), axis=0) <= CLUSTER.num_gpu,
    ]

def scale_factors_array(jobs):
    scale_factors_array = np.zeros((len(jobs), ))
    for i, job in enumerate(jobs):
        scale_factors_array[i] = job['num_gpu']
    return scale_factors_array

def get_isolated_throughputs(jobs):
    # time allocation
    allocation = np.array([math.ceil(CLUSTER.num_gpu / len(jobs)) for i in range((len(jobs)))])
    allocation = allocation / scale_factors_array(jobs)
    per_row_sum = np.maximum(allocation, np.ones(allocation.shape))
    allocation = allocation / per_row_sum#[:, None]
    isolated_throughputs = np.zeros((len(jobs), ), dtype=np.float64)
    for i, job in enumerate(jobs):
        isolated_throughputs[i] = float(THROUGHPUTS[job['model']['name']][str(
                    job['batch_size'])][str(job['num_gpu'])]) * allocation[i]
    isolated_throughputs = isolated_throughputs.reshape((len(jobs), 1))
    return allocation

def get_efficiency(job_dict):
    global_batch_size = str(job_dict['batch_size'])
    num = str(job_dict['num_gpu'])
    tpt = float(THROUGHPUTS[job_dict['model']['name']][global_batch_size][num])
    tpt_1 = float(THROUGHPUTS[job_dict['model']['name']][global_batch_size]['1'])
    return tpt / tpt_1


if __name__ == '__main__':
    print(get_host_ip())