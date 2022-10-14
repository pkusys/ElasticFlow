import math
import socket
import sys
import subprocess


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
    if cache:
        return overhead
    if restart:
        overhead += 27
    overhead += (math.log(num_gpu, 2) + 1) * 2
    return overhead


def allocate_set(gpu_num):
    return profile_allocate_set(gpu_num)
    if gpu_num == 1:
        return [ [1] ]
    if gpu_num == 2:
        # return [ [2], [1, 1]]
        return [[2]]
    if gpu_num == 3:
        # return [ [3], [2, 1], [1, 1, 1]]
        return [[3]]
    if gpu_num == 4:
        return [[4]]
        # return [ [4],  [2, 2], [2, 1, 1], [1, 1, 1, 1]]
    if gpu_num == 5:
        # return [ [5], [4, 1], [3, 2], [3, 1, 1], [2, 2, 1]]
        return [[5]]
    if gpu_num == 6:
        # return [ [6], [4, 2], [3, 3], [4, 1, 1], [2, 2, 2]]
        return [[6]]
    if gpu_num == 7:
        # return [ [7], [6, 1], [5, 2], [4, 3], [5, 1, 1], [4, 2, 1], [3, 3, 1], [3, 2, 2]]
        return [[7]]
    if gpu_num == 8:
        # return [ [8], [6, 2], [4, 4], [6, 1, 1], [4, 2, 2]]
        return [[8]]
    return list()


def profile_allocate_set(gpu_num):
    if gpu_num == 1:
        return [ [1] ]
    if gpu_num == 2:
        return [ [2], [1, 1]]
    if gpu_num == 3:
        return [ [3], [2, 1]]
    if gpu_num == 4:
        return [ [4],  [2, 2]]
    if gpu_num == 5:
        return [ [5], [3, 2], [2, 2, 1]]
    if gpu_num == 6:
        return [ [6], [4, 2], [2, 2, 2]]
    if gpu_num == 7:
        return [ [7], [6, 1], [4, 3], [3, 2, 2]]
    if gpu_num == 8:
        return [ [8],[4, 4], [6, 2], [4, 2, 2], [2, 2, 2, 2]]
    return list()

def estimate_overhead(num_gpu, restart=False, cache=False):
    overhead = 2
    if cache:
        return overhead
    if restart:
        overhead += 27
    overhead += (math.log(num_gpu, 2) + 1) * 2
    return overhead


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