import os
import sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
import copy
import numpy as np
from utils.util import profile_allocate_set, allocate_set


def sanity_check_gpu_list(all_gpu_list, all_resource_list):
    for idx, all_gpu in enumerate(all_gpu_list):
        true_gpu = all_resource_list[idx]['node'].check_free_gpus()
        assert true_gpu >= all_gpu
    return True


def GreedyPolicy(job_list, all_gpu_list=None, all_resource_list=None, key_function=None):
    policy_with_node_info = False
    if all_resource_list is not None:
        policy_with_node_info = True

    assert all_gpu_list is not None
    all_gpu_list = copy.deepcopy(all_gpu_list)
    
    if key_function is None:
        job_list = sorted(job_list, key=lambda job: job.max_reward - job.min_reward, reverse=True)
    else:
        job_list = sorted(job_list, key=key_function, reverse=True)

    reward = 0
    allocation_policy = list()
    # import pdb; pdb.set_trace()
    for job in job_list:
        assert sum(all_gpu_list) >= job['num_gpu']
        done = False
        if job['num_gpu'] in all_gpu_list:
            select_id = None
            for idx, free_gpu in enumerate(all_gpu_list):
                if free_gpu == job['num_gpu']:
                    select_id = idx
                    break
            all_gpu_list[select_id] -= free_gpu
            reward += job.optimistic_placement_reward([job['num_gpu'], ]) 
            done = True
            # print('condition 1, job {}, resource id {}'.format(job['job_id'], [select_id]))
            if policy_with_node_info: 
                allocation_policy.append( (job['job_id'], 
                                         [dict({'required_gpu_num':job['num_gpu'], 'resource':all_resource_list[select_id]}) ])
                                        )
            else:
                allocation_policy.append((job['job_id'], [job['num_gpu']])) 

        elif max(all_gpu_list) > job['num_gpu']:
            select_id = None
            max_gpu = None # max(all_gpu_list)
            for idx, free_gpu in enumerate(all_gpu_list):
                if free_gpu >= job['num_gpu']:
                    if max_gpu is None: 
                        max_gpu = free_gpu 
                        select_id = idx
                    if free_gpu < max_gpu: 
                        max_gpu = free_gpu 
                        select_id = idx
                # if free_gpu == max_gpu:
                #     select_id = idx
                #     break

            all_gpu_list[select_id] -= job['num_gpu']
            # print('condition 2, job {}, resource id {}'.format(job['job_id'], [select_id]))
            done = True
            reward += job.optimistic_placement_reward(list([job['num_gpu']]))
            if policy_with_node_info: 
                allocation_policy.append( (job['job_id'], 
                                        [dict({'required_gpu_num':job['num_gpu'], 'resource':all_resource_list[select_id]}) ])
                                        )
            else:
                allocation_policy.append((job['job_id'], [job['num_gpu']])) 

        else: # careful about 4, 4, but not exsits in current scenerio
            select_gpu1 = max(all_gpu_list)
            for idx1, free_gpu1 in enumerate(all_gpu_list):
                if free_gpu1 == select_gpu1:
                    left_gpu = job.required_gpu_num - select_gpu1

                    select_max_gpu2, select_max_id2 = None, None
                    select_min_gpu2, select_min_gpu2 = None, None

                    for idx2, free_gpu2 in enumerate(all_gpu_list):
                        if idx2 != idx1 and free_gpu2 >= left_gpu:
                            # update max
                            if select_max_gpu2 is None:
                                select_max_gpu2 = free_gpu2
                                select_max_id2 = idx2
                            elif select_max_gpu2 < free_gpu2:
                                select_max_gpu2 = free_gpu2
                                select_max_id2 = idx2
                            # update min
                            if select_min_gpu2 is None:
                                select_min_gpu2 = free_gpu2
                                select_min_id2 = idx2
                            elif select_min_gpu2 > free_gpu2:
                                select_min_gpu2 = free_gpu2
                                select_min_id2 = idx2
                        
                    if select_max_gpu2 is not None:
                        done = True
                        if select_min_gpu2 == left_gpu:
                            if policy_with_node_info: 
                                policy = [
                                    dict({'required_gpu_num':select_gpu1, 'resource': all_resource_list[idx1]}), 
                                    dict({'required_gpu_num': left_gpu, 'resource': all_resource_list[select_min_id2]})
                                ]
                            else:
                                policy = [select_gpu1, left_gpu]

                            all_gpu_list[idx1] -= select_gpu1
                            all_gpu_list[select_min_id2] -= left_gpu
                            # print('condition 3, job {}, resource id {}'.format(job['job_id'], [idx1, select_min_id2]))
                            
                        else:
                            # select min 
                            if policy_with_node_info: 
                                policy = [
                                        dict({'required_gpu_num':select_gpu1, 'resource': all_resource_list[idx1]}), 
                                        # dict({'required_gpu_num': left_gpu, 'resource': all_resource_list[select_max_id2]})
                                        dict({'required_gpu_num': left_gpu, 'resource': all_resource_list[select_min_id2]})
                                    ]
                            else:
                                policy = [select_gpu1, left_gpu]

                            all_gpu_list[idx1] -= select_gpu1
                            # all_gpu_list[select_max_id2] -= left_gpu
                            all_gpu_list[select_min_id2] -= left_gpu
                            # print('condition 4, job {}, resource id {}'.format(job['job_id'], [idx1, select_max_id2]))
                            
                        reward += job.optimistic_placement_reward(policy)
                        allocation_policy.append((job['job_id'], policy))
                    break

            # random select  
                      
            if not done:
                assert sum(all_gpu_list) >= job.required_gpu_num
                left_gpu = job.required_gpu_num
                policy = list()
                while not done:
                    max_id = np.argmax(all_gpu_list)
                    if left_gpu >= all_gpu_list[max_id]:
                        left_gpu -= all_gpu_list[max_id]
                        if policy_with_node_info: 
                            policy.append(dict({'required_gpu_num': all_gpu_list[max_id], 'resource':all_resource_list[max_id]}))
                        else:
                            policy.append(all_gpu_list[max_id])

                        all_gpu_list[max_id] = 0
                        # print('condition 5, job {}, resource id {}', job['job_id'], [max_id])
                    else:
                        if policy_with_node_info:
                            policy.append(dict({'required_gpu_num': left_gpu, 'resource':all_resource_list[max_id]}))
                        else:
                            policy.append(left_gpu)

                        all_gpu_list[max_id] -= left_gpu
                        # print('condition 5, job {}, resource id {}'.format(job['job_id'], [max_id]))
                            
                        left_gpu = 0
                    if left_gpu == 0:
                        done = True
                allocation_policy.append((job['job_id'], policy))
                reward += job.optimistic_placement_reward(policy)
    
    return reward, allocation_policy


def find_min_with_limit(val_list, val, filter_index):
    selected_idx = -1
    selected_val = -1
    for idx, ival in enumerate(val_list):
        if idx not in filter_index and ival >= val:
            if selected_idx == -1:
                selected_idx = idx 
                selected_val = ival 
            elif ival < selected_val: 
                selected_val = ival 
                selected_idx = idx
            
    return selected_idx


def GreedyPolicyWithConstrain(job_list, all_gpu_list=None, all_resource_list=None, key_function=None):
    return GreedyPolicy(job_list, all_gpu_list, all_resource_list, key_function)
    policy_with_node_info = False
    if all_resource_list is not None:
        policy_with_node_info = True

    assert all_gpu_list is not None
    all_gpu_list = copy.deepcopy(all_gpu_list)
    
    if key_function is None:
        job_list = sorted(job_list, key=lambda job: job.max_reward - job.min_reward, reverse=True)
    else:
        job_list = sorted(job_list, key=key_function, reverse=True)

    reward = 0
    allocation_policy = list()

    for job in job_list:
        assert sum(all_gpu_list) >= job['num_gpu']
        done = True 
        # constrain_policy_set = profile_allocate_set(job['num_gpu'])
        constrain_policy_set = allocate_set(job['num_gpu'])
        for policy_list in constrain_policy_set:
            selected_idx_list = list() 
            done = True
            for selected_gpu in policy_list:
                selected_idx = find_min_with_limit(all_gpu_list, selected_gpu, selected_idx_list)
                
                if selected_idx == -1: 
                    done = False
                    break
                selected_idx_list.append(selected_idx)
            if not done: continue 
            policy = list() 
            for (selected_idx, selected_gpu) in zip(selected_idx_list, policy_list):
                all_gpu_list[selected_idx] -= selected_gpu 
                if policy_with_node_info: 
                    policy.append(
                        dict({'required_gpu_num':selected_gpu, 'resource': all_resource_list[selected_idx]})
                    )
                else:
                    policy.append(selected_gpu)
            allocation_policy.append((job['job_id'], policy))
            reward += job.optimistic_placement_reward(policy)
            break
    
    return reward, allocation_policy


