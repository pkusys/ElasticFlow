import os, sys
import copy 
import math
import random
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from server.switch import _Switch
from server.node import _Node
from utils import util
from alg.utils.topology import Topology
from .base import BasePlaceMent
from .greedy import GreedyPolicy, GreedyPolicyWithConstrain
from .policy import PolicyPlaceMent
from .consolidate import ConsolidatePlaceMent
from .random import RandomPlaceMent

SEARCH_SPACE = 10_000


def sanity_check_gpu_list(all_gpu_list, all_resource_list):
    for idx, all_gpu in enumerate(all_gpu_list):
        true_gpu = all_resource_list[idx]['node'].check_free_gpus()
        assert true_gpu == all_gpu
    return True


def policy_summary(policy_list, summary_info):
    print(summary_info)
    for (job_id, policy) in policy_list:
            print('job info: job_id: {}, gpu_num: {}'.format(job_id, None))
            for policy_info in policy:
                resource = policy_info['resource']
                node = resource['node']
                switch = resource['switch']
                print('resource information: allocate gpu {},  switch id {}, node id {}, free_gpu {}, free_cpu {}'.format(policy_info['required_gpu_num'], switch.id, node.id, node.check_free_gpus(), node.check_free_cpus()))
        
    print('-'*20)


def max_reward_policy(job_list, all_gpu_list, all_resource_list, logger):
    reward_policy1 = GreedyPolicyWithConstrain(job_list, all_gpu_list=all_gpu_list, all_resource_list=all_resource_list, key_function=lambda job: job.max_reward-job.min_reward)
    reward_policy2 = GreedyPolicyWithConstrain(job_list, all_gpu_list=all_gpu_list, all_resource_list=all_resource_list, key_function=lambda job: job.required_gpu_num)
    reward_policy3 = GreedyPolicyWithConstrain(job_list, all_gpu_list=all_gpu_list, all_resource_list=all_resource_list, key_function=lambda job: -job.required_gpu_num)
    reward_policy4 = GreedyPolicyWithConstrain(job_list, all_gpu_list=all_gpu_list, all_resource_list=all_resource_list, key_function=lambda job: job.max_reward)

    # policy_summary(reward_policy1[1], 'with greedy policy 1')
    # policy_summary(reward_policy2[1], 'with greedy policy 2')
    # policy_summary(reward_policy3[1], 'with greedy policy 3')
    maxval = max([reward_policy1[0], reward_policy2[0], reward_policy3[0], reward_policy4[0]])
    minval = min([reward_policy1[0], reward_policy2[0], reward_policy3[0], reward_policy4[0]])
    logger.info('reward diff -- {}'.format(maxval - minval))
    return max([reward_policy1, reward_policy2, reward_policy3, reward_policy4], key=lambda x: x[0])



def bin_coarse_check(policy_list, gpu_list):
    bin_set = {}
    gpu_set = {}
    for i in range(1, 9): bin_set[i] = gpu_set[i] = 0
    
    for gpu_num in gpu_list: 
        if gpu_num != 0: 
            gpu_set[gpu_num] += 1

    for _, policy in policy_list:
        for gpu_num in policy:
            bin_set[gpu_num] += 1
    a, b = 0, 0
    for i in range(1, 9):
        a += bin_set[i] * i
        b += gpu_set[i] * i
    

    for job_gpu in range(8, 0, -1):
        delta_gpu = min(bin_set[job_gpu], gpu_set[job_gpu])
        bin_set[job_gpu] -= delta_gpu
        gpu_set[job_gpu] -= delta_gpu

        if bin_set[job_gpu] > 0:
            for j in range(8, job_gpu - 1, -1):
                delta_gpu = min(bin_set[job_gpu], gpu_set[j])
                bin_set[job_gpu] -= delta_gpu
                gpu_set[j] -= delta_gpu
                if (j - job_gpu) != 0:
                    gpu_set[j - job_gpu] += delta_gpu

        if bin_set[job_gpu] > 0:
            return False
    return True


def allocate_resource(policy_list, job_without_policy_list, all_resource_list):
    all_gpu_list = [resource['node'].check_free_gpus() for resource in all_resource_list]
    # if not bin_coarse_check(policy_list, all_gpu_list):
    #     return False, None, None

    # sorted by allocation policy, (7,) >= (4, 3)
    policy_list = sorted(policy_list, key=lambda x: (len(x[1]), x[1][0]), reverse=True) 
    # (job_id, resource_info)
    policy_with_node_info_list = list()

    reward = 0
    for job, policy in policy_list:
        reward += job.optimistic_placement_reward(policy)
        forbidden = list()
        resource_info_list = list() # {'required_gpu_num':job['num_gpu']}

        for client_gpu_num in policy:
            startn_id = random.randint(0, len(all_gpu_list) - 1)
            max_select_id, max_select_gpu_num = None, None
            min_select_id, min_select_gpu_num = None, None
            for nid, server_gpu_num in enumerate(all_gpu_list):
                if server_gpu_num >= client_gpu_num and nid not in forbidden:
                    # select max 
                    if max_select_id is None:
                        max_select_id = nid
                        max_select_gpu_num = server_gpu_num
                    elif server_gpu_num > max_select_gpu_num:
                        max_select_gpu_num = nid
                        max_select_gpu_num = server_gpu_num
                    # select min
                    if min_select_id is None:
                        min_select_id = nid
                        min_select_gpu_num = server_gpu_num
                    elif min_select_gpu_num > server_gpu_num:
                        min_select_id = nid
                        min_select_gpu_num = server_gpu_num
                    
            if min_select_id is not None and min_select_gpu_num == client_gpu_num:
                forbidden.append(min_select_id)
                all_gpu_list[min_select_id] -= client_gpu_num
                
                resource_info_list.append(dict(
                    {'required_gpu_num': client_gpu_num, 
                    'resource': all_resource_list[min_select_id]}
                ))

            elif max_select_gpu_num is not None:
                forbidden.append(max_select_id)
                all_gpu_list[max_select_id] -= client_gpu_num
                resource_info_list.append(dict(
                    {'required_gpu_num': client_gpu_num, 
                    'resource': all_resource_list[max_select_id]}
                ))
            else:
                return False, None, None
        policy_with_node_info_list.append((job['job_id'], resource_info_list))    
    
    
    remaining_reward, remaining_policy = max_reward_policy(job_without_policy_list, all_gpu_list, all_resource_list)
    for (job_id, policy) in policy_with_node_info_list:
        remaining_policy.append((job_id, policy))
    reward += remaining_reward
    return True, reward, remaining_policy

    

class LocalSearchPlaceMent(BasePlaceMent):
    __alias__ = 'local_search'
    def __init__(self, cluster, name, model_info):
        super(LocalSearchPlaceMent, self).__init__(cluster=cluster, name=name, model_info=model_info)
        self.all_node_list = list()
        self.policy_placement = PolicyPlaceMent(cluster=cluster, name='policy_placement_for_two_stage', model_info=model_info)
        self.consolidate_placement = RandomPlaceMent(cluster=cluster, name='consolidate_placement_for_two_stage', model_info=model_info)
        for switch in self.cluster.switch_list:
            for node in switch.node_list:
                self.all_node_list.append({'switch': switch, 'node':node})
    

    def init_search_state(self, ):
        self.search_max_reward = -sys.maxsize
        self.search_max_policy = list()
        self.success_policy = 0
        self.all_gpu_list = [node['node'].check_free_gpus() for node in self.all_node_list]
        

    
    def brute_force_search(self, search_list, remaining_list, alpha_function, allocate_policy, depth, prefix, lower_bound):
        if depth > len(search_list):
            success, reward, policy = allocate_resource(allocate_policy, remaining_list, self.all_node_list)
            # print('success == {}, reward == {}'.format(success, reward))
            if success and reward > self.search_max_reward:
                self.search_max_reward = reward
                self.success_policy += 1
                self.search_max_policy = policy
                        
        else:
            job = search_list[depth - 1]
            for i in range(len(job.reward_list) - 1, -1, -1):
                reward, policy = job.reward_list[i], job.policy_list[i]
                if prefix + reward + alpha_function[depth] > lower_bound:
                    allocate_policy.append((job, policy))
                    self.brute_force_search(search_list, remaining_list,  alpha_function, allocate_policy, depth+1, prefix+reward, lower_bound)
                    allocate_policy.pop()


    def place_jobs(self, all_job_list, logger):
        assert isinstance(all_job_list, list), 'should provide a batch of jobs'
        logger.info('batch processing {} job'.format(len(all_job_list)))

        job_list = list()
        for job in all_job_list:
            if job.required_gpu_num <= 8:
                job_list.append(job)
                continue
            assert self.consolidate_placement.place_jobs(job) == True
        
        if len(job_list) == 0:
            return True
        
        # init local search state
        self.init_search_state()
        
        # fully greedy algorithm
        
        greedy_reward, greedy_policy = max_reward_policy(job_list, self.all_gpu_list, self.all_node_list, logger)
        if greedy_policy == 0:
            import pdb; pdb.set_trace()
        
        possible_max_reward = sum([job.max_reward for job in job_list])
        if possible_max_reward != greedy_reward:
            logger.info('greedy_reward {}'.format(greedy_reward))
            logger.info('possible max reward {}'.format(sum([job.max_reward for job in job_list])))
            for idx, job in enumerate(job_list):
                show_policies = [show_policy['required_gpu_num'] for show_policy in greedy_policy[idx][1]]
                logger.info('reward {} / gpu_num {} / policy {}'.format(job.max_reward, job.required_gpu_num, show_policies))
                logger.info('gpu_list {}'.format([gpu for gpu in self.all_gpu_list if gpu != 0]))
        

        for (job_id, policy) in greedy_policy:
            job = util.search_dict_list(job_list, 'job_id', job_id)
            # print(job.required_gpu_num, self.cluster.check_free_gpus())
            # assert self.policy_placement.place_jobs(job, specified_policy=policy) == True
            if not self.policy_placement.place_jobs(job, specified_policy=policy):
                import pdb; pdb.set_trace()

        return True

        # filter uncertanity list
        uncertainity_list, remaining_list = list(), list()
        job_list.sort(key=lambda job: job.max_min_diff, reverse=True)
        prod = 1
        for idx, job in enumerate(job_list):
            prod = prod * len(job.reward_list)
            if prod > SEARCH_SPACE or len(job.reward_list) == 1:
                for j in range(idx, len(job_list)):
                    remaining_list.append(job_list[j])
                break
            else:
                uncertainity_list.append(job)

        logger.info('prod == {}'.format(prod))

        # possible max reward of left gpus
        max_remaining_list_reward = max_reward_policy(remaining_list, self.all_gpu_list, self.all_node_list)[0] if len(remaining_list) > 0 else 0
        alpha_function = [job.max_reward for job in uncertainity_list]
        for i in range(len(alpha_function)):
            alpha_function[i] = sum(alpha_function[i:]) + max_remaining_list_reward
        alpha_function.append(max_remaining_list_reward)
        
        
        # local search process
        self.init_search_state()
        # if greedy_reward < sum([job.max_reward for job in job_list]) * 0.9:
        #     self.brute_force_search(search_list=uncertainity_list, 
        #                             remaining_list=remaining_list,
        #                             alpha_function=alpha_function, 
        #                             allocate_policy=list(),
        #                             depth=1,
        #                             prefix=0,
        #                             lower_bound=greedy_reward
        #                     )

        # print('useless search max_reward {} / greedy_reward {}'.format(self.search_max_reward, greedy_reward))
        if self.search_max_reward > greedy_reward: print('search method success')
        if self.search_max_reward < greedy_reward:
            self.search_max_reward = greedy_reward
            self.search_max_policy = greedy_policy
            # print('search max_reward {} / greedy_reward {}'.format(self.search_max_reward, greedy_reward))
        
        for (job_id, policy) in self.search_max_policy:
            job = util.search_dict_list(job_list, 'job_id', job_id)
            # print(job.required_gpu_num, self.cluster.check_free_gpus())
            # assert self.policy_placement.place_jobs(job, specified_policy=policy) == True
            if not self.policy_placement.place_jobs(job, specified_policy=policy):
                import pdb; pdb.set_trace()
        


        
        return True
