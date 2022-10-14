import os, sys
import math
import random
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from server.switch import _Switch
from server.node import _Node
from utils import util
from alg.utils.topology import Topology
from .base import BasePlaceMent

MAX_GPU_OF_SINGLE_NODE = 8

    

class PolicyPlaceMent(BasePlaceMent):
    __alias__ = 'policy'
    def __init__(self, cluster, name, model_info):
        super(PolicyPlaceMent, self).__init__(cluster=cluster, name=name, model_info=model_info)
    

    def shortcut_check_policy(self, specified_policy):
        free_resource_numeric_list = [0 for _ in range(self.num_gpu_p_node + 1)]
        for switch in self.cluster.switch_list:
            for node in switch.node_list:
                left_gpu = node.check_free_gpus()
                free_resource_numeric_list[left_gpu] += 1

        for required_gpu_num in specified_policy:
            if any([free_resource_numeric_list[gpu_num] > 0 for gpu_num in range(required_gpu_num, self.num_gpu_p_node + 1)]):
                for gpu_num in range(required_gpu_num, self.num_gpu_p_node + 1):
                    if free_resource_numeric_list[gpu_num] > 0:
                        free_resource_numeric_list[gpu_num] -= 1
                        # free_resource_numeric_list[gpu_num - required_gpu_num] += 1 # comment this statement, can't use repeated node
                        break
                continue
            return False
        return True

    def corase_place_jobs(self, job, specified_policy):
        # policy only indicate breadk down ways, not determine specified node instance_instance & switch_instance
        specified_policy.sort(reverse=True)
        num_ps, num_w = len(job['ps_network']), job['num_gpu']
        if num_ps != num_w: num_ps = num_w
        assert sum(specified_policy) == job['num_gpu']
        
        # go through workers, each worker needs single gpu and two workers
        w_node_list, w_switch_list = list(), list()
        
        # fast check resource constrain
        if not self.shortcut_check_policy(specified_policy):
            return False

        free_resource_list = [list() for _ in range(MAX_GPU_OF_SINGLE_NODE+1)] # TODO

        for switch in self.cluster.switch_list:
            for node in switch.node_list:
                left_gpu = node.check_free_gpus()
                resource = {'node':node, 'switch':switch}
                free_resource_list[left_gpu].append(resource)
        
        for required_gpu_num in specified_policy:
            allocated = False
            # select resource to run
            for gpu_num in range(required_gpu_num, MAX_GPU_OF_SINGLE_NODE+1):
                if len(free_resource_list[gpu_num]) > 0:
                    for i in range(required_gpu_num):
                        node = free_resource_list[gpu_num][-1]['node']
                        cpu_num = 6 if (num_w == 1 and num_ps == 0) else 2
                        local_allocated = self.allocate_resource(job=job, resource=free_resource_list[gpu_num][-1], \
                                                node_list=w_node_list, switch_list=w_switch_list, gpu_num=1, cpu_num=cpu_num, job_num=1)
                        assert local_allocated == True, 'should have enough resource'
                    resource = free_resource_list[gpu_num].pop()
                    free_resource_list[resource['node'].check_free_gpus()].append(resource)
                    allocated = True
                    break
            
            assert allocated == True, 'should find enough resource'
        
        # go through PS, worker requires 4 cpu
        ps_node_list, ps_switch_list = list(), list()
        if num_ps > len(w_node_list): 
            import pdb; pdb.set_trace()

        for i in range(num_ps):
            resource = {
                'node':w_node_list[i],
                'switch':w_node_list[i].belong_switch,
            }
            # print('free_resource', resource['switch'].id, resource['node'].id, resource['node'].check_free_cpus(), resource['node'].check_free_gpus())
            allocated = self.allocate_resource(job=job, resource=resource, node_list=ps_node_list, \
                                            switch_list=ps_switch_list, gpu_num=0, cpu_num=4, job_num=1)
            assert allocated == True, 'should have enough resource to run'
   
        
        # process job placement information
        for i, (s_id, node) in enumerate(zip(w_switch_list, w_node_list)):
            node_dict = {
                'id' : node.id, 
                'node_instance' : node, 
                'num_gpu' : 1,
                'num_cpu' : 6 if (num_w == 1 and num_ps == 0) else 2,
                'mem' : job['model']['mem_util'],
                'tasks': list(), 
            }
            job['placements'].append({
                'switch' : s_id, 
                'nodes' : [node_dict],
            })

        

        for i, (s_id, node) in enumerate(zip(ps_switch_list, ps_node_list)):
            node_dict = {
                'id' : node.id,
                'node_instance' : node,  
                'num_gpu' : 0, 
                'num_cpu' : 4, 
                'mem' : 0, # job['model']['mem_util'], fix a bug
                'tasks' : list(), 
            }
            job['placements'].append({
                'switch': s_id, 
                'nodes' : [node_dict]
            })
        job['topology'] = Topology(job=job, placements=job['placements'])
        return True

    
    def fine_place_jobs(self, job, specified_policy=None):
        num_ps, num_w = len(job['ps_network']), job['num_gpu']
        if num_ps != num_w: num_ps = num_w
        w_node_list, w_switch_list = list(), list()

        for policy_info in specified_policy:
            resource = policy_info['resource']
            gpu_num = policy_info['required_gpu_num']
            for _ in range(gpu_num):
                node = resource['node']
                switch = resource['switch']
                # print('resource information: ', switch.id, node.id, node.check_free_gpus(), node.check_free_cpus())
                cpu_num = 6 if (num_w == 1 and num_ps == 0) else 2
                local_allocated = self.allocate_resource(job=job, resource=resource, \
                            node_list=w_node_list, switch_list=w_switch_list, gpu_num=1, cpu_num=cpu_num, job_num=1)
                assert local_allocated == True, 'should have enough resource'
        
        # go through PS, worker requires 4 cpu
        ps_node_list, ps_switch_list = list(), list()
        for i in range(num_ps):
            resource = {
                'node':w_node_list[i],
                'switch':w_node_list[i].belong_switch,
            }
            # print('free_resource', resource['switch'].id, resource['node'].id, resource['node'].check_free_cpus(), resource['node'].check_free_gpus())
            allocated = self.allocate_resource(job=job, resource=resource, node_list=ps_node_list, \
                                            switch_list=ps_switch_list, gpu_num=0, cpu_num=4, job_num=1)
            assert allocated == True, 'should have enough resource to run'
   
        
        
        # process job placement information
        for i, (s_id, node) in enumerate(zip(w_switch_list, w_node_list)):
            node_dict = {
                'id' : node.id, 
                'node_instance' : node, 
                'num_gpu' : 1,
                'num_cpu' : 6 if (num_w == 1 and num_ps == 0) else 2,
                'mem' : job['model']['mem_util'],
                'tasks': list(), 
            }
            job['placements'].append({
                'switch' : s_id, 
                'nodes' : [node_dict],
            })

        

        for i, (s_id, node) in enumerate(zip(ps_switch_list, ps_node_list)):
            node_dict = {
                'id' : node.id,
                'node_instance' : node,  
                'num_gpu' : 0, 
                'num_cpu' : 4, 
                'mem' : 0, # job['model']['mem_util'], fix a bug
                'tasks' : list(), 
            }
            job['placements'].append({
                'switch': s_id, 
                'nodes' : [node_dict]
            })
        job['topology'] = Topology(job=job, placements=job['placements'])
        return True




    def place_jobs(self, job, specified_policy=None):
        assert isinstance(specified_policy, list)
        if isinstance(specified_policy[0], int):
            return self.corase_place_jobs(job, specified_policy)

        elif isinstance(specified_policy[0], dict) and \
            'resource' in specified_policy[0] and 'required_gpu_num' in specified_policy[0]:  # {'resource': resource, 'required_gpu_num', required_gpu_num }
            return self.fine_place_jobs(job, specified_policy)

        else:
            raise NotImplementedError
