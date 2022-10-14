import os, sys
import math
import random
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from server.switch import _Switch
from server.node import _Node
from utils import util
from alg.utils.topology import Topology
from .base import BasePlaceMent


class ConsolidatePlaceMent(BasePlaceMent):
    __alias__ = 'consolidate'
    def __init__(self, cluster, name, model_info):
        super(ConsolidatePlaceMent, self).__init__(cluster=cluster, name=name, model_info=model_info)
    
    
    def select_fewest_node_list(self, required_gpu_num):
        best_select_list = list()
        
        # select in a single switch
        for switch in self.cluster.switch_list:
            free_gpu_num = switch.check_free_gpus()
            reverse = False if switch.node_list[0].check_total_gpus() >= required_gpu_num else True
            if free_gpu_num >= required_gpu_num:
                # method 1
                if reverse == True:
                    node_list = sorted(switch.node_list, key=lambda e: e.check_free_gpus(), reverse=reverse)
                else:
                    nonzero = lambda x: x if x >= required_gpu_num else 1000
                    node_list = sorted(switch.node_list, key=lambda e: nonzero(e.check_free_gpus()), reverse=reverse)
                select_list = list()
                
                local_required_gpu_num = required_gpu_num
                for node in node_list:
                    free_gpu_num = node.check_free_gpus()
                    if free_gpu_num == 0: continue
                    if free_gpu_num < local_required_gpu_num:
                        node_info = (node, free_gpu_num)
                    else:
                        node_info = (node, local_required_gpu_num)

                    local_required_gpu_num -= node_info[1]
                    select_list.append(node_info)
                    if local_required_gpu_num == 0:
                        break
                
                if local_required_gpu_num == 0 and (len(best_select_list) == 0 or (len(best_select_list) > len(select_list))):
                    best_select_list = select_list
                # print('length {}, required_gpu {}'.format(len(best_select_list), required_gpu_num))

        if len(best_select_list) == 0:
            local_required_gpu_num = required_gpu_num
            reverse = False if self.cluster.switch.node_list[0].check_total_gpus() >= required_gpu_num else True
            switch_list = sorted(self.cluster.switch_list, key=lambda e: e.check_free_gpus(), reverse=reverse)
            
            for switch in switch_list:
                free_gpu_num = switch.check_free_gpus()
                if free_gpu_num >= local_required_gpu_num:
                    node_list = sorted(switch.node_list, key=lambda e: e.check_free_gpus(), reverse=reverse)
                else:
                    node_list = switch.node_list

                for node in node_list:
                    free_gpu_num = node.check_free_gpus()
                    if free_gpu_num == 0:
                        continue
                    if free_gpu_num < local_required_gpu_num:
                        node_info = (node, free_gpu_num)
                    else:
                        node_info = (node, local_required_gpu_num)

                    local_required_gpu_num -= node_info[1]
                    best_select_list.append(node_info)
                    if local_required_gpu_num == 0:
                        break
                if local_required_gpu_num == 0:
                    break
        return best_select_list
  

    def place_jobs(self, job):
        '''
        consolidate first, but randomly pick machines;
        if cross machines, still try to consolidate.
        if can't consolidate, consider spreed the jobs;
        also PS is randomly placed on the selected machines
        '''
        # early return false
        if self.cluster.check_free_gpus() < job.required_gpu_num: return False

        num_ps, num_w = len(job['ps_network']), job['num_gpu']
        assert num_ps == num_w or (num_ps == 0 and num_w == 1)
        # place as few nodes as possible
        w_node_list, w_switch_list = list(), list()
        demand_node_list = self.select_fewest_node_list(num_w)
        demand_node_gpu_list = [node.check_total_gpus() for node, _ in demand_node_list]
        if sum(demand_node_gpu_list) - min(demand_node_gpu_list) >= job.required_gpu_num:
            return False

        for node_info in demand_node_list:
            node, need_gpu = node_info
            switch = node.belong_switch
            need_cpu = need_gpu * 6 if (num_w == 1 and num_ps == 0) else need_gpu * 2
            allocated = self.allocate_resource(job=job, resource={'node':node, 'switch':switch}, node_list=w_node_list, \
                switch_list=w_switch_list, gpu_num=1, cpu_num=need_cpu // need_gpu, job_num=need_gpu)
            assert allocated == True, 'should exist enough gpu resource'

        # go through workers
        assert len(w_node_list) == num_w
        # randomly place PS to node_list
        ps_node_list, ps_switch_list = list(), list()
        for i in range(num_ps):
            resource = {
                'node':w_node_list[i],
                'switch':w_node_list[i].belong_switch,
            }

            allocated = self.allocate_resource(job=job, resource=resource, node_list=ps_node_list, \
                                            switch_list=ps_switch_list, gpu_num=0, cpu_num=4, job_num=1)
            if allocated == False:
                import pdb; pdb.set_trace()
            assert allocated == True, 'should have enough resource to run'

        #go through all the related nodes
        # node_list = list() # useful for network load, but now no use
        # for i in range(len(w_node_list)):
        #     self.update_node_list_info(w_node_list[i], node_list, worker=1, ps=0)

        # for i in range(len(ps_node_list)):
        #     self.update_node_list_info(ps_node_list[i], node_list, worker=0, ps=i)

        # rocess job placement information
        
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

