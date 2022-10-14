import os, sys
import math
import random
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from server.switch import _Switch
from server.node import _Node
from utils import util
from alg.utils.topology import Topology
from .base import BasePlaceMent


class RandomPlaceMent(BasePlaceMent):
    __alias__ = 'random'
    def __init__(self, cluster, name, model_info):
        super(RandomPlaceMent, self).__init__(cluster=cluster, name=name, model_info=model_info)
    

    def place_jobs(self, job):    
        '''
        randomly pick up enough resource for both PS and worker in a job
        allocate one by one
        '''
        # early return false
        if self.cluster.check_free_gpus() < job.required_gpu_num: return False

        num_ps, num_w = len(job['ps_network']), job['num_gpu']
        assert num_ps == num_w or (num_ps == 0 and num_w == 1)

        # go through workers, each worker needs single gpu and two workers
        w_node_list, w_switch_list = list(), list()
        
        for w in range(num_w):
            start_ngid = random.randint(0, self.num_node - 1) 
            allocated = False
            for i in range(self.num_node):
                n_gid = (start_ngid + i) % self.num_node
                resource = self.get_node_with_gid(n_gid)
                node, free_gpu = resource['node'], resource['node'].check_free_gpus()
                if free_gpu > 0:
                    cpu_num = 6 if (num_w == 1 and num_ps == 0) else 2
                    allocated = allocated or self.allocate_resource(job=job, resource=resource, node_list=w_node_list, \
                                                                    switch_list=w_switch_list, gpu_num=1, cpu_num=cpu_num, job_num=1)
                    # print('switch free_resource', resource['switch'].id, resource['node'].id, resource['node'].check_free_cpus(), resource['node'].check_free_gpus())
            

                if allocated == True: break
                
            if allocated == False:
                assert False, 'should not run here'
                for node in w_node_list:
                    assert node.release_job_gpu_cpu(num_gpu=1, num_cpu=2, job=job) == True
                return False # short-cut return


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
            
        
        
        # end of worker node & ps node

        # node_list = list() # useful for network load, but now no use
        # for i in range(len(w_node_list)):
        #     self.update_node_list_info(w_node_list[i], node_list, worker=1, ps=0)

        # for i in range(len(ps_node_list)):
        #     self.update_node_list_info(ps_node_list[i], node_list, worker=0, ps=i)

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

        