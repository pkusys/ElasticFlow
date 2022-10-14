import os, sys
import math
import random
from .switch import _Switch
from .node import _Node
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..')
from utils import util

# TODO: latent bug
class _Cluster(object):

    def __init__(self, num_switch=0, num_node_p_switch=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_node=0):
        ''' Init GPU cluster with basic switch, node, gpu information'''
        self.set_spec(num_switch=num_switch, num_node_p_switch=num_node_p_switch, \
                    num_gpu_p_node=num_gpu_p_node, num_cpu_p_node=num_cpu_p_node, mem_p_node=mem_p_node)

        #for non-placement
        self.switch_list = list()
        #for gandiva
        self.set_node_group()
        
        # flag init
        self.ir_init = False
        self.init = False
 

    def set_node_group(self, ):
        self.free_nodes = list()
        self.node_g = dict()
        for i in [1, 2, 4, 8, 12, 16, 24, 32, 64]:
            setattr(self, 'node_g{}'.format(i), list())
            self.node_g[i] = getattr(self, 'node_g{}'.format(i))
        
    
    def set_spec(self, num_switch=0, num_node_p_switch=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_node=0):
        self.num_switch =  num_switch
        self.num_node_p_switch = num_node_p_switch
        self.num_gpu_p_node = num_gpu_p_node
        self.num_cpu_p_node = num_cpu_p_node
        self.mem_p_node = mem_p_node
        self.num_node = num_switch * num_node_p_switch
        self.num_gpu = self.num_node * num_gpu_p_node
        self.num_cpu = self.num_node * num_cpu_p_node
        self.free_gpu = self.num_gpu
        self.mem = self.num_node * mem_p_node


    def print_cluster_spec(self):
        print('Custer Spec')
        print('#ofswitch: %d, #ofnode: %d, #ofgpu: %d, #ofcpu: %d, #ofmem: %d'%(self.num_switch, self.num_node, self.num_gpu, self.num_cpu, self.mem))

    def init_infra(self, num_switch=0, num_node_p_switch=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_node=0):
        assert self.init == False and self.ir_init == False, 'only init once'
        self.init = True
        # Init and create cluster infration entities (switches, nodes) by using class _Switch, _Node
        self.set_spec(num_switch, num_node_p_switch, num_gpu_p_node, num_cpu_p_node, mem_p_node)

        # create/init switch and node objects    
        for switch_id in range(self.num_switch):
            switch_instance = _Switch(switch_id, self.num_node_p_switch, self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_node) 
            switch_instance.add_nodes(self.num_node_p_switch, self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_node, self)
            self.switch_list.append(switch_instance)


    def set_ir_spec(self, cluster_info):
        assert self.init == False and self.ir_init == False, 'only init once'
        self.ir_init = True

        self.num_switch =  len(cluster_info.keys())
        self.num_node_p_switch = dict()
        self.num_gpu_p_node = dict()
        self.num_cpu_p_node = dict()
        self.mem_p_node = dict()
        for switch_name in cluster_info.keys():
            assert 'switch' in switch_name, 'switch must exists in {}'.format(switch_name)
            self.num_node_p_switch[switch_name] = len(cluster_info[switch_name])
            switch_info = cluster_info[switch_name]
            for node_name in switch_info.keys():
                assert node_name not in self.num_gpu_p_node, 'exists same node name which is not allowed'
                self.num_gpu_p_node[node_name] = switch_info[node_name]['num_gpu']
                self.num_cpu_p_node[node_name] = switch_info[node_name]['num_cpu']
                self.mem_p_node[node_name] = switch_info[node_name]['mem']

        self.num_node = len(self.num_gpu_p_node)
        self.num_gpu = sum([self.num_gpu_p_node[node_name] for node_name in self.num_gpu_p_node.keys()])
        self.num_cpu = sum([self.num_cpu_p_node[node_name] for node_name in self.num_cpu_p_node.keys()])
        self.free_gpu = self.num_gpu
        self.mem = sum([self.mem_p_node[node_name] for node_name in self.num_cpu_p_node.keys()])


    def ir_init_infra(self, cluster_info):
        # Init and create cluster infration entities (switches, nodes) by using class _Switch, _Node
        self.set_ir_spec(cluster_info)

        # create/init switch and node objects    
        for switch_name in cluster_info.keys():
            switch_instance = _Switch(switch_name)
            switch_instance.ir_init(cluster_info[switch_name])
            switch_instance.add_ir_nodes(cluster_info[switch_name], self)
            self.switch_list.append(switch_instance)

    
    def init_gandiva_nodes(self, ):
        # init node class
        for switch in self.switch_list:
            for node in switch.node_list:
                self.free_nodes.append(node)
        assert len(self.free_nodes) == self.num_node, '# of free nodes {}  is incorrect'.format(len(self.free_nodes))



    def release_gpus(self, job, status='END'):
        for placement in job['placements']:
            assert 'switch' in placement and 'nodes' in placement
            switch = self.switch_list[placement['switch']]
            assert switch.release_gpus(placement['nodes'], job) == True
        
        if status == 'END':
            job['status'] = 'END'
            print('**** job[%d] completed' % job['job_idx'])
        return True


    def release_job_resource(self, job, status='END'):

        for placement in job['placements']:
            assert 'switch' in placement and 'nodes' in placement
            found = False
            for switch in self.switch_list:
                if switch.id == placement['switch']:
                    found = True
                    assert switch.release_job_resource(placement['nodes'], job=job) == True
                    break
            assert found == True, 'should exist in switch list'
        
        if status == 'END': job['status'] = 'END'
        
        job['gpus'] = list()
        job['placements'] = list() # prepare an empty job_placement 
        job['topology'] = None
        return True
    
    def cluster_partition(self, user_share):
        gpu_num = self.check_total_gpus()
        print(len(user_share))
        for user, share in user_share.items():
            required_gpu_num = int(share * gpu_num)
            for switch in self.switch_list:
                for node in switch.node_list:
                    if node.permission_user_list is None:
                        node.permission_user_list = [user]
                        required_gpu_num -= node.check_total_gpus()
                        if required_gpu_num <= 0: break
                if required_gpu_num <= 0: break
            assert required_gpu_num <= 0, '{} do not have resource'.format(user)

    def check_free_gpus(self, user_name=None):
        return sum([switch.check_free_gpus(user_name) for switch in self.switch_list])

    def check_free_guarante_gpus(self, user_name=None):
        return sum([switch.check_free_guarante_gpus(user_name) for switch in self.switch_list])
    
    def check_free_spot_gpus(self, user_name=None):
        return sum([switch.check_free_spot_gpus(user_name) for switch in self.switch_list])
    
    
    def check_total_gpus(self, user_name=None):
        return sum([switch.check_total_gpus(user_name) for switch in self.switch_list])

    def check_total_guarante_gpus(self, user_name=None):
        return sum([switch.check_total_guarante_gpus(user_name) for switch in self.switch_list])

    def check_total_spot_gpus(self, user_name=None):
        return sum([switch.check_total_spot_gpus(user_name) for switch in self.switch_list])

    def check_free_cpus(self, ):
        return sum([switch.check_free_cpus() for switch in self.switch_list])

    def check_total_cpus(self, ):
        return sum([switch.check_total_cpus() for switch in self.switch_list])
        
    def gandiva_node_set_adjust(self, cur_time, jobs):
        """
        when there are free nodes in cluster, reduce burden of heavy nodes
        """
        total_gpu_demands = 0
        nl_gpu_demands = dict()
        nl_gpu_occupied = dict()
        
        for num_gpu, node_list in self.node_g.items():
            total_jobs = 0
            occupied_gpus = 0

            for node_set in node_list:
                total_jobs += len(node_set['jobs'])
                occupied_gpus += sum([node.check_total_gpus() for node in node_set['nodes']])
            
            total_gpu_demands += total_jobs * num_gpu
            nl_gpu_demands[num_gpu] = total_jobs * num_gpu
            nl_gpu_occupied[num_gpu] = occupied_gpus
        
        if total_gpu_demands == 0:
            return 
        
        for num_gpu, node_list in self.node_g.items():
            if nl_gpu_demands[num_gpu] == 0:
                continue

            nl_gpu_plan = int(math.floor(1.0 * nl_gpu_demands[num_gpu] / total_gpu_demands * self.num_gpu))
            nl_gpu_target = min(nl_gpu_plan, nl_gpu_demands[num_gpu])
            nl_gpu_diff = nl_gpu_target - nl_gpu_occupied[num_gpu]

            if nl_gpu_diff > 0:
                # growth: 
                num_ns = int(math.ceil(1. * nl_gpu_diff / num_gpu))
                expand_ns = self.gandiva_node_set_expand(num_gpu, node_list, num_ns, cur_time, jobs)
            elif nl_gpu_diff < 0:
                # shrink
                num_ns = int(math.ceil(-1. * nl_gpu_diff / num_gpu))
                shrink_ns = self.gandiva_node_set_shrink(num_gpu, node_list, num_ns, cur_time, jobs)
        

    def gandiva_node_set_shrink(self, node_group, occupied_node_list, release_node_num, cur_time, jobs):
        '''
        ns_num_gpu: num_gpu of job in this node_set
        '''
        # can't shrink too many node_set ?? why
        # decrease ns nodes
        if len(occupied_node_list) <= release_node_num:
            release_node_num = len(occupied_node_list) - 1 # at least keep single node
        
        job_list = list()
        i = 0
        for i in range(1, release_node_num + 1):
            node_set = occupied_node_list.pop(0)

            if len(node_set['jobs']) > 0:
                job_list.extend(node_set['jobs'])
                update_info = {
                    'jobs': list(), 
                    'concurrency' : 0, 
                    'util' : 0, 
                    'num_jobs' : 0,
                }
                node_set.update(update_info)
                
            for node in node_set['nodes']:
                self.free_nodes.append(node)

        for job in job_list:
            node_set = occupied_node_list[0]
            job_util = round(job['model']['mem_util'], 2)
            node_set['util'] = round(node_set['util'] + job_util, 2)
            assert job not in node_set['jobs'], 'cannot repeat  too many times'
            node_set['jobs'].append(job)
            node_set['num_jobs'] += 1
            occupied_node_list.sort(key=lambda x: x.__getitem__('util'))
        if i > 0:
            print("node_g{} shrink {} node_sets" .format(node_group, i))
        return i
    

    def gandiva_node_set_expand(self, node_group, occupied_node_list, required_node_num, cur_time, jobs):
        acquired_node_num = 0
        for acquired_node_num in range(1, required_node_num + 1):
            sorted_free_nodes = sorted(self.free_nodes, key=lambda node: node.check_free_gpus(), reverse=True)
            cum_gpus = 0
            for idx, free_node in enumerate(sorted_free_nodes):
                if cum_gpus + free_node.check_free_gpus() >= node_group:
                    node_set = {
                        'nodes' : list(), 
                        'jobs' : list(), 
                        'concurrency' : 0, 
                        'capacity' : int((cum_gpus + free_node.check_free_gpus()) * 1.0 / node_group), 
                        'util' : 0, 
                        'num_gpus': node_group, 
                        'num_jobs' : 0, 
                    }
                    for j in range(idx+1):
                        free_node = sorted_free_nodes[j]
                        self.free_nodes.remove(free_node)
                        node_set['nodes'].append(free_node)
                    
                    occupied_node_list.append(node_set)
                    break
                else:
                    cum_gpus + free_node.check_free_gpus()


        if acquired_node_num > 0: # TODO 
            job_list = list()
            for node_set in occupied_node_list:
                if len(node_set['jobs']) > 0:
                    job_list.extend(node_set['jobs'])
                    update_info = {
                        'jobs': list(), 
                        'concurrency' : 0, 
                        'util' : 0, 
                        'num_jobs' : 0,
                    }
                    node_set.update(update_info)
            
            for job in job_list:
                node_set = occupied_node_list[0]
                job_util = round(job['model']['mem_util'], 2)
                node_set['util'] = round(node_set['util'] + job_util, 2)
                assert job not in node_set['jobs'], 'cannot repeat too many times'
                node_set['jobs'].append(job)
                node_set['num_jobs'] += 1
                occupied_node_list.sort(key=lambda x: x.__getitem__('util'))
        
        print("node_g{} expand {} node_sets".format(node_group, acquired_node_num))

    

    def time_slicing_execute(self, cur_time, jobs, time_diff):
        node_release = False
        switch_job = int(cur_time % 60) == 0 # specify time, switch job
        used_gpus = 0
        
        for num_gpu, node_list in self.node_g.items():
            release_nodes = list() # release nodes
            for node_set in node_list:
                concurrency = 0
                total_util = 0
                for r_job in node_set['jobs']:
                    total_util = total_util + r_job['model']['mem_util']
                    if total_util > node_set['capacity']:
                        break
                    concurrency += 1
                tmp_used_gpus = \
                    num_gpu if (len(node_set['jobs']) * num_gpu)  > node_set['nodes'][0].check_total_gpus() else (len(node_set['jobs']) * num_gpu) # TODO: figure out
                
                used_gpus += tmp_used_gpus

                i = 0
                end_job_list = list()
                for r_job in node_set['jobs']:
                    r_job['executed_time'] = r_job['executed_time'] + time_diff
                    if r_job['executed_time'] >= r_job['duration']:
                        r_job['end_time'] = cur_time + r_job['duration'] - r_job['executed_time']
                        r_job['status'] = 'END'
                        end_job_list.append(r_job)
                        print("job[%d] ends at time[%d]" %(r_job['job_id'], r_job['end_time']))
                    i += 1
                    if i >= concurrency:
                        break
                
                if switch_job and len(node_set['jobs']) > concurrency:
                    # node_set['jobs'].reverse()
                    random.shuffle(node_set['jobs'])

                for end_job in end_job_list:
                    jobs.running_jobs.remove(end_job)
                    node_set['jobs'].remove(end_job)
                    node_set['num_jobs'] = node_set['num_jobs'] - 1

                
                if len(node_set['jobs']) == 0:
                    assert node_set['num_jobs'] == 0
                    for node in node_set['nodes']:
                        self.free_nodes.append(node)
                    release_nodes.append(node_set)
                    used_gpus = used_gpus - tmp_used_gpus
                    node_release = True
                
            for release_node in release_nodes:
                node_list.remove(release_node)
            
        return node_release



CLUSTER = _Cluster()


_allowed_symbols = [
    'CLUSTER'
]