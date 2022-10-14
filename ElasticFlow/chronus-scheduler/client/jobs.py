import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..')
from client import models
from utils import util
from utils.util import allocate_set
from .time_estimator import TimeEstimator


class Job(dict):
    def __init__(self, info_dict):
        super(Job, self).__init__()
        # first initilize basic information
        self.init()
        self.reassign(info_dict)
        self.update(info_dict)
        self.max_reward = 1. * self.required_gpu_num
        self.min_reward = 0.
    
    @property
    def required_gpu_num(self, ):
        return self.__getitem__('num_gpu')
    
    @property
    def uesr_specified_deadline_time(self, ):
        if self.__getitem__('expect_time') is not None:
            return self.__getitem__('expect_time')
    
    @uesr_specified_deadline_time.setter
    def uesr_specified_deadline_time(self, value):
        if self.__getitem__('expect_time') is not None:
            self.__setitem__('expect_time', value)
    
    @property
    def estimate_completion_time(self, ):
        return self.__getitem__('duration')
    
    @property
    def submit_time(self, ):
        return self.__getitem__('submit_time')


    def estimate_reward(self, profile_info):
        self.time_estimator = TimeEstimator(self, profile_info)
        self.policy_list = allocate_set(self.__getitem__('num_gpu'))
        # self.reward_list = [self.optimistic_placement_reward(policy) for policy in self.policy_list]
        self.reward_list = list()
        for policy in self.policy_list:
            for info in profile_info:
                if info[0] == policy:
                    self.reward_list.append(info[1] * self.required_gpu_num)
                    break
        assert len(self.reward_list) == len(self.policy_list), 'length should be the same'


        self.max_reward = max(self.reward_list)
        self.min_reward = min(self.reward_list)
        self.max_min_diff = self.max_reward - self.min_reward
    
    
    def optimistic_placement_reward(self, policy):
        if isinstance(policy[0], dict):
            policy = sorted([item['required_gpu_num'] for item in policy], reverse=True)
        if policy in self.policy_list:
            return self.reward_list[self.policy_list.index(policy)]
        else:
            for idx in range(len(self.policy_list)):
                if len(self.policy_list) == len(policy):
                    return self.reward_list[idx]
            return min(self.reward_list) 

        return self.time_estimator.optimistic_estimate(policy)
    

    def pessimistic_placement_reward(self, policy):
        raise NotImplementedError
        return self.time_estimator.pessimistic_estimate(policy)


    def reassign(self, info_dict):
        for key, value in info_dict.items():
            if value is not None:
                try:
                    if value.isdigit():
                        info_dict[key] = int(float(value))
                except :
                    continue
        info_dict['model_scale'] = info_dict.get('model_scale', 1)
        self.get_job_model(info_dict)
        self.get_network_load(info_dict)
       


    def get_job_model(self, info_dict):
        if ('model_name' in info_dict) and ('model_scale' in info_dict):
            info_dict['model'] = models.get_model_with_scale(info_dict['model_name'], info_dict['model_scale'])
        else:
            info_dict['model'] = models.get_model_with_scale("alexnet", 1)

    
    def get_network_load(self, job_dict):
        # TODO: refactor
        if 'num_gpu' not in job_dict or 'model' not in job_dict:
            print('No gpu/model information')
            return
        
        num_w = job_dict['num_gpu']
        num_ps = num_w


        if num_w == 1:
            job_dict['ps_network'] = list()
            job_dict['w_network'] = list([0])
            job_dict['ps_ave'] = 0
            return


        job_dict['w_network'] = list([job_dict['model']['total_size']] * num_w)
        job_dict['ps_network'] = list([0] * num_ps)
        for i in range(0, len(job_dict['model']['tensors'])):
            ps_idx = int(i % num_ps)
            # job_dict['ps_network'][ps_idx] += (job_dict['model']['tensors'][i] * num_w)
            job_dict['ps_network'][ps_idx] += (job_dict['model']['tensors'][i])

        for i in range(0, len(job_dict['ps_network'])):
            job_dict['ps_network'][i] = round(job_dict['ps_network'][i], 1)



    def init(self, ):
        init_info = {
            'start_time' : sys.maxsize, 
            'end_time' : 0, 
            'pending_time' : 0, 
            'progress' : 0, 
            'total_executed_time': 0, 
            'last_start_time': 0, 
            'last_check_time': 0, 
            'executed_time': 0, 
            'preempt': 0, 
            'resume': 0, 
            'promote': 0, 
            'status': 'ADDED', 
            'gpus': list(), 
            'placements' : list(), 
            'topology' : None, 
            'expect_time': sys.maxsize,
            'usr_mode': 'deadline', 
            'user': None,
        }
        self.update(init_info)




class JobManager(object):
    def __init__(self, ):
        self.num_job = 0        
        self.job_list = list()
        self.job_events = list()        
        self.pending_jobs = list() # [{job_dict}, {job_dict}]
        self.runnable_jobs = list() # pending + running
        self.running_jobs = list() # running


    def sort_all_jobs(self, key):
        self.job_list.sort(key = lambda e:e.__getitem__(key))


    def submit_job(self, job):
        #  Add job (job_dict) into job_list
        
        self.job_list.append(job)
        self.num_job += 1


    def create_multi_nodes_placement(self, job, switch_id, node_list):
        placement_dict = {
            'switch': switch_id, 
            'nodes' : node_list
        }
        job['placements'].append(placement_dict)



    def create_single_node_placement(self, job, switch_id, node_id, num_gpu, num_cpu, mem=0):
        '''
        under this switch, there is only one need used
        {'switch': xx, 'nodes': [{'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'network': xxxx, 'tasks': [w0, w1, ps1]}]}
        '''

        node_dict = {
            'id': node_id, 
            'num_gpu': num_gpu, 
            'num_cpu': num_cpu, 
            'mem': mem, 
            'tasks': list(), 
            'network': 0, 
        }
        placement_dict = {
            'switch': switch_id, 
            'nodes' : node_dict
        }
        job['placements'].append(placement_dict)
        return node_dict['network']


    def print_all_job_events(self):
        print('    print all job events ')
        for event in self.job_events:
            print('      event.time[%d], with %d start_jobs, and %d end_jobs' % 
                            (event['time'], len(event['start_jobs']), len(event['end_jobs'])))
        print(' ')


    def prepare_job_start_events(self):
        '''
        add job start events into job_events list
        end events should be added when they are starting
        '''
        for job in self.job_list:
            submit_time = job['submit_time']
            job['status'] = 'EVENT' 
            job_dict = util.search_dict_list(self.job_events, 'time', submit_time)
            if job_dict is None:
                job_dict = {
                    'time' : submit_time, 
                    'start_jobs': [job], 
                    'end_jobs' : list()
                }
                self.job_events.append(job_dict)
            else:
                job_dict['start_jobs'].append(job)
            

        self.job_events.sort(key = lambda e:e.__getitem__('time'))
        #print('Init, add job start events')
        #self.print_all_job_events()


JOBS = JobManager()


_allowed_symbols = [
    'JOBS'
]
