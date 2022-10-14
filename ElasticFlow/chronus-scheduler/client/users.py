from abc import ABCMeta, abstractmethod
import os, sys
import math


class UserManager(object):
    def __init__(self, **kwargs):
        self.user_list = list()
    
    def __getitem__(self, idx):
        return self.user_list[idx]

    def __add__(self, user_list):
        self.user_list += user_list
    
    def add_user(self, user):
        self.user_list.append(user)
    
    def remove(self, user):
        self.user_list.remove(user)

    def index_user(self, name):
        for user in self.user_list:
            if user.name == name:
                return user
        assert '{} not found'.format(name)
    
    def __len__(self, ):
        return len(self.user_list)
    

class BaseUser(metaclass=ABCMeta):
    def __init__(self, JOBS, CLUSTER, name, logger, **kwargs):
        self.job_manager = JOBS
        self.cluster_manager = CLUSTER
        self.name = name
        self.logger = logger
        self.with_job_list = list()
        self.pending_jobs = list()
        self.running_jobs = list()
    
    @abstractmethod
    def submit_job(self, job):
        raise NotImplementedError
    
    
    @abstractmethod
    def finish_job(self, job):
        raise NotImplementedError
    


class VanillaUser(BaseUser):
    def __init__(self, JOBS, CLUSTER, name, logger, **kwargs):
        super(VanillaUser, self).__init__(JOBS=JOBS, CLUSTER=CLUSTER, name=name, logger=logger)
    
    def submit_job(self, job):
        self.with_job_list.append(job)
        self.job_manager.submit_job(job)
    

    def finish_job(self, job):
        self.with_job_list.remove(job)



class TimeAwareUser(BaseUser):
    def __init__(self,  JOBS, CLUSTER, name, logger, **kwargs):
        super(TimeAwareUser, self).__init__(JOBS=JOBS, CLUSTER=CLUSTER, name=name, logger=logger)
        self.quota = kwargs.get('quota')
        assert self.quota is not None, 'Specify `quota` value'
        self.before_deadline_weight = 0.5
        self.normal_weight = 1.
        self.miss_deadline_weight = 0.5



    def update_after_job(self, cost):
        self.quota -= cost


    def init_quota(self, quota):
        self.quota = quota


    def submit_job(self, job):
        if self.quota > 0:
            self.with_job_list.append(job)
            self.job_manager.submit_job(job)
        else:
            self.logger.info('Job [{}] failed to submit, because user [{}] quota [{}] less than zero'.format(job['job_idx'], self.name, self.quota))
    

    def calculate_cost(self, job):
        if job['usr_mode'] == 'deadline':
            buffer_time = job['expect_time'] - job['total_executation_time']
            if buffer_time > 0:
                return math.exp(-self.before_deadline_weight * buffer_time)
            else:
                return self.miss_deadline_weight

        elif job['usr_mode'] == 'normal':
            return self.normal_weight

        else:
            raise NotImplementedError
        
        
    def finish_job(self, job):
        self.with_job_list.remove(job)
        cost = self.calculate_cost(job)
        self.update_after_job(cost)
    
USERS = UserManager()


_allowed_symbols = [
    'USERS'
]


