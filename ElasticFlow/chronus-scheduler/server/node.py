
import utils
from .gpu import _GPU


class _Node(object):
    def __init__(self, id, num_gpu=0, num_cpu=0, mem=0):
        self.id = id
        self.num_cpu = num_cpu
        self.free_cpus = num_cpu
        self.num_gpu = num_gpu       
        self.free_gpus = num_gpu

        # in and out should be the same
        self.network_in = 0
        self.network_out = 0

        # memory
        self.mem = mem
        self.free_mem = mem

        # node class for gandiva
        self.job_gpu = 0
        self.num_jobs = 0
        self.gpu_list = list()
        # print('    Node[%d] has %d gpus, %d cpus, %d G memory' % (id, num_gpu, num_cpu, mem))

        # enable
        self.enable_resource = True
        # user
        self.permission_user_list = None


    # GPU
    def add_gpus(self, num_gpu, belong_switch):
        self.num_gpu = num_gpu
        self.belong_switch = belong_switch
        for gpu_id in range(num_gpu):
            gpu_instance = _GPU(gpu_id)
            gpu_instance.add_info(belong_switch=self.belong_switch, belong_node=self)
            self.gpu_list.append(gpu_instance)


    def check_free_gpus(self, user_name=None):
        if self.permission_user_list is not None and user_name is not None:
            if user_name not in self.permission_user_list: return 0
        return self.free_gpus if self.enable_resource else 0

    def check_free_guarantee_gpus(self, user_name=None):
        if self.enable_resource: return 0 
        if self.permission_user_list is not None and user_name is not None:
            if len(self.permission_user_list) == 1 and user_name == self.permission_user_list[0]:
                return self.free_gpus 
        return 0

    def check_free_spot_gpus(self, user_name=None):
        if not self.enable_resource: return 0
        return self.check_free_gpus(user_name=user_name) - self.check_free_guarantee_gpus(user_name=user_name)

    def check_total_gpus(self, user_name=None):
        if self.permission_user_list is not None and user_name is not None:
            if user_name not in self.permission_user_list: return 0
        return self.num_gpu if self.enable_resource else 0
    
    def check_total_guarantee_gpus(self, user_name=None):
        if self.enable_resource: return 0 
        if self.permission_user_list is not None and user_name is not None:
            if len(self.permission_user_list) == 1 and user_name == self.permission_user_list[0]:
                return self.num_gpu 
        return 0

    def check_total_spot_gpus(self, user_name=None):
        if self.enable_resource: return 0 
        return self.check_total_gpus(user_name=user_name) - self.check_total_guarantee_gpus(user_name=user_name)

    def check_total_cpus(self):
        return self.num_cpu if self.enable_resource else 0

    def alloc_gpus(self, num_gpu, job):
        if num_gpu == 0: return True
        if not self.enable_resource: return False

        if num_gpu > self.free_gpus:
            return False
        else:
            self.free_gpus -= num_gpu
            for gpu_instance in self.gpu_list:
                if gpu_instance.free_resource() > 0 and num_gpu > 0:
                    gpu_instance.allocate_resource(job)
                    num_gpu -= 1
            assert num_gpu == 0
            return True


    def release_gpus(self, num_gpu, job):
        if num_gpu == 0: return True
        assert self.free_gpus + num_gpu <= self.num_gpu
        for gpu_instance in self.gpu_list:
            if job in gpu_instance.job_list:
                gpu_instance.release_source(job)
                self.free_gpus += 1
                num_gpu -= 1
                if num_gpu == 0:
                    break
        assert num_gpu == 0
        return True


    # CPU
    def check_free_cpus(self):
        return self.free_cpus if self.enable_resource else 0

    def alloc_cpus(self, num_cpu=0):
        if num_cpu > self.free_cpus or not self.enable_resource:
            return False
        else:
            self.free_cpus -= num_cpu
            return True


    def release_cpus(self, num_cpu=0):
        if self.free_cpus + num_cpu > self.num_cpu:
            self.free_cpus = self.num_cpu
            return False
        else:
            self.free_cpus += num_cpu
            return True 


    # network
    def add_network_load(self, in_load=0, out_load=0):
        self.network_in += in_load
        self.network_out += out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)


    def release_network_load(self, in_load=0, out_load=0):
        self.network_in -= in_load
        self.network_out -= out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)

    def set_network_load(self, in_load=0, out_load=0):
        self.network_in = in_load
        self.network_out = out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)


    # release / allocation
    def alloc_job_resource(self, num_gpu, num_cpu, job):
        # alloc job resource
        gpu = self.alloc_gpus(num_gpu, job)
        cpu = self.alloc_cpus(num_cpu)

        if cpu == False or gpu == False:
            self.release_gpus(num_gpu, job)
            self.release_cpus(num_cpu)
            return False
        
        return True 

    def release_job_resource(self, node_dict, job):
        cpu = self.release_cpus(node_dict['num_cpu'])
        gpu = self.release_gpus(node_dict['num_gpu'], job=job)
        self.free_mem = self.free_mem + node_dict['mem']
        return (cpu and gpu)


    def release_job_gpu_cpu(self, num_gpu, num_cpu, job):
        cpu = self.release_cpus(num_cpu)
        gpu = self.release_gpus(num_gpu, job)
        return (cpu and gpu)
