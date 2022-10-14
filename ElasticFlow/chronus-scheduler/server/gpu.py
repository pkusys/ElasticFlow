import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..')
import utils

class _GPU():
    def __init__(self, id):
        self.id = id
        self.occupied = False
        self.nv_link = True
        self.job_list = list()
    

    def add_info(self, belong_switch, belong_node):
        self.belong_switch = belong_switch
        self.belong_node = belong_node
        assert self.belong_switch is not None
        assert self.belong_node is not None
    
    def switch_id(self, ):
        return self.belong_switch.id 
    
    def node_id(self, ):
        return self.belong_node.id 
    
    def gpu_id(self, ):
        return self.id 
    
    def same_switch(self, gpu):
        return gpu.belong_switch.id == self.belong_switch.id 
    

    def same_node(self, gpu):
        return gpu.belong_node == self.belong_switch
    

    def with_nv_link(self, ):
        return self.nv_link
    

    def free_resource(self, ):
        return 1 if not self.occupied else 0
    

    def allocate_resource(self, job):
        self.job_list.append(job)
        self.occupied = True
    
    
    def release_source(self, job):
        self.job_list.remove(job)
        assert len(self.job_list) == 0
        if len(self.job_list) == 0:
            self.occupied = False

    


