import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..')


class NodeCluster(object):
    def __init__(self, switch_id, node_id, twin_node):
        self.switch_id = switch_id
        self.node_id = node_id
        self.twin_node = twin_node
        self.gpu_list = list()
    
    def check_free_gpus(self, ):
        return self.twin_node.check_free_gpus()

    def check_total_gpus(self, ):
        return self.twin_node.check_total_gpus()


    def add_gpu(self, gpu_instance):
        assert self.match(gpu_instance=gpu_instance) == True
        self.gpu_list.append(gpu_instance)


    def match(self, **kwargs): 
        if 'switch_id' in kwargs and 'node_id' in kwargs:
            return self.switch_id == kwargs.get('switch_id') and self.node_id == kwargs.get('node_id')
        
        elif 'gpu_instance' in kwargs:
            gpu_instance = kwargs.get('gpu_instance')
            assert hasattr(gpu_instance, 'belong_switch') and hasattr(gpu_instance, 'belong_node') 
            return self.switch_id == kwargs.get('gpu_instance').belong_switch.id  and \
                    self.node_id == kwargs.get('gpu_instance').belong_node.id
        else:
            raise NotImplementedError


class SwitchCluster(object):
    def __init__(self, switch_id, twin_switch):
        self.switch_id = switch_id
        self.node_list = list()
        self.twin_switch = twin_switch

    def check_free_gpus(self, ):
        return self.twin_switch.check_free_gpus()

    def check_total_gpus(self, ):
        return self.twin_switch.check_total_gpus()

    def add_node(self, node_instance):
        assert self.match(node_instance=node_instance)
        self.node_list.append(node_instance)
    

    def match(self, **kwargs):
        if 'switch_id' in kwargs:
            return self.switch_id == kwargs.get('switch_id')
        elif 'node_instance' in kwargs:
            node_instance = kwargs.get('node_instance')
            assert isinstance(node_instance, NodeCluster)
            return self.switch_id == kwargs.get('node_instance').switch_id
        else:
            raise NotImplementedError
    

class Topology(object):
    def __init__(self, job, **kwargs):
        self.job = job
        self.gpu_list = list()
        self.node_group = list()
        self.switch_group = list()
        
        if 'placements' in kwargs:
            self.init_topology_from_placement(placements=kwargs.get('placements'))
    

    def init_topology_from_placement(self, placements):
        num_gpu = 0
        for placement in placements:
            switch_id = placement['switch']

            for node_dict in placement['nodes']:
                assert node_dict['num_gpu'] == 0 or node_dict['num_gpu'] == 1
                # process gpu
                if node_dict['num_gpu'] == 1:
                    num_gpu += 1
                    node_instance = node_dict['node_instance']
                    for gpu_instance in node_instance.gpu_list:
                        if self.job in gpu_instance.job_list and gpu_instance not in self.gpu_list:
                            self.gpu_list.append(gpu_instance)
        
        assert len(self.gpu_list) == num_gpu
        self.init_node_group(gpu_list=self.gpu_list)
        self.init_switch_group(node_list=self.node_group)
    

    def init_node_group(self, gpu_list):
        for gpu_instance in self.gpu_list:
            found_node_cluster = False
            for node_cluster in self.node_group:
                if node_cluster.match(gpu_instance=gpu_instance):
                    node_cluster.add_gpu(gpu_instance)
                    found_node_cluster = True
                    break
            if not found_node_cluster:
                node_cluster = NodeCluster(switch_id=gpu_instance.belong_switch.id, node_id=gpu_instance.belong_node.id, twin_node=gpu_instance.belong_node)
                node_cluster.add_gpu(gpu_instance=gpu_instance)
                self.node_group.append(node_cluster)


    def init_switch_group(self, node_list):
        for node_instance in node_list:
            found_switch_cluster = False
            for switch_cluster in self.switch_group:
                if switch_cluster.match(node_instance=node_instance):
                    found_switch_cluster = True
                    switch_cluster.add_node(node_instance=node_instance)
                    break

            if not found_switch_cluster:
                switch_cluster = SwitchCluster(switch_id=node_instance.switch_id, twin_switch=None)
                self.switch_group.append(switch_cluster)
        
