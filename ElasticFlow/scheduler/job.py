import os
import signal
import sys
import subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), './runtime/rpc_stubs'))

import master_to_worker_pb2 as m2w_pb2
import worker_to_trainer_pb2 as w2t_pb2

class ClusterJob:
    """ a data-parallel job runs on multiple nodes
    _job_name : The name of the workload
    _job_id : Each job is identified by a unique job_id
    _jobs : a list of NodeJob where the job_name and job_id must be equal to the ClusterJob
    """
    
    def __init__(self, job_name, job_id, jobs):
        self._job_name = job_name
        self._job_id = job_id
        self._jobs = jobs
        for job in jobs:
            assert job.job_name == self._job_name
            assert job.job_id == self._job_id

    def __str__(self):
        s = f"job_id : {self._id}\n"
        for node in self._nodes:
            s += f"GPUs {self._devices[node]} on node {node}\n"
        return s

class NodeJob:
    """ 
    (maybe part of) a data-parallel job runs on one node
    it is managed by one worker
    """
    __PYTHON_PATH = "python3" # default system path, if you use conda, you should set this mannually

    def __init__(self, job_name, batch_size, job_id, nproc_per_node, nnodes, node_rank, master_addr, 
        master_port, gpu_list, iterations, ranks, fast_forward=False, from_scratch=False):
        self._job_name = job_name
        self._global_batch_size = batch_size
        self._job_id = job_id
        self._nproc_per_node = nproc_per_node
        self._nnodes = nnodes
        self._node_rank = node_rank
        self._master_addr = master_addr
        self._master_port = master_port
        self._gpu_list = gpu_list
        self._iterations = iterations
        self._ranks = ranks
        self._fast_forward = fast_forward
        self._from_scratch = from_scratch
        self._job_handle = None
    
    @classmethod
    def set_python_path(cls, path):
        cls.__PYTHON_PATH = path
    
    @classmethod
    def get_python_path(cls):
        return cls.__PYTHON_PATH
    
    @property
    def job_name(self):
        return self._job_name

    @property
    def batch_size(self):
        return self._global_batch_size
    
    @property
    def job_id(self):
        return self._job_id
    
    @property
    def nproc_per_node(self):
        return self._nproc_per_node
    
    @property
    def nnodes(self):
        return self._nnodes
    
    @property
    def node_rank(self):
        return self._node_rank
    
    @property
    def master_addr(self):
        return self._master_addr

    @property
    def master_port(self):
        return self._master_port

    @property
    def gpu_list(self):
        return self._gpu_list

    @property
    def job_handle(self):
        return self._job_handle
    
    @property
    def global_batch_size(self):
        return self._global_batch_size

    @property
    def iterations(self):
        return self._iterations

    #===========================================================
    #             utility functions for master client
    #    convert a node job into the grpc data type accordingly
    #===========================================================
    def toRunJobRequest(self):
        request = m2w_pb2.RunJobRequest()
        request.job_info.job_name = self._job_name
        request.job_info.batch_size = int(self._global_batch_size)
        request.job_info.job_id = int(self._job_id)
        request.job_info.nproc_per_node = int(self._nproc_per_node)
        request.job_info.nnodes = int(self._nnodes)
        request.job_info.node_rank = int(self._node_rank)
        request.job_info.master_ip = self._master_addr
        request.job_info.master_port = int(self._master_port)
        request.job_info.gpu_list = int(self._gpu_list)
        request.job_info.iterations = int(self._iterations)
        request.job_info.ranks.extend(list(self._ranks))
        request.job_info.from_scratch = self._from_scratch
        return request
    
    def toUpdateJobRequest(self):
        request = m2w_pb2.UpdateJobRequest()
        request.job_info.job_name = self._job_name
        request.job_info.job_id = int(self._job_id)
        request.job_info.nproc_per_node = int(self._nproc_per_node)
        request.job_info.nnodes = int(self._nnodes)
        request.job_info.node_rank = int(self._node_rank)
        request.job_info.master_ip = self._master_addr
        request.job_info.master_port = int(self._master_port)
        request.job_info.gpu_list = int(self._gpu_list)
        request.job_info.iterations = int(self._iterations)
        request.job_info.ranks.extend(list(self._ranks))
        request.job_info.from_scratch = self._from_scratch
        return request
    
    def init_run(self, rpc_client, trainers):
        """
        use rpc_client to send InitStandbyRequest to all the corresponding trainers
        """
        for local_rank, (trainer_id, rank) in enumerate(zip(self._gpu_list, self._ranks)):
            gpu_job = GPUJob(self._job_id, self._job_name, self._master_addr, self._master_port, self._node_rank * self._nproc_per_node + rank, 
            local_rank, self._nproc_per_node * self._nnodes, self._global_batch_size, self._iterations, self._from_scratch
            )
            rpc_client.init_standby(gpu_job, trainers[trainer_id].location)
    
    def kill(self, rpc_client, trainers):
        """
        use rpc_client to send KillActiveRequest to all the corresponding trainers
        """
        for trainer_id in self._gpu_list:
            rpc_client.kill_active(self._job_id, trainers[trainer_id].location)
            break


    #========================================================================
    #         !!!!  warning : deprecated  !!!!                              #
    #         useful when using torch.distributed.launch to run ddp jobs    #
    #========================================================================
    def run(self, run_dir):
        p = None
        # if self._job_name == "fake":
        #     p = subprocess.Popen([NodeJob.get_python_path(), "fake.py"], cwd=run_dir)
        #     self._logger.debug(f"pid : {p.pid}")
        gpu_list_str = ",".join(self._gpu_list)
        if self._job_name == "resnet_mn" or self._job_name == "resnet50":
            job_script = "resnet/mnmc_ddp_launch.py" 
        elif self._job_name == "vgg" or self._job_name == "vgg16":
            job_script = "vgg/vgg_ddp_launch.py"
        elif self._job_name == "inception_v3":
            job_script = "inceptionv3/inceptionv3_ddp_launch.py"
        elif self._job_name == "bert":
            job_script = "bert/bert_ddp_launch.py"
        elif self._job_name == "gpt2":
            job_script = "gpt2/gpt2_ddp_launch.py"
        elif self._job_name == "deepspeech2":
            job_script = "deepspeech2/dp2_ddp_launch.py"
        if self._fast_forward:
            p = subprocess.Popen([NodeJob.get_python_path(), "-m", "torch.distributed.launch", 
                f"--nproc_per_node={self._nproc_per_node}", 
                f"--nnodes={self._nnodes}", 
                f"--node_rank={self._node_rank}", 
                f"--master_addr={self._master_addr}",
                 f"--master_port={self._master_port}", 
                job_script,
                f"--suffix={self._job_id}",
                f"--batch_size={self._global_batch_size // len(self._gpu_list)}",
                f"--iterations={self._iterations}", "-f=True"],
                env={"CUDA_VISIBLE_DEVICES" : gpu_list_str}, cwd=run_dir, shell=False)
        else:
            p = subprocess.Popen([NodeJob.get_python_path(), "-m", "torch.distributed.launch", 
                f"--nproc_per_node={self._nproc_per_node}", 
                f"--nnodes={self._nnodes}", 
                f"--node_rank={self._node_rank}", 
                f"--master_addr={self._master_addr}",
                 f"--master_port={self._master_port}", 
                job_script,
                f"--suffix={self._job_id}",
                f"--batch_size={self._global_batch_size // len(self._gpu_list)}",
                 f"--iterations={self._iterations}"],
                env={"CUDA_VISIBLE_DEVICES" : gpu_list_str}, cwd=run_dir, shell=False)

        self._job_handle = p
    
    def update(self, job_name, nproc_per_node, nnodes, node_rank, master_addr, master_port, gpu_list, iterations, ranks, run_dir):
        assert self._job_handle is not None
        self._job_handle.terminate() # kill the old job, this will send SIGTERM to the subprocess
        self._job_name = job_name
        self._nproc_per_node = nproc_per_node
        self._nnodes = nnodes
        self._node_rank = node_rank
        self._master_addr = master_addr
        self._master_port = master_port
        self._gpu_list = gpu_list
        self._iterations = iterations
        self._ranks = ranks
        self.run(run_dir)

    
class GPUJob():
    """
    (maybe part of) a partial data parallel job which runs on one GPU
    it is managed by one trainer
    """
    def __init__(self, job_id, job_name, master_addr, master_port, rank, local_rank, world_size, batch_size, iterations, from_scratch):
        self._job_id = job_id
        self._job_name = job_name
        self._master_addr = master_addr
        self._master_port = master_port
        self._rank = rank
        self._local_rank = local_rank
        self._world_size = world_size
        self._batch_size = batch_size
        self._iterations = iterations
        self._from_scratch = from_scratch

    #===========================================================
    #             utility functions for worker client
    #    convert a GPU job into the grpc data type accordingly
    #===========================================================
    def toInitStandbyRequest(self):
        request = w2t_pb2.InitStandbyRequest()
        request.job_id = self._job_id
        request.job_name = self._job_name
        request.master_addr = self._master_addr
        request.master_port = self._master_port 
        request.rank = self._rank 
        request.local_rank = self._local_rank 
        request.world_size = self._world_size
        request.batch_size = self._batch_size
        request.iterations = self._iterations 
        request.from_scratch = self._from_scratch
        return request
 



if __name__ == "__main__":
    NodeJob.set_python_path("/home/anaconda/bin/python")
    NodeJob("resnet_mn", 0, 2, 2, 0, "127.0.0.1", 8888, 2)
    print(NodeJob.get_python_path())
