import logging
import time
import signal
import socket
import subprocess
import os
import threading
import argparse
import random
from runtime.rpc import worker_client
from runtime.rpc import worker_server
from job import NodeJob
from utils import get_host_ip

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class TrainerInfo():
    """information of the trainer"""
    def __init__(self, rank, trainer_id, trainer_addr, trainer_port, process_group_master_ip, process_group_master_port, world_size, 
        run_dir, dynamic_requests=False, load_checkpoint=False, scheduler_addr=None, scheduler_port=None):
        self._rank = rank
        self._trainer_id = trainer_id
        self._trainer_addr = trainer_addr
        self._trainer_port = trainer_port
        self._process_group_master_ip = process_group_master_ip
        self._process_group_master_port = process_group_master_port
        self._world_size = world_size
        self._handle = subprocess.Popen([NodeJob.get_python_path(), 
                                  "trainer.py",
                                  f"--rank={self._rank}",
                                  f"--id={self._trainer_id}",
                                  f"--addr={self._trainer_addr}",
                                  f"--port={self._trainer_port}",
                                  f"--pg_addr={self._process_group_master_ip}",
                                  f"--pg_port={self._process_group_master_port}",
                                  f"--world_size={self._world_size}",
                                  f"--dynamic_requests={dynamic_requests}",
                                  f"--load_checkpoint={load_checkpoint}",
                                  f"--scheduler_addr={scheduler_addr}",
                                  f"--scheduler_port={scheduler_port}"],
                                  cwd=run_dir, shell=False)
   
    @property
    def location(self):
        return '%s:%d' % (self._trainer_addr, self._trainer_port)
    
    @property
    def handle(self):
        return self._handle

class Worker:
    def __init__(self, master_addr, master_port, worker_addr, worker_port, num_gpus, pg_master_addr, pg_master_port, pg_world_size,
        run_dir='../elastic-training-executor/', dynamic_requests=False, scheduler_addr=None, scheduler_port=None):
        # contains all the jobs that are ready to run or running, each element is a key-value pair -> job_id : NodeJob
        self._jobs = {} 
        # contains all the jobs that wait for killing
        self._wait_for_kill_jobs = {}

        self._dynamic_requests = dynamic_requests
        # the path to workloads, also where the trainer.py is located
        self._run_dir = run_dir
        # number of gpu on this node
        self._num_gpus = num_gpus
        # ip address of this worker (i.e. the host ip)
        self._worker_addr = worker_addr
        # port number of the worker rpc server
        self._worker_port = worker_port
        # process group master address
        self._pg_master_addr = pg_master_addr
        # process group master port
        self._pg_master_port = pg_master_port
        # process group world size
        self._pg_world_size = pg_world_size
        # whether this worker is killed, useful to end the worker gracefully
        self._killed = False
        # the process handle of trainers, each trainer is responsible for one GPU
        self._trainers = []        
        # worker rpc client to send request to master/trainer
        self._worker_rpc_client = worker_client.WorkerRpcClient(
            self._worker_addr, self._worker_port, master_addr, master_port)
        # register worker to master
        self._worker_id = self.register()
        self.scheduler_addr = scheduler_addr
        self.scheduler_port = scheduler_port

        # logger
        logger = logging.getLogger('worker' + str(self._worker_id))
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                          style='{'))
        logger.addHandler(ch)
        self._logger = logger
 
        # initialize trainer for each GPU
        for i in range(self._num_gpus):
            trainer = TrainerInfo(self._num_gpus * self._worker_id + i,  i, self._worker_addr, self._worker_port + i + 1, pg_master_addr, 
                pg_master_port, pg_world_size, self._run_dir, self._dynamic_requests, False, scheduler_addr, scheduler_port)
            self._trainers.append(trainer)
        # wait for all the trainer to be initialized
        time.sleep(2)

        # initialize worker rpc server for master
        callbacks = {
            "RunJob" : self._run_job_callback,
            "UpdateJob" : self._update_job_callback,
            "NewGroup" : self._new_group_callback,
            "Broadcast" : self._broadcast_callback,
            "KillJob" : self._kill_job_callback,
            "ShutDown" : self._shut_down_callback,
            "BroadcastFinish" : self._broadcast_finish_callback,
            "TrainingBegin" : self._training_begin_callback,
            "RestartTrainers" : self._restart_trainers_callback,
        }
        self._server_thread = threading.Thread(
            target=worker_server.serve,
            args=(self._worker_id, worker_port, callbacks)
        )
        self._server_thread.setDaemon(True)
        self._server_thread.start()

        # wait for kill
        while True:
            if self._killed:
                time.sleep(5)
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(10)
    
    def register(self):
        """ register the worker to its master
        This is a interface around self._worker_rpc_client
        @return : the allocated worker id of this worker
        """
        return self._worker_rpc_client.register_worker(self._num_gpus)


    def _fetch_GPU_list(self, compressed_list):
        """ decode the gpu_list from the compressed integer
        @param compressed_list -> uint32 : i-th bit equals 1 means the 2^i-th gpu is used, and equals 0 otherwise
        @return decoded_list -> list of str : the gpu list that this job uses on this specific node (the gpu index is sorted increasingly)
        This is useful to construct the CUDA_VISIBLE_DEVICES string
        """
        decoded_list = []
        for i in range(32):
            if ((1 << i) & compressed_list) != 0:
                decoded_list.append(str(i))
        return decoded_list

    def _fetch_GPU_list_to_int(self, compressed_list):
        """ decode the gpu_list from the compressed integer
        @param compressed_list -> uint32 : i-th bit equals 1 means the 2^i-th gpu is used, and equals 0 otherwise
        @return decoded_list -> list of int : the gpu list that this job uses on this specific node (the gpu index is sorted increasingly)
        """
        decoded_list = []
        for i in range(32):
            if ((1 << i) & compressed_list) != 0:
                decoded_list.append(i)
        return decoded_list

    def _run_job_callback(self, job_name, batch_size, job_id, nproc_per_node, nnodes, node_rank, master_ip, master_port, compressed_list, iterations, ranks, from_scratch):
        #if job_id in self._jobs:
        #    self._wait_for_kill_jobs[job_id] = self._jobs[job_id]
        #    self._jobs.pop(job_id)
        #    self._logger.info(f"job {job_id} run new configuration elastically")
        # assert the job is new!
        if job_id in self._jobs:
            self._jobs.pop(job_id)
        gpu_list = self._fetch_GPU_list_to_int(compressed_list)
        node_job = NodeJob(job_name, batch_size, job_id, nproc_per_node, nnodes, node_rank, master_ip, master_port, 
            gpu_list, iterations, ranks, self._dynamic_requests, from_scratch)
        self._jobs[job_id] = node_job
        node_job.init_run(self._worker_rpc_client, self._trainers)
        return True
    
    def _update_job_callback(self, job_name, batch_size, job_id, nproc_per_node, nnodes, node_rank, master_ip, master_port, compressed_list, iterations, ranks):
        if job_id not in self._jobs:
            self._logger.error("the job to be updated must be run first !")
            return False
        node_job = self._jobs[job_id]
        gpu_list = self._fetch_GPU_list(compressed_list)
        node_job.update(job_name, nproc_per_node, nnodes, node_rank, master_ip, master_port, gpu_list, 
            iterations, ranks, self._run_dir)
        if node_job.job_handle is None:
            return False
        self._logger.info(f"updated job pid : {node_job.job_handle.pid}")
        return True
    
    def _new_group_callback(self, for_ddp, job_id, ranks, src):
        # ranks are the global rank in n*8 cluster
        for trainer in self._trainers:
            self._worker_rpc_client.new_group(for_ddp, job_id, ranks, src, trainer.location)
        return True

    def _broadcast_callback(self, job_id, local_rank, src):
        # src is the global rank in n*8 cluster
        # TODO: send this to trainer
        print(f"trainer on local_rank {local_rank} will broadcast the model of job {job_id} from source {src}")
        return True

    def _broadcast_finish_callback(self):
        for trainer in self._trainers:
            self._worker_rpc_client.broadcast_finish(trainer.location)
        return True

    def _training_begin_callback(self):
        for trainer in self._trainers:
            self._worker_rpc_client.training_begin(trainer.location)
        return True
        
    def _kill_job_callback(self, job_id):
        if job_id in self._wait_for_kill_jobs:
            # there is new elastic configuration for this job on this worker, kill the old one
            self._wait_for_kill_jobs[job_id].kill(self._worker_rpc_client, self._trainers)
            self._wait_for_kill_jobs.pop(job_id)
        elif job_id in self._jobs:
            # the job will run on other worker, so kill the ones on this worker
            self._jobs[job_id].kill(self._worker_rpc_client, self._trainers)
            self._jobs.pop(job_id)
        else:
            self._logger.info(f"job {job_id} has already been killed")
        return True

    def _wait_trainer_die(self):
        for trainer in self._trainers:
            while trainer.handle.poll() is None:
                time.sleep(2)
        time.sleep(2) # TODO: fix

    def _wait_trainer_start(self):
        check_idx = 0
        while check_idx < len(self._trainers):
            try:
                self._worker_rpc_client.probe_trainer(self._trainers[check_idx].location)
                check_idx += 1
            except:
                time.sleep(2)
            

    def _restart_trainers_callback(self):
        # kill all the trainers first
        for trainer in self._trainers:
            self._worker_rpc_client.kill_trainer(trainer.location)
        # wait until all the trainers are killed (the state of all the trainers have been checkpointed)
        self._wait_trainer_die()
        # avoid address reuse
        self._pg_master_port += 100
        seed = random.randint(5, 100)
        # restart the trainers from checkpoint
        for i in range(self._num_gpus):
            trainer = TrainerInfo(self._num_gpus * self._worker_id + i,  i, self._worker_addr, self._worker_port + i + seed, self._pg_master_addr, 
                self._pg_master_port, self._pg_world_size, self._run_dir, self._dynamic_requests, True, self.scheduler_addr, self.scheduler_port)
            self._trainers[i] = trainer
        # make sure all the trainers are restarted
        self._wait_trainer_start()
        return True
       

    def _shut_down_callback(self):
        for trainer in self._trainers:
            self._worker_rpc_client.kill_trainer(trainer.location)
        self._killed = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a worker process")
    parser.add_argument('-i', '--ip_addr', type=str, required=True,
                        help='IP address for master server')
    parser.add_argument('-P', '--master_port', type=int, default=6888,
                        help='Port number for master server')
    parser.add_argument('-p', '--worker_port', type=int, default=6889,
                        help='Port number for worker server')
    parser.add_argument('-n', '--num_gpus', type=int, required=True,
                        help='Number of available GPUs')
    parser.add_argument('-A', '--pg_addr', type=str, required=True,
                        help='IP address for process group master')
    parser.add_argument('-g', '--pg_port', type=int, required=True,
                        help='Port number for process group master')
    parser.add_argument('-w', '--world_size', type=int, required=True,
                        help='World size of the process group')
    parser.add_argument('-r', '--run_dir', type=str, required=True,
                        help='Directory to run jobs from')
    parser.add_argument('-d', '--dynamic_requests', type=bool, default=False,
                        help='Accept scheduling requests from scheduler')
    parser.add_argument('--scheduler_addr', type=str, default='127.0.0.1',
                        help='scheduler server ip address')
    parser.add_argument('--scheduler_port', type=int, default=6889,
                        help='scheduler server port')
    parser.add_argument('-x', '--python', type=str, required=True,
                        help='Python interpreter used to run python scripts. You may use your own anaconda env.')

    args = parser.parse_args()
    opt_dict = vars(args)
    NodeJob.set_python_path(opt_dict['python'])
    worker_addr = get_host_ip()
    worker = Worker(opt_dict['ip_addr'], 
                    opt_dict['master_port'], 
                    worker_addr, 
                    opt_dict['worker_port'], 
                    opt_dict['num_gpus'], 
                    opt_dict['pg_addr'],
                    opt_dict['pg_port'],
                    opt_dict['world_size'],
                    opt_dict['run_dir'], 
                    opt_dict['dynamic_requests'],
                    opt_dict['scheduler_addr'],
                    opt_dict['scheduler_port'])
    
