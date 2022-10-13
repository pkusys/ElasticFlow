import grpc
import logging
import os
import sys
import time
import subprocess
from concurrent import futures

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import worker_to_master_pb2 as w2m_pb2
import worker_to_master_pb2_grpc as w2m_pb2_grpc
import worker_to_trainer_pb2 as w2t_pb2
import worker_to_trainer_pb2_grpc as w2t_pb2_grpc
import common_pb2

#LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
LOG_FORMAT = '{name}:{levelname} [{asctime}.{msecs}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class WorkerRpcClient:
    """Worker client for sending RPC requests to the master or trainer"""
    def __init__(self, worker_ip, worker_port,
                master_ip, master_port):
        logger = logging.getLogger("worker_client")
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                            style='{'))
        logger.addHandler(ch)
        self._logger = logger
        self._worker_ip = worker_ip
        self._worker_port = worker_port
        self._master_ip = master_ip
        self._master_port = master_port
        self._master_loc = f'{master_ip}:{master_port}'
    
    #============================================================
    #      rpc client for master                                #
    #============================================================
    def register_worker(self, num_gpus):
        request = w2m_pb2.RegisterWorkerRequest(
            ip_addr=self._worker_ip,
            port=self._worker_port,
            num_gpus=num_gpus
        )
        with grpc.insecure_channel(self._master_loc) as channel:
            self._logger.debug('Trying to register worker...')
            stub = w2m_pb2_grpc.WorkerToMasterStub(channel)
            response = stub.RegisterWorker(request)
            if response.success:
                self._worker_id = response.worker_id
                self._logger.info(
                    f"Client : Successfully registered worker with id {self._worker_id}"
                )
                return self._worker_id
            else:
                self._logger.error("Failed to register worker!")
                return None
    
    def done(self, job_id):
        self._logger.debug(f"worker {self._worker_id} informs master that job {job_id} is done")
        request = w2m_pb2.DoneRequest(worker_id=self._worker_id, job_id=job_id)
        with grpc.insecure_channel(self._master_loc) as channel:
            stub = w2m_pb2_grpc.WorkerToMasterStub(channel)
            stub.Done(request)
        

    #================================================================
    #      rpc client for trainer                                   #
    #================================================================
    def new_group(self, for_ddp, job_id, ranks, src, trainer_loc):
        with grpc.insecure_channel(trainer_loc) as channel:
            stub = w2t_pb2_grpc.WorkerToTrainerStub(channel)
            self._logger.info(f"Trying to init a new process group for ranks {ranks}")
            request = w2t_pb2.InitNewGroupRequest()
            request.for_ddp = for_ddp
            request.job_id = job_id
            request.ranks.extend(list(ranks))
            request.src = src
            response = stub.NewGroup(request)

    def broadcast_finish(self, trainer_loc):
        with grpc.insecure_channel(trainer_loc) as channel:
            stub = w2t_pb2_grpc.WorkerToTrainerStub(channel)
            request = common_pb2.Empty()
            response = stub.BroadcastFinish(request)

    def training_begin(self, trainer_loc):
        with grpc.insecure_channel(trainer_loc) as channel:
            stub = w2t_pb2_grpc.WorkerToTrainerStub(channel)
            request = common_pb2.Empty()
            response = stub.TrainingBegin(request)
        
    def init_standby(self, gpu_job, trainer_loc):
        with grpc.insecure_channel(trainer_loc) as channel:
            stub = w2t_pb2_grpc.WorkerToTrainerStub(channel)
            self._logger.info(f"Trying to init {gpu_job._job_id}:{gpu_job._job_name} on {trainer_loc}")
            request = gpu_job.toInitStandbyRequest()
            response = stub.InitStandby(request)
    
    def kill_active(self, job_id, trainer_loc):
        with grpc.insecure_channel(trainer_loc) as channel:
            stub = w2t_pb2_grpc.WorkerToTrainerStub(channel)
            self._logger.info(f"Trying to kill active workload {job_id} on {trainer_loc}")
            request = w2t_pb2.KillActiveRequest()
            request.job_id = job_id
            response = stub.KillActive(request)


    def kill_trainer(self, trainer_loc):
        with grpc.insecure_channel(trainer_loc) as channel:
            stub = w2t_pb2_grpc.WorkerToTrainerStub(channel)
            self._logger.info(f"Trying to kill trainer on {trainer_loc}")
            request = w2t_pb2.KillTrainerRequest()
            response = stub.KillTrainer(request)
    
    def probe_trainer(self, trainer_loc):
        with grpc.insecure_channel(trainer_loc) as channel:
            stub = w2t_pb2_grpc.WorkerToTrainerStub(channel)
            self._logger.info(f"Trying to probe trainer on {trainer_loc}")
            request = common_pb2.Empty()
            response = stub.ProbeTrainer(request)
            

 

        