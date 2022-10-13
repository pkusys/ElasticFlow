import grpc
import os 
import sys
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import master_to_worker_pb2 as m2w_pb2
import master_to_worker_pb2_grpc as m2w_pb2_grpc
import common_pb2

#LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
LOG_FORMAT = '{name}:{levelname} [{asctime}.{msecs}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class MasterRpcClient:
    """Scheduler client for sending RPC requests to a worker server"""
    def __init__(self):
        logger = logging.getLogger("master_client")
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                            style='{'))
        logger.addHandler(ch)

        self._logger = logger
   
    def run_job(self, node_job, worker_loc):
        """ run the node job on the specific worker
        @param node_job : the job to be run
        @param worker_loc : the address of the worker server, a tuple of (ip, port)
        """
        with grpc.insecure_channel(worker_loc) as channel:
            stub = m2w_pb2_grpc.MasterToWorkerStub(channel)
            self._logger.info(f"Run {node_job.job_id}:{node_job.job_name} on {worker_loc}")
            request = node_job.toRunJobRequest()
            response = stub.RunJob(request)
    
    def update_job(self, node_job, worker_loc):
        with grpc.insecure_channel(worker_loc) as channel:
            stub = m2w_pb2_grpc.MasterToWorkerStub(channel)
            self._logger.info(f"Update {node_job.job_id}:{node_job.job_name} on {worker_loc}")
            request = node_job.toUpdateJobRequest()
            response = stub.UpdateJob(request)

    def broadcast(self, job_id, global_rank, src, worker_loc):
        with grpc.insecure_channel(worker_loc) as channel:
            stub = m2w_pb2_grpc.MasterToWorkerStub(channel)
            request = m2w_pb2.BroadcastRequest()
            request.job_id = job_id
            request.local_rank = global_rank
            request.src = src
            response = stub.Broadcast(request)

    def broadcast_finish(self, worker_loc):
        with grpc.insecure_channel(worker_loc) as channel:
            stub = m2w_pb2_grpc.MasterToWorkerStub(channel)
            request = common_pb2.Empty()
            response = stub.BroadcastFinish(request)

    def training_begin(self, worker_loc):
        with grpc.insecure_channel(worker_loc) as channel:
            stub = m2w_pb2_grpc.MasterToWorkerStub(channel)
            request = common_pb2.Empty()
            response = stub.TrainingBegin(request)

    def new_group(self, for_ddp, job_id, ranks, src, worker_loc):
        with grpc.insecure_channel(worker_loc) as channel:
            stub = m2w_pb2_grpc.MasterToWorkerStub(channel)
            request = m2w_pb2.NewGroupRequest()
            request.for_ddp = for_ddp
            request.job_id = job_id
            request.ranks.extend(list(ranks))
            if src is not None:
                request.src = src
            response = stub.NewGroup(request)
        
    def kill_job(self, job_id, worker_loc):
        self._logger.info(f"Kill job {job_id}")
        with grpc.insecure_channel(worker_loc) as channel:
            stub = m2w_pb2_grpc.MasterToWorkerStub(channel)
            request = m2w_pb2.KillJobRequest()
            request.job_id = job_id
            response = stub.KillJob(request)
    
    def shutdown(self, worker_loc):
        self._logger.info("Shutting down all the workers")
        with grpc.insecure_channel(worker_loc) as channel:
            stub = m2w_pb2_grpc.MasterToWorkerStub(channel)
            response = stub.ShutDown(common_pb2.Empty())
    
    def restart_trainers(self, worker_loc):
        self._logger.info("Restart all the trainers")
        with grpc.insecure_channel(worker_loc) as channel:
            stub = m2w_pb2_grpc.MasterToWorkerStub(channel)
            response = stub.RestartTrainers(common_pb2.Empty())
 