import grpc
import logging
import os
import sys
import time
import subprocess
from concurrent import futures

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import common_pb2
import master_to_worker_pb2 as m2w_pb2
import master_to_worker_pb2_grpc as m2w_pb2_grpc

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class WorkerRpcServer(m2w_pb2_grpc.MasterToWorkerServicer):
    """ each workerServer runs on one node to run/kill the jobs according to master's command
    """
    def __init__(self, logger, callbacks):
        self._logger = logger
        self._callbacks = callbacks
    
    def RunJob(self, request, context):
        self._logger.debug(f'Received run job {request.job_info.job_id} request from server')
        success = self._callbacks["RunJob"](request.job_info.job_name, 
                                    request.job_info.batch_size, request.job_info.job_id, 
                                    request.job_info.nproc_per_node, request.job_info.nnodes,
                                    request.job_info.node_rank,
                                    request.job_info.master_ip, request.job_info.master_port,
                                    request.job_info.gpu_list, request.job_info.iterations,
                                    request.job_info.ranks, request.job_info.from_scratch)
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to run the requested job")
        
        self._logger.info(f"Process the run job {request.job_info.job_id} request successfully")
        return common_pb2.Empty()
    
    def UpdateJob(self, request, context):
        self._logger.debug(f'Received update job {request.job_info.job_id} request from server')
        success = self._callbacks["UpdateJob"](request.job_info.job_name, 
                                    request.job_info.batch_size, request.job_info.job_id, 
                                    request.job_info.nproc_per_node, request.job_info.nnodes,
                                    request.job_info.node_rank,
                                    request.job_info.master_ip, request.job_info.master_port,
                                    request.job_info.gpu_list, request.job_info.iterations,
                                    request.job_info.ranks)
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to update the requested job")
        
        self._logger.info(f"Update the job {request.job_info.job_id} successfully")
        return common_pb2.Empty()
    
    def NewGroup(self, request, context):
        self._logger.debug('Received new group request from server')
        success = self._callbacks["NewGroup"](request.for_ddp, request.job_id, request.ranks, request.src)
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to build new group")
        return common_pb2.Empty()

    def Broadcast(self, request, context):
        self._logger.debug('Received broadcast request from server')
        success = self._callbacks["Broadcast"](request.job_id, request.local_rank, request.src)
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to broadcast")
        self._logger.info(f'Broadcast job {request.job_id} successfully')
        return common_pb2.Empty()
    
    def BroadcastFinish(self, request, context):
        self._logger.debug('Received broadcast finish request from server')
        success = self._callbacks["BroadcastFinish"]()
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to broadcast_finish")
        self._logger.info(f'Broadcast finish successfully')
        return common_pb2.Empty()

    def TrainingBegin(self, request, context):
        self._logger.debug('Received training begin request from server')
        success = self._callbacks["TrainingBegin"]()
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to run training_begin")
        self._logger.info(f'Training began successfully')
        return common_pb2.Empty()


    def KillJob(self, request, context):
        self._logger.debug(f'Received kill job {request.job_id} request from server')
        success = self._callbacks["KillJob"](request.job_id)
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to kill the job")
        self._logger.info(f'Kill the job {request.job_id} successfully')
        return common_pb2.Empty()

    def RestartTrainers(self, request, context):
        self._logger.debug(f"Restart the all the trainers")
        success = self._callbacks["RestartTrainers"]()
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to restart the trainers")
        self._logger.info(f"Restart the trainers successfully")
        return common_pb2.Empty()

    def ShutDown(self, request, context):
        self._logger.info(f'Shutting down the worker')
        self._callbacks["ShutDown"]()
        return common_pb2.Empty()

    
def serve(worker_id, port, callbacks):
    logger = logging.getLogger("worker_server" + str(worker_id))
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                      style='{'))
    logger.addHandler(ch)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    m2w_pb2_grpc.add_MasterToWorkerServicer_to_server(
        WorkerRpcServer(logger, callbacks), server
    )
    server.add_insecure_port('[::]:%d' % (port))
    server.start()
    logger.info(f'Starting worker server at port {port}')
    server.wait_for_termination()

if __name__ == '__main__':
    serve(6888)
