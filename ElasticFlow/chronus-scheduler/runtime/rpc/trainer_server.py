import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import grpc
import common_pb2
import worker_to_trainer_pb2 as w2t_pb2
import worker_to_trainer_pb2_grpc as w2t_pb2_grpc
import logging
from concurrent import futures

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class TrainerRpcServer(w2t_pb2_grpc.WorkerToTrainerServicer):
    """ each TrainerServer receives messages from worker and processes accordingly
    """
    def __init__(self, logger, callbacks):
        self._logger = logger
        self._callbacks = callbacks
    
    def NewGroup(self, request, context):
        self._logger.info("Received InitNewGroup request from worker")
        success = self._callbacks["NewGroup"](
            request.for_ddp,
            request.job_id,
            request.ranks,
            request.src
        )
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to init the new process group")
        self._logger.info(f"process the init process group {request.ranks} successfully") 
        return common_pb2.Empty()
    
    def BroadcastFinish(self, request, context):
        self._logger.info("Received BroadcastFinish from worker")
        success = self._callbacks["BroadcastFinish"]()
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to broadcast finish")
        self._logger.info(f"process the broadcast finish request successfully") 
        return common_pb2.Empty()

    def TrainingBegin(self, request, context):
        self._logger.info("Received TrainingBegin from worker")
        success = self._callbacks["TrainingBegin"]()
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to training begin")
        self._logger.info(f"process the training begin request successfully") 
        return common_pb2.Empty()
    
    def InitStandby(self, request, context):
        self._logger.info("Received InitStandby request from worker")
        success = self._callbacks["InitStandby"](
           request.job_id, request.job_name, request.master_addr, request.master_port, request.rank, 
           request.local_rank, request.world_size, request.batch_size, request.iterations, request.from_scratch 
        )
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to init the requested job or invalid workload")
        
        self._logger.info(f"process the init job {request.job_id} successfully")
        return common_pb2.Empty()
    
    def KillActive(self, request, context):
        self._logger.debug("Received KillActive request from worker")
        success = self._callbacks["KillActive"](
            request.job_id 
        )
        if not success:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "failed to kill the active workload")
        self._logger.info(f"process the kill job {request.job_id} successfully")
        return common_pb2.Empty()
    
    def KillTrainer(self, request, context):
        self._logger.debug("Received KillTrainer request from worker")
        self._callbacks["KillTrainer"]()
        self._logger.info(f"process the kill trainer successfully")
        return common_pb2.Empty()
    
    def ProbeTrainer(self, request, context):
        self._logger.debug("Received ProbeTrainer request from worker")
        return common_pb2.Empty()
 
 
def serve(trainer_id, port, callbacks):
    logger = logging.getLogger(__name__ + str(trainer_id))
    logger.setLevel(logging.DEBUG)
    logger.propagate = False 
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                      style='{'))
    logger.addHandler(ch)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    w2t_pb2_grpc.add_WorkerToTrainerServicer_to_server(
        TrainerRpcServer(logger, callbacks), server
    )
    server.add_insecure_port('[::]:%d' % (port))
    server.start()
    logger.info(f"Starting trainer server at port {port}")
    server.wait_for_termination()

if __name__ == '__main__':
    # just for sanity test
    serve(7999, None)
