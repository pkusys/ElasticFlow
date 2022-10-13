from concurrent import futures
import time

import grpc
import logging
import os
import sys
import socket
import traceback
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import worker_to_master_pb2 as w2m_pb2 
import worker_to_master_pb2_grpc as w2m_pb2_grpc
import scheduler_to_master_pb2 as s2m_pb2 
import scheduler_to_master_pb2_grpc as s2m_pb2_grpc
import common_pb2

#LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
LOG_FORMAT = '{name}:{levelname} [{asctime}.{msecs}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class MasterRpcServer(w2m_pb2_grpc.WorkerToMasterServicer):
    def __init__(self, logger, callbacks):
        self._logger = logger
        self._callbacks = callbacks
    
    def RegisterWorker(self, request, context):
        try:
            register_callback = self._callbacks["RegisterWorker"]
            worker_id = register_callback(request.ip_addr, request.port, request.num_gpus)
            self._logger.info(
                f"Master : Successfully registered worker with id {worker_id} on {request.ip_addr}:{request.port} with {request.num_gpus} GPUs")
            response = w2m_pb2.RegisterWorkerResponse(success=True, worker_id=worker_id)
            return response

        except Exception as e:
            self._logger.error('Could not register worker: {0}'.format(e))
            return w2m_pb2.RegisterWorkerResponse(success=False, error_message=e)
    
    def Done(self, request, context):
        self._callbacks["Done"](request.job_id, request.worker_id)
        return common_pb2.Empty()


class MasterSchedulerRpcServer(s2m_pb2_grpc.SchedulerToMasterServicer):
    def __init__(self, logger, callbacks):
        self._logger = logger
        self._callbacks = callbacks
        
    def Schedule(self, request, context):
        try:
            self._callbacks["Schedule"](request.command)
            return s2m_pb2.ScheduleResponse(success=True)
        except Exception as e:
            return s2m_pb2.ScheduleResponse(success=False, error_message=e)


def serve(port, callbacks):
    logger = logging.getLogger("master_server")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                      style='{'))
    logger.addHandler(ch)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    w2m_pb2_grpc.add_WorkerToMasterServicer_to_server(
        MasterRpcServer(logger, callbacks), server)
    s2m_pb2_grpc.add_SchedulerToMasterServicer_to_server(
        MasterSchedulerRpcServer(logger, callbacks), server)
    server.add_insecure_port('[::]:%d' % (port))
    server.start()
    logger.info('Starting master server at port {0}'.format(port))
    server.wait_for_termination()

if __name__ == '__main__':
    serve(6999)
