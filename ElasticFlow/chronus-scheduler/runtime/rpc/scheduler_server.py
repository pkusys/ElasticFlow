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

import trainer_to_scheduler_pb2 as t2s_pb2 
import trainer_to_scheduler_pb2_grpc as t2s_pb2_grpc
import common_pb2

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class MasterRpcServer(t2s_pb2_grpc.TrainerToSchedulerServicer):
    def __init__(self, logger, callbacks):
        self._logger = logger
        self._callbacks = callbacks

    def ReportStable(self, request, context):
        try:
            report_stable_callback = self._callbacks["ReportStable"]
            report_stable_callback(request.job_id)
            self._logger.info(
            	f"Scheduler : job {request.job_id} ready for fast forward")
            return common_pb2.Empty()
        except Exception as e:
            self._logger.error('Trainer: failed to request for fast forward, {0}'.format(e))
            #return t2s_pb2.ReportStableResponse(success=False, error_message=e)
            return common_pb2.Empty()

    def ReportReady(self, request, context):
        try:
            report_ready_callback = self._callbacks["ReportReady"]
            report_ready_callback(request.trainer_id)
            self._logger.info(
                f"Scheduler : trainer {request.trainer_id} ready for training")
            return common_pb2.Empty()
        except Exception as e:
            self._logger.error('Trainer: failed to request for report ready, {0}'.format(e))
            return common_pb2.Empty()



def serve(port, callbacks):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                      style='{'))
    logger.addHandler(ch)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    t2s_pb2_grpc.add_TrainerToSchedulerServicer_to_server(
        MasterRpcServer(logger, callbacks), server)
    server.add_insecure_port('[::]:%d' % (port))
    server.start()
    logger.info('Starting scheduler server at port {0}'.format(port))
    server.wait_for_termination()

if __name__ == '__main__':
    serve(6998)
