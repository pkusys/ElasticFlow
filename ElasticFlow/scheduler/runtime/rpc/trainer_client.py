import grpc
import logging
import os
import sys
import time
import subprocess
from concurrent import futures

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import trainer_to_scheduler_pb2 as t2s_pb2
import trainer_to_scheduler_pb2_grpc as t2s_pb2_grpc

#LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
LOG_FORMAT = '{name}:{levelname} [{asctime}.{msecs}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class TrainerRpcClient:
    """trainer client for sending RPC requests to the scheduler"""
    def __init__(self, scheduler_ip, scheduler_port):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                            style='{'))
        logger.addHandler(ch)
        self._logger = logger
        self.scheduler_ip = scheduler_ip
        self._scheduler_port = scheduler_port
        self._scheduler_loc = f'{scheduler_ip}:{scheduler_port}'

    def report_stable(self, job_id):
        self._logger.debug(f"Trainer of job {job_id} request for fast forward")
        request = t2s_pb2.ReportStableRequest()
        request.job_id = job_id
        with grpc.insecure_channel(self._scheduler_loc) as channel:
            stub = t2s_pb2_grpc.TrainerToSchedulerStub(channel)
            stub.ReportStable(request)

    def report_ready(self, trainer_id):
        print(f"Trainer {trainer_id} is ready for training")
        request = t2s_pb2.ReportReadyRequest()
        request.trainer_id = trainer_id
        with grpc.insecure_channel(self._scheduler_loc) as channel:
            stub = t2s_pb2_grpc.TrainerToSchedulerStub(channel)
            stub.ReportReady(request)
    
        
