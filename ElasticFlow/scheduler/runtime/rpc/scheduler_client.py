import grpc
import logging
import os
import sys
import time
import subprocess
from concurrent import futures

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import scheduler_to_master_pb2 as s2m_pb2
import scheduler_to_master_pb2_grpc as s2m_pb2_grpc

#LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
LOG_FORMAT = '{name}:{levelname} [{asctime}.{msecs}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class SchedulerRpcClient:
    """Scheduler client for sending RPC requests to the master"""
    def __init__(self, master_ip, master_port):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                            style='{'))
        logger.addHandler(ch)
        self._logger = logger
        self._master_ip = master_ip
        self._master_port = master_port
        self._master_loc = f'{master_ip}:{master_port}'
    
    def schedule(self, command):
        request = s2m_pb2.ScheduleRequest(command=command)
        with grpc.insecure_channel(self._master_loc) as channel:
            self._logger.debug('Tring to send scheduling command...')
            stub = s2m_pb2_grpc.SchedulerToMasterStub(channel)
            response = stub.Schedule(request)
            if response.success:
                self._logger.info(
                    f"Client : Successfully sent command {command}"
                )
            else:
                self._logger.error("Failed to send scheduling command!")

        
