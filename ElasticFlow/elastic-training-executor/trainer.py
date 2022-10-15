import sys
import logging
import argparse
import time
from enum import Enum
import threading
import torch.distributed as dist
import datetime
import signal
import pickle
import pandas as pd

import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../scheduler/runtime/rpc'))
import trainer_server, trainer_client
import utils

from resnet.resnet_ddp import ResNetWorkload
from vgg.vgg_ddp import VGGWorkload
from inceptionv3.inceptionv3_ddp import InceptionV3Workload
from GPT2.gpt2_ddp import GPT2Workload
from bert.bert_ddp import BertWorkload
from deepspeech2.deepspeech2_ddp import DeepSpeechWorkload
from torch.multiprocessing import Process
import torch

#LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
LOG_FORMAT = '{name}:{levelname} [{asctime}.{msecs}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
NCCL_TIMEOUT = 5

class STATUS(Enum):
    """ the status of the trainer
    When there is no workload designated from worker, the trainer is in IDLE state.
    When one job comes, the trainer rpc server will process the request and use the callback function
    to initialize the standby workload and response to the worker.
    When the trainer finds its state is SWITCH, it will throw its standby workload, change its state to RUNNING
    and start to run its active workload.
    """
    KILL = 1
    IDLE = 2
    RUNNING = 3
    SWITCH = 4

class Trainer():
    def __init__(self, global_rank, trainer_id, trainer_ip, trainer_port, pg_master_ip, pg_master_port,
        pg_world_size, dynamic_requests, load_checkpoint, scheduler_addr, scheduler_port) -> None:
        logger = logging.getLogger('trainer' + str(trainer_id))
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        ch = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                          style='{')
        formatter.default_msec_format = '%s.%03d'
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # logger
        self._logger = logger
        # global rank of the trainer
        self.global_rank = global_rank
        # trainer's id, which is identical to the GPU id
        self.trainer_id = trainer_id
        # the ip address of the trainer
        self.trainer_ip = trainer_ip
        # the port number of the trainer rpc server
        self.trainer_port = trainer_port
        # the ip address of the process group master
        self.pg_master_ip = pg_master_ip
        # the ip port of the process group master
        self.pg_master_port = pg_master_port
        # the world size of the global process group
        self.pg_world_size = pg_world_size
        # for simulation or not
        self.dynamic_requests = dynamic_requests
        # the status of this trainer, see more in the comments at the top of this file
        self.status = STATUS.IDLE
        # active workload, which run the current training job
        self.active_workload = None
        # standby workload, which may have initialized the incoming job and ready for switch
        self.standby_workload = None
        # job_id -> ddp process group
        self.ddp_process_group = {}
        # job_id -> broadcast src
        self.broadcast_src = {}
        # job_id -> broadcast dst
        self.broadcast_dst = {}
        # job_id -> job name
        self.jobid_to_name = {}
        # checkpoint filename
        self.trainer_checkpoint = f"trainer{self.trainer_id}_state.pickle"
        if load_checkpoint:
            self._load_checkpoint()
        # process groups that include this trainer
        self.process_groups_in_trainer = []
        # ensure there is only one process group in process at a time
        self.cuda_lock = threading.Lock()
        # ensure that broadcast has finished before init DDP model
        self.broadcast_finish = False
        # ensure that the groups are built in exactly the same order
        self.new_group_queue = []
        # job_name -> workload class
        self.workload_list = {
            "resnet50" : ResNetWorkload,
            "vgg16" : VGGWorkload,
            "inception3" : InceptionV3Workload,
            "gpt2" : GPT2Workload,
            "bert" : BertWorkload,
            "deepspeech2" : DeepSpeechWorkload,
        }
        self.model_list = utils.model_list()

        # bitmap cache for sub process groups
        self.process_groups = {}

        # run rpc server to receive messages from worker
        callbacks = {
            "NewGroup" : self._new_group_callback,
            "BroadcastFinish" : self._broadcast_finish_callback,
            "TrainingBegin" : self._training_begin_callback,
            "InitStandby" : self._init_standby_callback,
            "KillActive" : self._kill_active_callback,
            "KillTrainer" : self._kill_trainer_callback,
        }

        torch.cuda.set_device(device=self._get_device())
        # to warm up process groups
        self.flag = torch.Tensor([0]).to(self._get_device())
        # initialize the global process group
        os.environ['MASTER_ADDR'] = self.pg_master_ip
        os.environ['MASTER_PORT'] = str(self.pg_master_port)
        dist.init_process_group(backend="nccl", world_size=self.pg_world_size, rank=self.global_rank)
        assert dist.get_world_size() == self.pg_world_size
        print("$$$ world_size", self.pg_world_size, self.pg_master_ip, str(self.pg_master_port))

        self._server_thread = threading.Thread(
            target=trainer_server.serve,
            args=(trainer_id, trainer_port, callbacks)
        )
        self._server_thread.setDaemon(True)
        self._server_thread.start()

        if self.dynamic_requests:
            self.trainer_rpc_client = trainer_client.TrainerRpcClient(scheduler_addr, scheduler_port)

        self.start_time = time.time()
        # life-long loop
        self.loop()

    def _ranks_to_str(self, rank_list):
        """ decode the rank_list from the compressed string
        @param rank_list -> uint32 : i-th char equals '1' means the 2^i-th rank is used, and equals '0' otherwise
        @return decoded_str -> str : the str that represent the rank list
        """
        decoded_list = []
        for _ in range(self.pg_world_size):
            decoded_list.append('0')
        for rank in rank_list:
            decoded_list[rank] = '1'
        return ''.join(decoded_list)

    def _new_group_callback(self, for_ddp, job_id, ranks, src):
        if for_ddp:
            self.ddp_process_group[job_id] = None
            self.new_group_queue.append((job_id, ranks))
        else:
            if self.global_rank in ranks:
                self.broadcast_src[job_id] = src
                for rank in ranks:
                    if rank != src:
                        self.broadcast_dst[job_id] = rank
            self._logger.info(f"ready for new broadcast between ranks {ranks}")
        return True

    def _broadcast_thread(self):
        trainer_job_id = None
        while len(self.new_group_queue) > 0:
            job_id, ranks = self.new_group_queue[0]
            rank_str = self._ranks_to_str(ranks)
            if self.global_rank in ranks:
                assert trainer_job_id is None # only one job on a GPU
                if rank_str in self.process_groups:
                    trainer_group = self.process_groups[rank_str]
                    self._logger.info("using group in cache")
                else:
                    trainer_group = dist.new_group(ranks=ranks)
                    self.process_groups[rank_str] = trainer_group
                    self.process_groups_in_trainer.append(trainer_group)
                #trainer_group = dist.new_group(ranks=ranks)
                #self.process_groups_in_trainer.append(trainer_group)
                trainer_group_ranks = ranks
                trainer_job_id = job_id
            else:
                if rank_str not in self.process_groups:
                    self.process_groups[rank_str] = dist.new_group(ranks=ranks)
                #dist.new_group(ranks=ranks)
            self._logger.info(f"init new ddp process group for ranks {ranks}")
            self.new_group_queue.pop(0)

        if trainer_job_id is not None:
            self.ddp_process_group[trainer_job_id] = trainer_group
            dist.all_reduce(self.flag, group=self.ddp_process_group[trainer_job_id], async_op=False)
        self._logger.info(f"finish building all process groups")

        if len(self.broadcast_src) == 0:
            self.broadcast_finish = True
            return
        broadcast_queue = list(sorted(self.broadcast_src.keys()))
        while self.standby_workload is not None and self.standby_workload.signal is None:
            time.sleep(0.1)

        for job_id in broadcast_queue:
            self._logger.info(self.jobid_to_name)
            job_name = self.jobid_to_name[job_id]
            src = self.broadcast_src[job_id]
            if src == self.global_rank:
                # this trainer is the sender
                if self.standby_workload is not None and job_id == self.standby_workload.job_id:
                    # this trainer is also the receiver
                    model = self.standby_workload.load_checkpoint()
                else:
                    model = self.workload_list[job_name].load_from_checkpoint(job_id, self._get_device())
                self._logger.info("load job " + str(job_id) + " " + job_name + " from checkpoint for broadcast")
            else:
                # this trainer is the receiver
                model = self.standby_workload.ori_model
            self._logger.info("start broadcast for job " + str(job_id))
            torch.cuda.set_device(device=self._get_device())
            for each in model.state_dict():
                if src == self.global_rank:
                    dist.send(model.state_dict()[each], dst=self.broadcast_dst[job_id])
                else:
                    dist.recv(model.state_dict()[each], src=src)

            self._logger.info("finish broadcast for job " + str(job_id))
        self.broadcast_src = {}
        self.broadcast_finish = True

        return

    def _broadcast_finish_callback(self):
        self.broadcast_finish = False
        bc_thread = threading.Thread(
            target=self._broadcast_thread)
        bc_thread.start()
        return True

    def _training_begin_callback(self):
        self.status = STATUS.SWITCH
        return True

    def _get_device(self):
        return torch.device("cuda", self.trainer_id)

    def _init_standby(self, from_scratch):
        job_id = self.standby_workload.job_id
        self._logger.info(f"[overhead starts] : job {job_id} starts to init dataset")
        self.standby_workload.init_dataset()
        self._logger.info(f"job {job_id} starts to init model")
        self.standby_workload.init_model(1)
        self._logger.info(f"trainer {self.trainer_id} finished initializing the standby workerload job {self.standby_workload.job_id}")
        # Now trainer should still in IDLE state, waiting for broadcast command

    def _init_standby_callback(self, job_id, job_name, master_addr, master_port, rank, local_rank, world_size, batch_size, iterations, from_scratch):
        # TODO error catching if the standby workload is not None
        self.jobid_to_name[job_id] = job_name
        self._logger.info(self.jobid_to_name)
        if job_name not in self.workload_list:
            return False

        self.standby_workload = self.workload_list[job_name](job_id, master_addr, master_port, rank, local_rank, world_size, batch_size // world_size, iterations, self.trainer_id)
        self.standby_workload.ori_model = self.model_list[job_name]

        init_thread = threading.Thread(
            target=self._init_standby,
            args=(from_scratch,))
        init_thread.start()
        return True


    def _mark_active_workload_cleared(self):
        if self.active_workload is None:
            return
        self.active_workload.received_killed = True

    def _clear_standby_workload(self):
        if self.standby_workload is None:
            return
        self.standby_workload.destruction()
        del self.standby_workload
        self.standby_workload = None

    def _kill_active(self):
        self._mark_active_workload_cleared()

    def _kill_active_callback(self, job_id):
        # sanity check
        if self.active_workload is None:
            return True
        if job_id != self.active_workload.job_id:
            return True
        kill_thread = threading.Thread(
            target=self._kill_active,
        )
        kill_thread.start()
        return True

    def _kill_trainer_callback(self):
        self.status = STATUS.KILL
        self._mark_active_workload_cleared()

    def _clear_active_workload(self):
        self.active_workload = None
        self.ddp_process_group = {}
        self.broadcast_src = {}
        torch.cuda.set_device(device=self._get_device())
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device=self._get_device())

    def _wait_for_init_DDP(self):
        while self.standby_workload.job_id not in self.ddp_process_group:
            time.sleep(0.1)
        while self.ddp_process_group[self.standby_workload.job_id] is None:
            time.sleep(0.1)
        while self.standby_workload.signal is None:
            time.sleep(0.1)
        while self.broadcast_finish is False:
            time.sleep(0.1)

    def loop(self):
        self._logger.info(f"trainer {self.trainer_id} is started | local rank : {self.trainer_id} | global rank : {self.global_rank} | pid {os.getpid()}")
        while True:
            if self.status == STATUS.KILL:
                # make sure that the workload is cleared in all workers of this job
                # otherwise, train for another iteration
                if self.active_workload is None:
                    break
            if self.status == STATUS.IDLE:
                time.sleep(2)
                self._logger.info(f"trainer {self.trainer_id} idles ... ...")
                continue
            elif self.status == STATUS.SWITCH:
                if self.standby_workload is None:
                    self.active_workload = self.standby_workload
                    self.standby_workload = None
                    self.status = STATUS.IDLE
                else:
                    self._logger.info(f"trainer {self.trainer_id} switches its active-standby workload")
                    self.cuda_lock.acquire()
                    self._wait_for_init_DDP()
                    self.active_workload = self.standby_workload
                    self.standby_workload = None
                    if self.active_workload is not None:
                        self.active_workload.init_DDP_model(self.ddp_process_group[self.active_workload.job_id])
                        self.active_workload.train_init()
                    self.cuda_lock.release()
                    self.status = STATUS.RUNNING
                    self.start_time = time.time()
                    self.iter_count = 0
                    self._logger.info(f"[overhead ends] : job {self.active_workload.job_id} start training")
            elif self.status == STATUS.RUNNING or self.status == STATUS.KILL:
                # the workload is still running
                self.cuda_lock.acquire()
                if not self.active_workload.train_iteration():
                    self.iter_count += 1
                    if self.active_workload.rank == 0:
                        throughput = self.iter_count / (time.time() - self.start_time)
                        print("TPT: ", throughput)
                        if not args.dynamic_requests:
                            filename = args.throughput_path + self.jobid_to_name[self.active_workload.job_id] + ".csv"
                            #the "empty files need to be prepared."
                            df = pd.read_csv(filename, encoding='utf-8', index_col='global_batch_size')
                            print(df[str(self.active_workload.world_size)])
                            df[str(self.active_workload.world_size)].loc[self.active_workload.batch_size * self.active_workload.world_size] = throughput
                            print(self.active_workload.world_size, "GPUs", self.active_workload.batch_size, "batch size, tpt:", throughput)
                            df.to_csv(filename, encoding='utf-8')
                    self._clear_active_workload()
                    self._logger.info(f"trainer {self.trainer_id} killed its active workload")

                    if self.standby_workload is None or self.standby_workload.signal is None:
                        self._logger.info(f"trainer {self.trainer_id} changes into idle")
                        if self.status == STATUS.RUNNING:
                            self.status = STATUS.IDLE
                    else:
                        self._logger.error(f"the standby_workload should be None when the active_workload is killed or finished")
                    if self.dynamic_requests and self.status != STATUS.KILL:
                        self.trainer_rpc_client.report_ready(int(self.global_rank))
                else:
                    if self.active_workload.rank == 0 and not self.active_workload.stable_reported and self.dynamic_requests:
                        self.trainer_rpc_client.report_stable(int(self.active_workload.job_id))
                        self.active_workload.stable_reported = True
                    # self._logger.info(f"trainer {self.trainer_id} running ... ...")
                self.cuda_lock.release()
                self.iter_count += 1

        self._logger.info(f"trainer {self.trainer_id} will be killed")
        # checkpoint trainer state for restart
        self._checkpoint_state()
        # clear all the resources
        self._logger.info(f"trainer {self.trainer_id} is killed")
        self._clear_active_workload()
        for group in self.process_groups_in_trainer:
            dist.destroy_process_group(group=group)
        dist.destroy_process_group()
        os.kill(os.getpid(), signal.SIGKILL)

    def _checkpoint_state(self):
        with open(self.trainer_checkpoint, 'wb') as checkpoint_file:
            pickle.dump(self.jobid_to_name, checkpoint_file)

    def _load_checkpoint(self):
        with open(self.trainer_checkpoint, 'rb') as checkpoint_file:
            self.jobid_to_name = pickle.load(checkpoint_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trainer process")
    parser.add_argument('-r', '--rank', type=int, required=True,
                        help="the trainer's rank among all the trainers")
    parser.add_argument('-i', '--id', type=int, required=True,
                        help='trainer id, which is identical to the GPU id')
    parser.add_argument('-a', '--addr', type=str, required=True,
                        help='IP address for trainer server')
    parser.add_argument('-p', '--port', type=int, required=True,
                        help='Port number for trainer server')
    parser.add_argument('-A', '--pg_addr', type=str, required=True,
                        help='IP address for process group master')
    parser.add_argument('-P', '--pg_port', type=int, required=True,
                        help='Port number for process group master')
    parser.add_argument('-w', '--world_size', type=int, required=True,
                        help='World size of the process group')
    parser.add_argument('-d', '--dynamic_requests', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='simulation mode or debug mode')
    parser.add_argument('-l', '--load_checkpoint', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='load checkpoint or not')
    parser.add_argument('--scheduler_addr', type=str, help='scheduler server ip address')
    parser.add_argument('--scheduler_port', type=int, help='scheduler server port')
    parser.add_argument('--throughput_path', type=str, default="../scheduler/throughputs_A100/", help='scheduler server port')
    args = parser.parse_args()
    #torch.backends.cudnn.enabled = False
    opt_dict = vars(args)
    print(opt_dict['load_checkpoint'])
    trainer = Trainer(opt_dict['rank'], opt_dict['id'], opt_dict['addr'], opt_dict['port'], opt_dict['pg_addr'],
        opt_dict['pg_port'], opt_dict['world_size'], opt_dict['dynamic_requests'], opt_dict['load_checkpoint'],
        opt_dict['scheduler_addr'], opt_dict['scheduler_port'])

