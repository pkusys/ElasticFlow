import logging
import argparse
import time
import threading
from utils.util import get_global_rank
from runtime.rpc import master_server
from runtime.rpc import master_client
from job import NodeJob

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class WorkerInfo:
    """
    The information of the registered worker_server
    """
    def __init__(self, ip_addr, port, num_gpus):
        self._ip_addr = ip_addr
        self._port = port
        self._num_gpus = num_gpus
        self._gpu_list = [1] * num_gpus
    
    @property
    def ip_addr(self):
        return self._ip_addr
    
    @property
    def port(self):
        return self._port

    @property
    def location(self):
        return '%s:%d' % (self._ip_addr, self._port)

    @property
    def gpu_list(self):
        return self._gpu_list
    
    @property
    def num_gpus(self):
        return self._num_gpus

class JobInfo:
    """
    The information of the all jobs in history
    """
    def __init__(self, job_id, worker_id):
        self._id = job_id
        self._updated_location = None
        self._gpu_set = {}
        self._old_gpu_set = {}
        self._last_reschedule_timestamp = None
        self._node_num = None
        self._gpu_num = 0
        self._master_worker = worker_id
        self._job_descriptions = {}

    def _fetch_GPU_list(self, compressed_list):
        """ decode the gpu_list from the compressed integer
        @param compressed_list -> uint32 : i-th bit equals 1 means the 2^i-th gpu is used, and equals 0 otherwise
        @return decoded_list -> list of str : the gpu list that this job uses on this specific node (the gpu index is sorted increasingly)
        """
        decoded_list = []
        for i in range(32):
            if ((1 << i) & compressed_list) != 0:
                decoded_list.append(str(i))
        return decoded_list

    @property
    def updated_location(self):
        return self._updated_location

    @property
    def master_worker(self):
        return self._master_worker

    @property
    def gpu_set(self):
        return self._gpu_set

    @property
    def gpu_num(self):
        return self._gpu_num

    @property
    def old_gpu_set(self):
        return self._old_gpu_set

    @property
    def job_descriptions(self):
        return self._job_descriptions

    @property
    def last_reschedule_timestamp(self):
        return self._last_reschedule_timestamp

    def update(self, command):
        reschedule, updated = False, False
        if self._last_reschedule_timestamp is not None:
            assert int(command[-1]) >= self._last_reschedule_timestamp
        worker_id = int(command[1])
        job_description = command[2:-1]
        if self._last_reschedule_timestamp is None or int(command[-1]) > self._last_reschedule_timestamp:
            # update model location
            self._gpu_num = 0
            if self._last_reschedule_timestamp is not None:
                reschedule = True
                self._updated_location = (list(self._gpu_set.keys())[0], 
                    self._gpu_set[list(self._gpu_set.keys())[0]][0])
            self._last_reschedule_timestamp = int(command[-1])
            del self._old_gpu_set
            self._old_gpu_set = self._gpu_set
            self._gpu_set = {}
            self._node_num = int(command[6])
            self._master_worker = worker_id
            del self._job_descriptions
            self._job_descriptions = {}
        assert len(list(self._gpu_set.keys())) < self._node_num
        assert worker_id not in self._gpu_set and worker_id not in self._job_descriptions
        self._gpu_set[worker_id] = self._fetch_GPU_list(int(command[10]))
        self._gpu_num += len(self._gpu_set[worker_id])
        self._job_descriptions[worker_id] = job_description
        if len(list(self._gpu_set.keys())) == self._node_num:
            updated = True # all nodes' commands have been ready
        return reschedule, updated


class Master:
    """ 
    The master runs a master_server in background to collect information from workers,
    and designate jobs to the registered workers according to the trace file
    """
    def __init__(self, port, trace_file=None, debug=False, num_workers=None):
        # Configure logger.
        logger = logging.getLogger("master")
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        logger.addHandler(ch)

        # setup attributes
        self._logger = logger
        self._workers = {} # each element is a ( worker_id : WorkerInfo )
        self._jobs = {} # each element is a ( job_id : JobInfo )
        self._free_worker_id = 0
        self._cmds = []
        self._debug = debug
        self._schedule_counts = 0
        callbacks = {
            'RegisterWorker' : self._register_worker_callback,
            'Done' : self._done_callback, 
            "Schedule": self._schedule_callback,
        }
    
        # run master rpc server in the background 
        self._server_thread = threading.Thread(
            target=master_server.serve,
            args=(port, callbacks))
        self._server_thread.setDaemon(True)
        self._server_thread.start()

        self._master_rpc_client = master_client.MasterRpcClient()
        self._timestamp = None
        print("debug", debug, num_workers)

        if not self._debug:
            self._wait_for_workers(num_workers)
            while True:
                time.sleep(5)
        else:
            assert trace_file is not None
            # parse the trace file
            num_workers = self._parse_tracefile(trace_file)
            self._wait_for_workers(num_workers)
            for cmd in self._cmds:
                self.debug_run(cmd)


    def _wait_for_workers(self, num_workers):
        # wait for all the workers to regster
        while self._free_worker_id < num_workers:
            time.sleep(5)
        # wait for all the trainers to be launched
        time.sleep(5)

    def shutdown(self):
        # shut down all the workers
        for worker in self._workers.values():
            self._master_rpc_client.shutdown(worker.location)
    
    def _restart_trainers(self, worker):
        self._master_rpc_client.restart_trainers(worker.location)

    def restart_trainers(self):
        threads = []
        for worker in self._workers.values():
            #self._master_rpc_client.restart_trainers(worker.location)
            rt_thread = threading.Thread(
                target=self._restart_trainers,
                args=(worker,))
            threads.append(rt_thread)
            rt_thread.start()
        for rt_thread in threads:
            while rt_thread.is_alive():
                time.sleep(1)
    
    def debug_run(self, cmd):
        if cmd[0] == "K":
            worker_id = int(cmd[1])
            job_id = int(cmd[2])
            self._master_rpc_client.kill_job(job_id, self._workers[worker_id].location)
            self._logger.info(f"received kill reply from worker {worker_id}")
        elif cmd[0] == "R":
            worker_id = int(cmd[1])
            job_description = cmd[2:12]
            from_scratch = int(cmd[12])
            ranks = list(map(int, cmd[13:]))
            self._master_rpc_client.run_job(NodeJob(*job_description, ranks, False, from_scratch), self._workers[worker_id].location)
            self._logger.info(f"received run reply from worker {worker_id}")
        elif cmd[0] == "U":
            worker_id = int(cmd[1])
            job_description = cmd[2:]
            self._master_rpc_client.update_job(NodeJob(*job_description), self._workers[worker_id].location)
            self._logger.info(f"received update reply from worker {worker_id}")
            for each_worker in self._worker_stable:
                self._worker_stable[each_worker] = 0
        elif cmd[0] == "G":
            for_ddp = (int(cmd[1]) == 1)
            job_id = int(cmd[2])
            if for_ddp == 1:
                ranks = [int(rank) for rank in cmd[3:]]
                src = None
            else:
                ranks = [int(rank) for rank in cmd[3:-1]]
                src = int(cmd[-1])
            for worker_id in self._workers:
                self._master_rpc_client.new_group(for_ddp, job_id, ranks, src, self._workers[worker_id].location)
                self._logger.info(f"received new_group reply from worker {worker_id}")
        elif cmd[0] == "F":
            self._schedule_counts += 1
            for worker_id in self._workers:
                self._master_rpc_client.broadcast_finish(self._workers[worker_id].location)
            self._logger.info(f"received broadcast finish reply from worker {worker_id}")
        elif cmd[0] == "T":
            # begin to train onall trainers
            for worker_id in self._workers:
                self._master_rpc_client.training_begin(self._workers[worker_id].location)
            self._logger.info(f"received training begin reply from worker {worker_id}")
        elif cmd[0] == "W":
            seconds = int(cmd[1])
            self._logger.info(f"master will sleep {seconds} seconds")
            time.sleep(seconds)
        elif cmd[0] == "S":
            self.shutdown()
        elif cmd[0] == "RE":
            self.restart_trainers()

    def run_command(self, cmd):
        if cmd[0] == "K":
            worker_id = int(cmd[1])
            job_id = int(cmd[2])
            self._master_rpc_client.kill_job(job_id, self._workers[worker_id].location)
            self._logger.info(f"received kill reply from worker {worker_id}")
        elif cmd[0] == "R":
            self._timestamp = int(cmd[-1])
            worker_id = int(cmd[1])
            job_id = int(cmd[4])
            if job_id not in self._jobs:
                self._jobs[job_id] = JobInfo(job_id, worker_id)
                cmd[8] = self._workers[worker_id].ip_addr
            else:
                # get ip address of every worker
                cmd[8] = self._workers[self._jobs[job_id].master_worker].ip_addr
            reschedule, updated = self._jobs[job_id].update(cmd)
            if updated:
                src = None
                global_ranks = []
                for worker_id in self._jobs[job_id].gpu_set:
                    for local_rank in self._jobs[job_id].gpu_set[worker_id]:
                        # Assume that the number of GPU on each node is the same
                        global_ranks.append(get_global_rank(worker_id, local_rank, self._workers[worker_id].num_gpus))
                for worker_id in self._workers:
                    self._master_rpc_client.new_group(
                        True, job_id, global_ranks, src, self._workers[worker_id].location)
                ranks = list(range(self._jobs[job_id].gpu_num))
                for worker_id in self._jobs[job_id].job_descriptions:
                    # train from scratch only on the first run
                    self._master_rpc_client.run_job(
                        NodeJob(*self._jobs[job_id].job_descriptions[worker_id], ranks, False, 
                            len(self._jobs[job_id].old_gpu_set)==0), 
                        self._workers[worker_id].location)
            self._logger.info(f"received run reply from worker {worker_id}")

        elif cmd[0] == "F":
            self._schedule_counts += 1
            # send groups for broadcasts
            for job_id in self._jobs:
                if self._jobs[job_id].last_reschedule_timestamp is None:
                    continue
                if self._jobs[job_id].last_reschedule_timestamp != self._timestamp:
                    continue
                if len(list(self._jobs[job_id].old_gpu_set.keys())) == 0:
                    continue
                global_ranks = []
                old_worker_id = None
                for worker_id in self._jobs[job_id].old_gpu_set:
                    for local_rank in self._jobs[job_id].old_gpu_set[worker_id]:
                        src = get_global_rank(worker_id, local_rank, self._workers[worker_id].num_gpus)
                        if int(src) not in global_ranks:
                            global_ranks.append(int(src))
                            if old_worker_id is None:
                                old_worker_id = worker_id
                rank_0 = None
                for worker_id in self._jobs[job_id].gpu_set:
                    if worker_id in self._jobs[job_id].old_gpu_set:
                        break
                    for local_rank in self._jobs[job_id].gpu_set[worker_id]:
                        rank_0 = get_global_rank(worker_id, local_rank, self._workers[worker_id].num_gpus)
                        break 
                    break 

                # broadcast only when new rank 0 and old rank 0 are on different nodes
                if rank_0 is not None and len(global_ranks) != 0 and rank_0 not in global_ranks:
                    old_rank_0 = global_ranks[0]
                    for worker_id in self._jobs[job_id].gpu_set:
                        self._master_rpc_client.new_group(
                            False, job_id, [min(rank_0, old_rank_0), max(rank_0, old_rank_0)], old_rank_0, self._workers[worker_id].location)
                    if old_worker_id not in self._jobs[job_id].gpu_set:
                        self._master_rpc_client.new_group(
                            False, job_id, [min(rank_0, old_rank_0), max(rank_0, old_rank_0)], old_rank_0, self._workers[old_worker_id].location)

            
            # broadcast finish
            for worker_id in self._workers:
                self._master_rpc_client.broadcast_finish(self._workers[worker_id].location)
                self._logger.info(f"received broadcast finish reply from worker {worker_id}")
        elif cmd[0] == "T":
            # begin to train onall trainers
            for worker_id in self._workers:
                self._master_rpc_client.training_begin(self._workers[worker_id].location)
                self._logger.info(f"received training begin reply from worker {worker_id}")

        elif cmd[0] == "W":
            seconds = int(cmd[1])
            self._logger.info(f"master will sleep {seconds} seconds")
            time.sleep(seconds)
        elif cmd[0] == "RE":
            self.restart_trainers()
        elif cmd[0] == "S":
            self.shutdown()

    def _parse_tracefile(self, trace_file):
        """ Parse the trace file into command flow
        @param trace_file : the name of the trace file
        @return : the number of workers
        """
        with open(trace_file, 'r') as f:
            num_workers = int(f.readline().strip())
            for line in f:
                self._cmds.append(line.split())
        return num_workers
        
    def _register_worker_callback(self, ip_addr=None, port=None, num_gpus=1):
        """Registers a worker with the master.

        Initializes state for a new worker in a WorkerInfo class, and assigns it an id.
        The worker provides an IP address and port for its RPC server
        so that the scheduler can establish an RPC client for
        scheduler-to-worker communication.         

        @param ip_addr: IP address of the worker's RPC server.
        @param port: Port number for the worker's RPC server.
        @param num_gpus: The number of GPUs available on the worker.
        
        @return: The worker_id of the newly registered worker.
        """
        worker_id = self._free_worker_id
        self._workers[self._free_worker_id] = WorkerInfo(ip_addr, port, num_gpus)
        self._free_worker_id += 1
        return worker_id
    
    def _done_callback(self, job_id, worker_id):
        """Handles completion of a scheduled job.
        #TODO maintain the pool of available devices 

        @param job_id: The id of the completed job(s).
        @param worker_id: The id of the worker where the job(s) were completed.
        """
        self._logger.info(f"worker {worker_id} finished the job {job_id}")

    def _schedule_callback(self, command):
        """Receives scheduling commands from scheduler.

        @param command: The command following format defined in ./trace.
        """
        self.run_command(command.split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a master process")
    parser.add_argument("-p", "--port", type=int, default=6888,
                        help="Port for the master server")
    parser.add_argument("-t", "--tracefile", type=str, default=None, 
                        help="Tracefile to execute")
    parser.add_argument("-d", "--debug", type=bool, default=False,
                        help="Debug mode for develepment")
    parser.add_argument("-n", "--num_workers", type=int, 
                        help="Number of workers, required when not in debug mode")
    args = parser.parse_args()
    opt_dict = vars(args)
    master = Master(opt_dict['port'], opt_dict['tracefile'], opt_dict['debug'], opt_dict['num_workers'])