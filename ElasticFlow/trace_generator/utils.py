import csv
from datetime import datetime
import json
import os
import pickle
import psutil
import random
import re
import socket
import subprocess

from job import Job
from job_table import JobTable

def _generate_scale_factor(rng, max_scale_factor=16):
    # Sample the scale factor from the Philly distribution.
    scale_factor = 1
    r = rng.uniform(0, 1)
    if 0.83 <= r <= 0.86:
        scale_factor = 2
    elif 0.86 <= r <= 0.89:
        scale_factor = 4
    elif 0.89 <= r <= 0.99:
        scale_factor = 8
    elif 0.99 <= r:
        #scale_factor = random.choice([16, 32, 64])
        scale_factor = 16
    if scale_factor > max_scale_factor:
        scale_factor = max_scale_factor
    return scale_factor

def _generate_duration(rng):
    # Sample the job duration from the Philly distribution.
    """if rng.random() >= 0.8:
        run_time = 60 * (10 ** rng.uniform(3, 4))
    else:
        run_time = 60 * (10 ** rng.uniform(1.5, 3))
    return run_time"""
    if rng.random() >= 0.95:
        run_time = 60 * (10 ** rng.uniform(3, 4))
    else:
        run_time = 60 * (10 ** rng.uniform(1, 3))
    return run_time

def generate_job(throughputs, rng=None,
                 job_id=None, fixed_job_duration=None,
                 generate_multi_gpu_jobs=False,
                 generate_multi_priority_jobs=False, run_dir=None,
                 scale_factor_generator_func=_generate_scale_factor,
                 duration_generator_func=_generate_duration,
                 scale_factor_rng=None, duration_rng=None, SLO_rng=None,
                 always_generate_scale_factor=True,
                 max_scale_factor=16):
    """Generates a new job.

       Args:
         throughputs: A dict containing pre-measured throughputs.
         rng: A random number generator for selecting job parameters.
         job_id: The job's ID.
         fixed_job_duration: If set, fixes the duration to the specified value.
         generate_multi_gpu_jobs: If set, generate a scale factor >= 1.
         generate_multi_priority_jobs: If set, generate a priority >= 1.
         run_dir: The directory to run the job from.
         scale_factor_generator_func: A function that accepts an RNG parameter
                                      and returns a job size.
         duration_generator_func: A function that accepts an RNG parameter and
                                  returns a job duration in seconds.
         scale_factor_rng: A random number generator specifically for
                           generating scale factors.
         duration_rng: A random number generator specifically for generating
                       durations.
         SLO_rng: If set, generate an SLO >= 1 using this RNG.
         always_generate_scale_factor: If set, generate a scale factor
                                       regardless of whether user has
                                       requested multi-GPU jobs.
      Returns:
        The generated Job.
    """
    if rng is None:
        rng = random.Random()
    if scale_factor_rng is None:
        scale_factor_rng = rng
    if duration_rng is None:
        duration_rng = rng

    job_template = None

    if always_generate_scale_factor:
        scale_factor = scale_factor_generator_func(scale_factor_rng, max_scale_factor)
    else:
        # NOTE: We select the job template here to maintain backwards
        # compatability with scripts/utils/generate_trace.py
        #if generate_multi_gpu_jobs and job_template.distributed:
        if generate_multi_gpu_jobs:
            scale_factor = scale_factor_generator_func(scale_factor_rng, max_scale_factor)
        else:
            scale_factor = 1
        job_template = rng.choice(JobTable)

    if fixed_job_duration:
        run_time = fixed_job_duration
    else:
        run_time = duration_generator_func(duration_rng)
    if not generate_multi_gpu_jobs:
        scale_factor = 1
    assert(run_time > 0)
    assert(scale_factor >= 1 and scale_factor <= 16)

    # Sample the job type.
    if job_template is None:
        while True:
            job_template = rng.choice(JobTable)
            if (scale_factor == 1 or
                (scale_factor > 1 and job_template.distributed)):
                break
    assert job_template.name in throughputs

    if str(job_template.batch_size) not in throughputs[job_template.name]:
        print(str(job_template.batch_size), throughputs[job_template.name])
    assert str(job_template.batch_size) in throughputs[job_template.name]
    print(scale_factor, job_template.name)
    assert str(scale_factor) in throughputs[job_template.name][str(job_template.batch_size)]
    num_steps = run_time * float(throughputs[
    job_template.name][str(job_template.batch_size)][str(scale_factor)])
    assert(num_steps > 0)

    # Optionally assign a priority to the job.
    priority_weight = 1.0
    if generate_multi_priority_jobs:
        r = rng.uniform(0, 1)
        if 0.0 <= r <= 0.2:
            priority_weight = 5.0

    # Optionally assign an SLO to the job.
    SLO = None
    if SLO_rng is not None:
        r = SLO_rng.uniform(0, 1)
        if 0.0 <= r < 0.33:
            SLO = 1.2
        elif 0.33 <= r < 0.67:
            SLO = 2.0
        else:
            SLO = 10.0
    print("job_id", job_id,  "num_steps", num_steps, 
        "duration", run_time, "batch_size", job_template.batch_size)

    job = Job(job_id=job_id,
              job_type=None,
              batch_size=job_template.batch_size,
              command=None,
              working_directory=job_template.working_directory,
              num_steps_arg=job_template.num_steps_arg,
              total_steps=num_steps,
              duration=run_time,
              scale_factor=scale_factor,
              priority_weight=priority_weight,
              SLO=SLO,
              needs_data_dir=job_template.needs_data_dir,
              name=job_template.name)

    return job

def generate_job_from_itp(throughputs, rng=None,
                 job_id=None, fixed_job_duration=None,
                 generate_multi_gpu_jobs=False,
                 generate_multi_priority_jobs=False, run_dir=None,
                 scale_factor=None):

    job_template = None

    while job_template is None or job_template.batch_size < scale_factor:
        job_template = rng.choice(JobTable)

    run_time = fixed_job_duration
    
    assert(run_time > 0)
    #assert(scale_factor >= 1 and scale_factor <= 8)

    assert job_template.name in throughputs

    assert str(job_template.batch_size) in throughputs[job_template.name]
    print(scale_factor, job_template.name)
    if str(scale_factor) not in throughputs[job_template.name][str(job_template.batch_size)]:
        return False
    assert str(scale_factor) in throughputs[job_template.name][str(job_template.batch_size)]
    num_steps = run_time * float(throughputs[
    job_template.name][str(job_template.batch_size)][str(scale_factor)])
    #num_steps = run_time * 1.817190623296384
    assert(num_steps > 0)

    # Optionally assign a priority to the job.
    priority_weight = 1.0
    if generate_multi_priority_jobs:
        r = rng.uniform(0, 1)
        if 0.0 <= r <= 0.2:
            priority_weight = 5.0

    print("job_id", job_id, "num_steps", num_steps, 
        "duration", run_time, "batch_size", job_template.batch_size)

    job = Job(job_id=job_id,
              job_type=None,
              batch_size=job_template.batch_size,
              command=None,
              working_directory=job_template.working_directory,
              num_steps_arg=job_template.num_steps_arg,
              total_steps=num_steps,
              duration=run_time,
              scale_factor=scale_factor,
              priority_weight=priority_weight,
              needs_data_dir=job_template.needs_data_dir,
              name=job_template.name)

    return job


def get_ip_address():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def get_num_gpus():
    command = 'nvidia-smi -L'
    output = subprocess.run(command, stdout=subprocess.PIPE, check=True,
                            shell=True).stdout.decode('utf-8').strip()
    return len(output.split('\n'))

def get_pid_for_job(command):
    pids = []
    for proc in psutil.process_iter():
        cmdline = ' '.join(proc.cmdline())
        if cmdline == command:
            pids.append(proc.pid)
    return min(pids)

def get_gpu_processes():
    output = subprocess.check_output('nvidia-smi').decode('utf-8')
    gpu_processes = {}
    processes_flag = False
    for line in output.split('\n'):
        if 'Processes' in line:
            processes_flag = True
            continue
        if processes_flag:
            res = re.search('(\d+) +(\d+) +(\w+) +(.+) +(\d+)MiB', line)
            if res is not None:
                gpu_id = int(res.group(1))
                if gpu_id not in gpu_processes:
                    gpu_processes[gpu_id] = []
                pid = int(res.group(2))
                process_name = res.group(4)
                if process_name != 'nvidia-cuda-mps-server':
                    gpu_processes[gpu_id].append(pid)
    return gpu_processes


def parse_job_type_str(job_type):
    if job_type is None:
        return None
    match = re.match('(.*) \(scale factor (\d+)\)', job_type)
    if match is None:
        return (job_type, 1)
    model = match.group(1)
    scale_factor = int(match.group(2))
    return (model, scale_factor)

def parse_job_type_tuple(job_type):
    match = re.match('\(\'(.*)\', (\d+)\)', job_type)
    if match is None:
        return None
    model = match.group(1)
    scale_factor = int(match.group(2))
    return (model, scale_factor)


def print_allocation(allocation, current_time=None):
    """Prints the allocation.

       Debug method used for printing the allocation of each job on each
       worker type.
    """
    print('=' * 80)
    if current_time is not None:
        print('Allocation\t(Current_time: %f)' % (current_time))
        print('-' * 80)
    for job_id in sorted(list(allocation.keys())):
        allocation_str = 'Job ID %s:' % (job_id)
        for worker_type in sorted(list(allocation[job_id].keys())):
            value = allocation[job_id][worker_type]
            allocation_str += ' [%s: %f]' % (worker_type, value)
        print(allocation_str)
    print('=' * 80)
