import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import math
import numpy as np
import random
import csv

from job import Job
from job_table import JobTable
import utils

def generate_interarrival_time(rng, lam):
    return -math.log(1.0 - rng.random()) * lam

def generate_duration(durations, rng):
    if rng.random() >= 0.80:
        run_time = 60 * (10 ** rng.uniform(2.5, 4))
    else:
        run_time = 60 * (10 ** rng.uniform(1, 2.5))
    return run_time

def generate_scale_factor(rng, max_scale_factor=16):
    # Sample the scale factor from the ITP distribution.
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

def parse_throughput_file(throughput_file):
    model_name = throughput_file[:-4].split('/')[-1]
    tmp_dict = dict()
    fd = open(throughput_file, 'r')
    deli = ','
    if ((throughput_file.find('.csv') == (len(throughput_file) - 4))):
        deli = ','
    elif ((throughput_file.find('.txt') == (len(throughput_file) - 4))):
        deli = ' '

    reader = csv.DictReader(fd, delimiter = deli)
    keys = reader.fieldnames
    #utils.print_fn('--------------------------------- Read throughput information from: %s ---------------------------------' % throughput_file) 
    #utils.print_fn('    we get throughputs for the following numbers of GPUs:\n        %s' % keys[1:])
    for row in reader:
        #a new model
        global_batch_size = row['global_batch_size']
        del row['global_batch_size']
        tmp_dict[global_batch_size] = row
    print(model_name, tmp_dict)
    fd.close()
    return tmp_dict

def main(args):
    job_generator = random.Random()
    job_generator.seed(args.seed)

    interarrival_time_generator = random.Random()
    interarrival_time_generator.seed(args.seed + 1)

    duration_generator = random.Random()
    duration_generator.seed(args.seed + 2)

    scale_factor_generator = random.Random()
    scale_factor_generator.seed(args.seed + 3)
    #scale_factor_generator = generate_scale_factor

    throughputs = dict()
    for each_file in os.listdir(args.throughputs_dir):
        throughputs[each_file[:-4]] = parse_throughput_file(
            args.throughputs_dir + "/" + each_file)
    print(throughputs)


    durations = np.linspace(args.min_duration, args.max_duration,
                            args.num_durations)
    duration_generator_func = lambda rng: generate_duration(durations, rng)

    prev_arrival_time = None
    with open(args.output_file, 'w') as f:
        f.write('job_id,submit_time,iteration,model_name,ddl,batch_size,num_gpu,duration,best_effort\n')
        for i in range(args.num_jobs):
            job = utils.generate_job(
                    throughputs=throughputs,
                    rng=job_generator,
                    job_id=None,
                    fixed_job_duration=None,
                    generate_multi_gpu_jobs=args.generate_multi_gpu_jobs,
                    generate_multi_priority_jobs=args.generate_multi_priority_jobs,
                    scale_factor_generator_func=generate_scale_factor,
                    duration_generator_func=duration_generator_func,
                    scale_factor_rng=scale_factor_generator,
                    duration_rng=duration_generator,
                    always_generate_scale_factor=False,
                    max_scale_factor=args.max_scale_factor)
            if prev_arrival_time is None:
                arrival_time = 0
            elif args.lam > 0:
                interarrival_time = \
                    generate_interarrival_time(interarrival_time_generator,
                                               args.lam)
                arrival_time = prev_arrival_time + interarrival_time
            prev_arrival_time = arrival_time
            #f.write('%s\t%d\n' % (str(job), arrival_time))
            #todo: ddl, batch size
            ddl = arrival_time + int(random.uniform(args.min_ddl, args.max_ddl) * job.duration)
            #ddl = arrival_time + int(0.8 * job.duration)
            be_r = random.uniform(0, 1)
            be = 0
            if be_r < args.best_effort_percentage:
                be = 1
            f.write('%d,%d,%d,%s,%d,%d,%d,%d,%d\n' % (i, 
                arrival_time, job.total_steps, job.name, ddl, job.batch_size, 
                job.scale_factor, job.duration, be)) #random.choice([0, 1])

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic trace')
    parser.add_argument('--num_jobs', type=int, required=True,
                        help='Number of jobs to generate')
    parser.add_argument('-l', '--lam', type=float, default=0.0,
                        help='Lambda for Poisson arrival rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--throughputs_dir', type=str,
                        default=('../scheduler/throughputs_A100'),
                        help='Oracle throughputs file')
    parser.add_argument('-a', '--min_duration', type=float, default=1,
                        help='Minimum job duration in hours')
    parser.add_argument('-b', '--max_duration', type=float, default=4,
                        help='Maximum job duration in hours')
    parser.add_argument('-s', '--max_scale_factor', type=int, default=16,
                        help='Maximum number of GPUs for a job')
    parser.add_argument('-n', '--num_durations', type=int, default=4,
                        help='Number of possible job durations')
    parser.add_argument('-m', '--generate-multi-gpu-jobs', action='store_true',
                        default=True,
                        help=('If set, generates multi-GPU jobs according to '
                              'a pre-defined distribution'))
    parser.add_argument('--generate-multi-priority-jobs', action='store_true',
                        default=False,
                        help=('If set, generates some jobs with higher priority'))
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file name')
    parser.add_argument('--best_effort_percentage', type=float, default=0,
                        help='The percentage of best effort jobs')
    parser.add_argument('--max_ddl', type=float, default=1.5)
    parser.add_argument('--min_ddl', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
