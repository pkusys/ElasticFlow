import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import csv
import math
import numpy as np
import random

from job import Job
from job_table import JobTable
import utils

def generate_interarrival_time(rng, lam):
    return -math.log(1.0 - rng.random()) * lam

def generate_duration(durations, rng):
    """if rng.random() >= 0.95:
        run_time = 60 * (10 ** rng.uniform(3, 4))
    else:
        run_time = 60 * (10 ** rng.uniform(1, 3))"""
    if rng.random() >= 0.80:
        run_time = 60 * (10 ** rng.uniform(2.5, 4))
    else:
        run_time = 60 * (10 ** rng.uniform(1, 2.5))
    return run_time
    return 3600 * rng.choice(durations)

def generate_scale_factor(rng):
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

    throughputs = dict()
    for each_file in os.listdir(args.throughputs_dir):
        throughputs[each_file[:-4]] = parse_throughput_file(
            args.throughputs_dir + "/" + each_file)
    print(throughputs)

    with open(args.output_file, 'w') as f:
        f.write('job_id,submit_time,iteration,model_name,ddl,batch_size,num_gpu,duration\n')
        input_file = open(args.source_file, "r", newline='')
        reader = csv.DictReader(input_file)
        for row in reader:
            if row['status'] == 'failed':
                continue
            if int(row['submit_time']) < 0:
                    continue
            if int(row['gpus']) % 8 != 0:
                if int(row['gpus']) == 3 or int(row['gpus']) == 5 or int(row['gpus']) == 6 or int(row['gpus']) == 7:
                    continue
            if int(row['duration']) < 10 * 60:
                continue
            if int(row['gpus']) > 64:
                continue
            job = utils.generate_job_from_itp(
                    throughputs=throughputs,
                    reference_worker_type='v100',
                    rng=job_generator,
                    job_id=row['job_id'],
                    scale_factor=int(row['gpus']),
                    fixed_job_duration=int(row['duration']))
            if job is False:
                continue
            #f.write('%s\t%d\n' % (str(job), arrival_time))
            #todo: ddl, batch size
            ddl = int(row['submit_time']) + int(random.uniform(args.min_ddl, args.max_ddl) * job.duration)
            f.write('%s,%d,%d,%s,%d,%d,%d,%d\n' % (row['job_id'], 
                int(row['submit_time']), job.total_steps, job.name, ddl, job.batch_size, 
                job.scale_factor, job.duration))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic trace')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--throughputs_dir', type=str,
                        default=('../scheduler/throughputs_A100'),
                        help='Oracle throughputs file')
    parser.add_argument('--source_file', type=str, required=True,
                        help='original collected ITP trace')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file name')
    parser.add_argument('--max_ddl', type=float, default=1.5)
    parser.add_argument('--min_ddl', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
