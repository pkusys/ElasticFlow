from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
#import utils

#pre-run throughput information
THROUGHPUTS = {}

def parse_throughput_file(throughput_file):
    model_name = throughput_file[:-4].split('/')[-1]
    THROUGHPUTS[model_name] = dict()
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
        THROUGHPUTS[model_name][global_batch_size] = row
    print(model_name, THROUGHPUTS[model_name])
    fd.close()

"""JOBS = _TFJobs()


_allowed_symbols = [
    'JOBS'
]"""
