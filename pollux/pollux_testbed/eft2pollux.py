import os, io, csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", type=str)
parser.add_argument('-o', "--output", type=str)
args = parser.parse_args()

#name,time,application,num_replicas,batch_size,samples
#job_id,submit_time,iteration,model_name,ddl,batch_size,num_gpu,duration
fd = open(args.input, 'r')
reader = csv.DictReader(fd, delimiter=',')
keys = reader.fieldnames
start_time = 3715485

output_file = open(args.output, "a+")
writer = csv.writer(output_file)
header = ["name","time","application", "num_replicas", "batch_size", "samples"]

if not os.path.getsize(args.output):
    writer.writerow(header)

for row in reader:
    name = row['job_id']
    time = str(int(row['submit_time'])-start_time)
    application = row['model_name']
    if application == "inception3":
    	application = "inceptionv3"
    num_replicas = row['num_gpu']
    batch_size = row['batch_size']
    samples = str(int(row['iteration']) * int(row['batch_size']))
    result = [(name, time, application, num_replicas, batch_size, samples)]
    writer.writerows(result)


