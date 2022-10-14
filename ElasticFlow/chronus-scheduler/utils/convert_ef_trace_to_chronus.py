import argparse
import csv
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trace', type=str, required=True,
	help='ElasticFlow trace file')
parser.add_argument('-o', '--output_file', type=str, required=True,
	help='ElasticFlow trace file')
args = parser.parse_args()
fd = open(args.trace, 'r')
deli = ','
reader = csv.DictReader(fd, delimiter = deli)
keys = reader.fieldnames
# read csv file, convert iter to duration
# todo: real_duration
# job_id,user,num_gpu,submit_time,iterations,model_name,duration,interval,expect_time_list,expect_value_list,best_effort,job_type
with open(args.output_file, 'w') as f:
	f.write("job_id,user,num_gpu,batch_size,submit_time,iterations,model_name,duration,interval,expect_time_list,expect_value_list,best_effort,job_type\n")
	cnt = 0
	for row in reader:
		user = 'user' + str(cnt)
		cnt += 1
		f.write('%s,%s,%s,%s,%s,%s,%s,%s,%d,%s,%d,%d,%s\n' % (row['job_id'], user, row['num_gpu'], row['batch_size'],
			str(int(row['submit_time'])), row['iteration'], row['model_name'], row['duration'],
			0, str(int(row['ddl'])-int(row['submit_time'])), 10, 0, 'strict'))
