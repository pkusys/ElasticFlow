import io, os, json, csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", type=str)
parser.add_argument('-t', "--trace", type=str)
args = parser.parse_args()

def time2timestamp(time_str):
	# 2022-06-04T04:55:19Z
	day = int(time_str.split('T')[0].split('-')[-1])
	hour = int(time_str.split('T')[1].split(':')[0])
	minute = int(time_str.split('T')[1].split(':')[1])
	second = int(time_str.split('T')[1].split(':')[2][:-1])
	return second + 60 * minute + 3600 * hour + 24 * 3600 * day

def time2timestamp_completiontime(time_str):
	# 2022-06-04T04:55:19Z
	day = int(time_str.split('T')[0].split('-')[-1])
	hour = int(time_str.split('T')[1].split(':')[0])
	minute = int(time_str.split('T')[1].split(':')[1])
	second = int(time_str.split('T')[1].split('.')[0].split(':')[2])
	return second + 60 * minute + 3600 * hour + 24 * 3600 * day

job_info = {}
with open(args.trace) as trace_file:
    reader = csv.DictReader(trace_file, delimiter=',')
    keys = reader.fieldnames
    start_time = 3733274
    for row in reader:
        job_info[row['job_id']] = int(row['ddl']) - int(row['submit_time'])

print(job_info)
abnormal_jobs = 0
accepted_jobs = 0
declined_jobs = 0
# make sure that there is only one json in the file
with open(args.input, 'r') as f:
	job_data = json.load(f)
	jobs = job_data["submitted_jobs"]
	for job in jobs:
		if job["completion_time"] is None:
			abnormal_jobs += 1
			print("abnormal / running / pending job", job['name'], "ddl:", job_info[job['name']])
			continue
		submitted_time = time2timestamp(job["submission_time"])
		completion_time = time2timestamp_completiontime(job["completion_time"])
		jct = completion_time - submitted_time
		if job['name'] in job_info:
			if jct <= job_info[job['name']]:
				accepted_jobs += 1
				print("accepted job", job['name'], "jct:", jct, "ddl: ", job_info[job['name']])
			else:
				declined_jobs += 1
				print("missed job", job['name'], "jct:", jct, "ddl: ", job_info[job['name']])
print("abnormal_jobs:", abnormal_jobs)
print("accepted_jobs", accepted_jobs)
print("declined_jobs", declined_jobs)

