import argparse
import csv
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trace', type=str, required=True,
	help='trace file')
parser.add_argument('-o', '--output_file', type=str, required=True,
	help='ElasticFlow trace file')
args = parser.parse_args()
name_list = []
fd = open(args.trace, 'r')
deli = ','
reader = csv.DictReader(fd, delimiter = deli)
keys = reader.fieldnames
for row in reader:
	if row['user'] not in name_list:
		name_list.append(row['user'])
fd.close()
f = open(args.output_file, 'w')
for name in name_list:
	f.write(name + "\n")
f.close()