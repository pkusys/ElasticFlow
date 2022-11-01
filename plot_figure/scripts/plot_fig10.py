# This file can draw the subfigures for Figure 10 in our paper.
# Usage: 
# 1. Change the dir to the traces accordingly (line9-14)
# 2. python3 plot_fig10.py
import matplotlib.pyplot as plt
import csv
import sys

edf_file = "../logs/figure10/edf/scheduling_events.csv"
elasticflow_file = "../logs/figure10/ef-accessctrl/scheduling_events.csv"
g_file = "../logs/figure10/gandiva/scheduling_events.csv"
t_file = "../logs/figure10/dlas-gpu/scheduling_events.csv"
themis_file = "../logs/figure10/themis/scheduling_events.csv"
c_file = "../logs/figure10/chronus/time-aware-with-lease_resource.csv"
fontsizeValue=32

def hist_gen(filename, debug=False):
    submission_time = []
    ces = []
    last_time = -610
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            submit_time = int(float(line['time']))
            if submit_time - last_time < 480:
                continue
            last_time = submit_time
            ce = float(line['ce'])
            if len(submission_time) > 0:
                submission_time.append(submit_time-1e-6)
                ces.append(ces[-1])
            submission_time.append(submit_time)
            ces.append(ce)
    submission_time = [(t - submission_time[0]) / 3600 for t in submission_time]
    if debug:
        for i,submit_time in enumerate(submission_time):
            print(submit_time, ces[i])
    return {k: v for k, v in zip(submission_time, ces)}


#gandiva = hist_gen(g_file)
tiresias_L = hist_gen(t_file)
#elasticflow = {0:32, 19.08:64, 20:128, 50 * scale:128}
elasticflow = hist_gen(elasticflow_file)
gandiva = hist_gen(g_file)
themis = hist_gen(themis_file)
edf = hist_gen(edf_file)
chronus = hist_gen(c_file)
#edf = {0:32, 6.883:64, 19.08:128, 50 * scale:128}
plt.figure(figsize=(20, 7))
#plt.xlim(0, 48 * scale)
plt.ylim(0, 0.35)
plt.grid(True, color="#D3D3D3")
plt.plot(edf.keys(), edf.values(), label='EDF',linewidth=3)
plt.plot(gandiva.keys(), gandiva.values(), ":", label='Gandiva',linewidth=3.0)
plt.plot(tiresias_L.keys(), tiresias_L.values(), "--", label='Tiresias',linewidth=3.0)
plt.plot(themis.keys(), themis.values(), "--", label='Themis',linewidth=3.0)
plt.plot(chronus.keys(), chronus.values(), "-.", label='Chronus',linewidth=3.0)
plt.plot(elasticflow.keys(), elasticflow.values(), label='ElasticFlow',linewidth=3.0)
#plt.xticks([0, 12 * scale, 24 * scale, 36 * scale, 48 * scale], [0, 12, 24, 36, 48])
#plt.yticks([0, 20, 40, 60, 80, 100, 120, 128])
plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
plt.tick_params(bottom=False, top=False, left=False, right=False)
plt.xlabel("Time (hour)", fontsize=fontsizeValue)
plt.ylabel("Cluster Efficiency", fontsize=fontsizeValue)
plt.legend(fontsize=fontsizeValue, frameon=False)
plt.savefig("fig10.pdf")
