# This file can draw the subfigures for Figure 6 in our paper.
# Usage: 
# 1. Change the raw values accordingly (existing numbers are our test results)
# 2. python3 draw_fig6.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fontsizeValue=28

edf=[0.08]
gandiva = [0.24]
tiresias = [0.32]
themis=[0.28]
chronus = [0.4]
pollux = [0.32]
ef = [0.64]

x = np.arange(len(edf))  # the label locations

total_width, n = 0.3, 7
width = total_width / n
x = x - (total_width - width) / 2

fig, (ax1) = plt.subplots(1, 1, sharex=True)

plt.figure(figsize=(7, 7))
#plt.figure(figsize=(10, 7))
plt.grid(True, color="#D3D3D3", zorder=0)
l1 = plt.bar(x-3*width, edf, width, zorder=100,color='white',edgecolor='k',hatch='/')
l2 = plt.bar(x-2*width, gandiva, width, zorder=100,color='white',edgecolor='blue', hatch='\\')
l3 = plt.bar(x-1*width, tiresias, width, zorder=100,color='white',edgecolor='green', hatch='+')
l4 = plt.bar(x, themis, width, zorder=100,edgecolor='orange', hatch='*', color='white')
l5 = plt.bar(x+1*width, chronus, width, zorder=100,edgecolor='k', hatch='x', color='white')
l6 = plt.bar(x+2*width, pollux, width, zorder=100,edgecolor='purple', hatch='-', color='white')
l7 = plt.bar(x+3*width, ef, width, zorder=100)

#plt.xticks(x, label, fontsize=fontsizeValue)

#plt.xlabel("Cluster Size", fontsize=fontsizeValue)
plt.ylabel("Deadline Satisfactory Ratio", fontsize=fontsizeValue)

plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
#plt.tick_params(bottom=False, top=False, left=False, right=False)

d = .5  # proportion of vertical to horizontal extent o

plt.legend([l1, l2, l3, l4, l5, l6, l7], [
  'EDF', 'Tiresias', 'Gandiva', 'Themis', 'Chronus', 'Pollux', 'ElasticFlow'],fontsize=fontsizeValue,
  ncol=3,loc="best",bbox_to_anchor=(1,0),borderaxespad=0, frameon=False)
plt.savefig("fig6a.pdf")

edf=[0.0872]
gandiva = [0.2103]
tiresias = [0.4564]
themis=[0.3897]
chronus = [0.4103]
#pollux = [0.32]
ef = [0.6667]

x = np.arange(len(edf))  # the label locations

total_width, n = 0.3, 7
width = total_width / n
x = x - (total_width - width) / 2

fig, (ax1) = plt.subplots(1, 1, sharex=True)

plt.figure(figsize=(7, 7))
#plt.figure(figsize=(10, 7))
plt.grid(True, color="#D3D3D3", zorder=0)
l1 = plt.bar(x-2.5*width, edf, width, zorder=100,color='white',edgecolor='k',hatch='/')
l2 = plt.bar(x-1.5*width, gandiva, width, zorder=100,color='white',edgecolor='blue', hatch='\\')
l3 = plt.bar(x-0.5*width, tiresias, width, zorder=100,color='white',edgecolor='green', hatch='+')
l4 = plt.bar(x+0.5*width, themis, width, zorder=100,edgecolor='orange', hatch='*', color='white')
l5 = plt.bar(x+1.5*width, chronus, width, zorder=100,edgecolor='k', hatch='x', color='white')
#l6 = plt.bar(x+2*width, pollux, width, zorder=100,edgecolor='purple', hatch='-', color='white')
l7 = plt.bar(x+2.5*width, ef, width, zorder=100)

#plt.xticks(x, label, fontsize=fontsizeValue)

#plt.xlabel("Cluster Size", fontsize=fontsizeValue)
plt.ylabel("Deadline Satisfactory Ratio", fontsize=fontsizeValue)

plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
#plt.tick_params(bottom=False, top=False, left=False, right=False)

d = .5  # proportion of vertical to horizontal extent o

plt.legend([l1, l2, l3, l4, l5, l7], [
  'EDF', 'Tiresias', 'Gandiva', 'Themis', 'Chronus', 'ElasticFlow'],fontsize=fontsizeValue,
  ncol=3,loc="upper center",frameon=False)
plt.savefig("fig6b.pdf")


