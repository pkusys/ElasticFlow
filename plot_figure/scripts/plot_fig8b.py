# This file can draw the subfigures for Figure 8(b) in our paper.
# Usage: 
# 1. Change the raw values accordingly (existing numbers are our test results)
# 2. python3 draw_fig8b.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fontsizeValue=32

label = [str(i) for i in range(1,11)]
label.append("Philly")

EDF=[0.182, 0.228, 0.095, 0.007, 0.375, 0.105, 0.046, 0.252, 0.800, 0.769, 0.01]
Gandiva = [0.246, 0.286, 0.314, 0.228, 0.228, 0.381, 0.351, 0.325, 0.329, 0.354, 0.15]
Tiresias = [0.478, 0.470, 0.401, 0.120, 0.420, 0.466, 0.346, 0.373, 0.468, 0.473, 0.32]
Themis = [0.464, 0.431, 0.389, 0.432, 0.430, 0.425, 0.437, 0.423, 0.456, 0.431, 0.33]
Chronus = [0.474, 0.445, 0.426, 0.450, 0.453, 0.432, 0.460, 0.447, 0.466, 0.454, 0.324]
ElasticFlow=[0.788, 0.822, 0.455, 0.631, 0.832, 0.607, 0.665, 0.896, 0.938, 0.981, 0.87]

x = np.arange(len(EDF))  # the label locations

total_width, n = 0.9, 6
width = total_width / n
x = x - (total_width - width) / 2

fig, (ax1) = plt.subplots(1, 1, sharex=True)

plt.figure(figsize=(30, 7))
plt.grid(True, color="#D3D3D3",zorder=0)
l1 = plt.bar(x-2.5*width, EDF, width, color='white',edgecolor='k',hatch='/',zorder=100)
l2 = plt.bar(x-1.5*width, Gandiva, width, color='white',edgecolor='blue', hatch='\\',zorder=100)
l3 = plt.bar(x-0.5*width, Tiresias, width,color='white',edgecolor='green', hatch='+',zorder=100)
l4 = plt.bar(x+0.5*width, Themis, width,color='white',edgecolor='orange', hatch='*',zorder=100)
l5 = plt.bar(x+1.5*width, Chronus, width,color='white',edgecolor='k', hatch='x',zorder=100)
l6 = plt.bar(x+2.5*width, ElasticFlow, width,zorder=100)

plt.xticks(x, label, fontsize=fontsizeValue)

plt.xlabel("Trace ID", fontsize=fontsizeValue)
plt.ylabel('Deadline Satisfactory Ratio', fontsize=fontsizeValue)

plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
#plt.tick_params(bottom=False, top=False, left=False, right=False)

d = .5  # proportion of vertical to horizontal extent o

plt.legend([l1, l2, l3, l4, l5, l6], [
  'EDF', 'Gandiva', 'Tiresias', 'Themis', 'Chronus', 'ElasticFlow'], fontsize=fontsizeValue, 
  ncol=1,loc=3,bbox_to_anchor=(1,0),borderaxespad=0, frameon=False) 
plt.savefig("fig8b.pdf")

