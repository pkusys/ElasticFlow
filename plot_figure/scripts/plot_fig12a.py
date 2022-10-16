# This file can draw the subfigures for Figure 12a in our paper.
# Usage: 
# 1. Change the raw values accordingly (existing numbers are our test results)
# 2. python3 draw_fig12a.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fontsizeValue=22

resnet50=[549.759]
vgg16 = [550.011]
inceptionv3 = [366.84]
gpt2 = [336.054]
bert = [305.462]
deepspeech2=[337.726]

x = np.arange(len(resnet50))  # the label locations

total_width, n = 0.3, 1
width = total_width / n
x = x - (total_width - width) / 2

fig, (ax1) = plt.subplots(1, 1, sharex=True)

plt.figure(figsize=(7, 7))
plt.grid(True, color="#D3D3D3",zorder=0)
l1 = plt.bar(x-2.5*width, resnet50, width, color='white',edgecolor='k',hatch='/',zorder=100)
l2 = plt.bar(x-1.5*width, vgg16, width, color='white',edgecolor='blue', hatch='\\',zorder=100)
l3 = plt.bar(x-0.5*width, inceptionv3, width,color='white',edgecolor='green', hatch='+',zorder=100)
l4 = plt.bar(x+0.5*width, gpt2, width,color='white',edgecolor='orange', hatch='*',zorder=100)
l5 = plt.bar(x+1.5*width, bert, width,color='white',edgecolor='k', hatch='x',zorder=100)
l6 = plt.bar(x+2.5*width, deepspeech2, width,zorder=100, edgecolor='purple', hatch='-', color='white')

#plt.xticks(x, label, fontsize=fontsizeValue)

#plt.xlabel("Trace ID", fontsize=fontsizeValue)
plt.ylabel('Profiling Overhead (s)', fontsize=fontsizeValue)

plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
#plt.tick_params(bottom=False, top=False, left=False, right=False)

d = .5  # proportion of vertical to horizontal extent o

plt.legend([l1, l2, l3, l4, l5, l6], [
  'ResNet50', 'VGG16', 'InceptionV3', 'GPT2', 'BERT', 'DeepSpeech2'], fontsize=16, 
  ncol=3,loc="upper center", bbox_to_anchor=(0.5,1.15),borderaxespad=0,frameon=False) 
plt.savefig("fig12a.pdf")

