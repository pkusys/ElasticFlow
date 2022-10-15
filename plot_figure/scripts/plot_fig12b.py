import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fontsizeValue=20

resnet50=[1.884,1.659,1.666,1.787,0.887]
vgg16 = [1.82,1.629,2.407,2.354, 3.209]
inceptionv3 = [1.819,1.099,1.738,1.733, 0.69]
gpt2=[2.344,3.101,3.203,3.198,2.119]
bert = [3.904,2.611,4.239,4.222, 3.191]
deepspeech2 = [3.904,2.611,4.239,4.222, 3.191]
ef = [1.573,1.131,1.757,1.636, 2.429]

x = np.arange(len(resnet50))  # the label locations

total_width, n = 0.7, 6
width = total_width / n
x = x - (total_width - width) / 2

fig, (ax1) = plt.subplots(1, 1, sharex=True)

#plt.figure(figsize=(16, 5))
plt.figure(figsize=(25, 7)) # for legend
plt.grid(True, color="#D3D3D3", zorder=0)
l1 = plt.bar(x-2.5*width, resnet50, width, zorder=100,color='white',edgecolor='k',hatch='/')
l2 = plt.bar(x-1.5*width, vgg16, width, zorder=100,color='white',edgecolor='blue', hatch='\\')
l3 = plt.bar(x-0.5*width, inceptionv3, width, zorder=100,color='white',edgecolor='green', hatch='+')
l4 = plt.bar(x+0.5*width, gpt2, width, zorder=100,edgecolor='orange', hatch='*', color='white')
l5 = plt.bar(x+1.5*width, bert, width, zorder=100,edgecolor='k', hatch='x', color='white')
l6 = plt.bar(x+2.5*width, deepspeech2, width, zorder=100,edgecolor='purple', hatch='-', color='white')
label = ["1GPU->2GPU", "2GPU->1GPU", "8GPU->16GPU", "16GPU->8GPU", "migration(8GPU)"]

plt.xticks(x, label, fontsize=fontsizeValue)

#plt.xlabel("Cluster Size", fontsize=fontsizeValue)
plt.ylabel("Scaling Overhead (s)", fontsize=fontsizeValue)

plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
#plt.tick_params(bottom=False, top=False, left=False, right=False)

d = .5  # proportion of vertical to horizontal extent o

plt.legend([l1, l2, l3, l4, l5, l6], [
  'ResNet50', 'VGG16', 'InceptionV3', 'GPT2', 'BERT', 'DeepSpeech2'],fontsize=fontsizeValue,
  ncol=6,loc="upper center",bbox_to_anchor=(0.5,1),borderaxespad=0,frameon=False)
plt.savefig("fig12b.pdf")

