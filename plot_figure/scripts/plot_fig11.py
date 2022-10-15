import matplotlib.pyplot as plt
import csv

x = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"]
edf=[0.09,0.1,0.15,0.15,0.23,0.28,0.44,0.79,0.95]
gandiva=[0.25,0.24,0.2,0.18,0.19,0.17,0.18,0.18,0.16]
tiresias=[0.46,0.46,0.44,0.42,0.43,0.41,0.44,0.5,0.53]
themis=[0.41,0.4,0.37,0.34,0.36,0.34,0.35,0.42,0.37]
chronus=[0.42,0.42,0.4,0.37,0.4,0.39,0.42,0.5,0.53]
elasticflow=[0.74,0.67,0.67,0.69,0.68,0.76,0.82,0.92,0.95]

fontsizeValue=22



plt.figure(figsize=(8, 6))
#plt.xlim(0, 48 * scale)
#plt.ylim(0, 140)
plt.grid(True, color="#D3D3D3")
plt.plot(x, edf, label='EDF',marker='8', linewidth=3)
plt.plot(x, gandiva, ":", label='Gandiva',marker='8', linewidth=3)
plt.plot(x, tiresias, "--", label='Tiresias',marker='8', linewidth=3)
plt.plot(x, themis, "-.", label='Themis',marker='8', linewidth=3)
plt.plot(x, chronus, "--", label='Chronus',marker='8', linewidth=3)
plt.plot(x, elasticflow, "-.", label='ElasticFlow',marker='8', linewidth=3)

plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
plt.tick_params(bottom=False, top=False, left=False, right=False)
plt.xlabel("Percentage of Best Effort Jobs", fontsize=fontsizeValue)
plt.ylabel("Deadline Satisfactory Ratio", fontsize=fontsizeValue)
plt.legend(fontsize=fontsizeValue,ncol=1,loc=3,bbox_to_anchor=(0,1),borderaxespad=0)
plt.savefig("fig11a.pdf")

x = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"]
edf=[0.05,0.05,0.05,0.05,0.07,0.07,0.08,0.08,0.1]
gandiva=[1,1,1,1,1,1,1,1,1]
tiresias=[1.44,1.39,1.33,1.31,1.26,1.27,1.28,1.3,1.45]
themis=[1.37,1.36,1.34,1.32,1.26,1.27,1.29,1.29,1.46]
chronus=[1.5,1.45,1.4,1.38,1.29,1.29,1.28,1.29,1.44]
elasticflow=[1.6,1.37,1.18,1.11,1.27,1.24,1.19,1.16,1.31]

fontsizeValue=22



plt.figure(figsize=(8,6))
#plt.xlim(0, 48 * scale)
#plt.ylim(0, 140)
plt.grid(True, color="#D3D3D3")
plt.plot(x, edf, label='EDF',marker='8', linewidth=3)
plt.plot(x, gandiva, ":", label='Gandiva',marker='8', linewidth=3)
plt.plot(x, tiresias, "--", label='Tiresias',marker='8', linewidth=3)
plt.plot(x, themis, "-.", label='Themis',marker='8', linewidth=3)
plt.plot(x, chronus, "--", label='Chronus',marker='8', linewidth=3)
plt.plot(x, elasticflow, "-.", label='ElasticFlow',marker='8', linewidth=3)

plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
plt.tick_params(bottom=False, top=False, left=False, right=False)
plt.xlabel("Percentage of Best Effort Jobs", fontsize=fontsizeValue)
plt.ylabel("JCT improvement", fontsize=fontsizeValue)
plt.legend(fontsize=22,ncol=2,loc="upper center",bbox_to_anchor=(1,0),borderaxespad=0)
plt.savefig("fig11b.pdf")
