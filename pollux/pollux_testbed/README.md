# Pollux Testbed Experiments

Note: Due to the execution scripts of testbed experiments are highly related to intracompany platform, we only demonstrate the functionality and provide the reproduction steps on the hardware devices we use. Please adjust to your platform if you would like to execute the testbed experiment.

This directory contains the code used for the testbed experiments 
of Pollux baseline in Figure6(a) of the paper. The code is adapted from
[Pollux (OSDI'21) artifact](https://github.com/petuum/adaptdl/tree/osdi21-artifact).
The key files are:

- models/ contains the implementations of each evaluated model listed in Table 1.
- run_workload.py submits jobs according to a workload trace.
- run_monitor.py monitors and logs the cluster state during each cluster scheduling experiment.

## Getting Started
1. Hardware requirements

The testbed experiments requires at least four Azure Standard_ND96asr_A100 nodes, each with 8 A100 GPUs, 96 CPU cores, 900 GB RAM, and eight NVIDIA Mellanox HDR InfiniBand HCAs. 
NVMe is required for dataset and DL model checkpoint storage to speed up the I/O process. 
At least **160G NVMe** storage is needed on each node for the dataset and model checkpoints.

Running the testbed experiment of Pollux takes about one day, and more time is needed for environment preparation.

NFS is needed by Pollux according to the [public Pollux benchmark](https://github.com/petuum/adaptdl/tree/osdi21-artifact/benchmark).
You need to add this line to `/etc/exports` on the master node:
```Bash
/mnt/data1 *(rw,sync,no_subtree_check,no_root_squash)
```
Then, the other configurations needed by NFS will be automatically cnfigured by the scripts in the following steps.

Also, please modify the `prefix` in `environmrnt.yaml` to the dir where your conda is.

2. Dataset

The datasets include:
 - [ImageNet](https://www.image-net.org)
 - [CoLA](https://nyu-mll.github.io/CoLA/)
 - [aclImdb](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
 - [LibriSpeech](https://pytorch.org/audio/main/generated/torchaudio.datasets.LIBRISPEECH.html)

The model configuration files include [merges.txt](https://huggingface.co/gpt2/raw/main/merges.txt) and [vocab.json](https://huggingface.co/gpt2/raw/main/vocab.json) for GPT2 model.

You can download the datasets following the official guide of each dataset.
We have packed a smaller dataset (including the model configuration files) and you can download it from [this link](https://drive.google.com/file/d/1gxFg842sYH6JNqCkKtYf7DfkFAunkh_n/view?usp=sharing). 

The datasets need to be placed in the `/mnt/data1/` directory.
The `/mnt/data1/` directory should be like:
```
/mnt/
| - data1/
|	| - imagenet/
|	| - LibriSpeech/
|	| - aclImdb/
|	| - bert/
|	| - gpt2/
```
If there is data corruption, please download the datasets from the official website.

3. Kubernetes configuration

Configuring Kubernets requires the disk usgae of the `/` directory is under 80%. You make check the disk usage with the `df -h` command. 

First, you need to log in docker on each node:
```Bash
$ docker login -u <docker_username> -p <docker_password>
```
Modify `/etc/docker/daemon.json` to set up `nvidia-container-runtime` according to [this guide](https://github.com/NVIDIA/k8s-device-plugin#quick-start). 

Remember to restart docker service:
```Bash
$ systemctl restart docker
```

Then, you need to install a certain version of k8s to run Pollux. On all of the nodes, run:
```Bash
$ bash scripts/install_k8s.sh
```
Then, install the pollux scheduler and config the cluster.
On the master node, run:
```Bash
$ bash scripts/master_node.sh
```

On the other worker nodes, run:
```Bash
$ bash scripts/worker_node.sh <master_ip>
```
If you only have four nodes, you need to remove the scheduler taint from the master node so that you can use the GPUs in Pollux. Run `kubectl describe node <master_node_name>` to get the taint information and run `kubectl taint nodes --all <taint>-` to remove the taints. 

## Reproducing the Testbed result in Figure 6(a)
To run all jobs, run on the master node:
```Bash
$ python run_workload.py pollux pollux_testbed_trace.csv --repository=<your_docker_username>/pollux
```

To get the scheduling results, run:
```Bash
$ python run_monitor.py pollux_result.json
```

To parse the results and get the final deadline satisfactory ratio, run:
```Bash
$ python parse_result.py --input pollux_result.json  --trace ../ElasticFlow/traces_for_ElasticFlow/25job_endtoend_trace.csv
```
