# ElasticFlow Experiments

## Contents
- `ElasticFlow/` contains code for simulation and is adapted from Tiresias.
	- `elastic-training-executor/` contains testbed training code for testbed experiments. It is not needed in simulation experiments.
	- `scheduler/` contains the implementation of ElasticFlow scheduling algorithm and some baseline algorithms.
		- `cluster_spec/` contains configuration files for cluster, e.g., the number of nodes, the number of GPU per node.
		- `runtime/` contains the gRPC source code for communication between the scheduler, master, worker, and trainer. 
		- `throughputs_A100/` contains the profiled throughputs for DL models on A100 GPUs.
		- `throughputs_T4/`ontains the profiled throughputs for DL models on T4 GPUs (provided by [adaptdl](https://github.com/petuum/adaptdl/tree/osdi21-artifact)).

calc.py computes metrics, e.g., avg. JCT, Makespan, and 99th JCT.
cluster.py, switch.py, and node.py contain implementations of the cluster.
jobs.py and model.py contain information of the jobs.
flags.py contains the argument definition method.
log.py and utils.py contain auxiliary functions.
matching.py contains the implementation of the matching algorithm for Muri.
run_sim.py contains the implementation of different scheduling policies.


## Reproduce simulation results

### Environment
```Bash
# create conda env
conda create -n elasticflow python=3.8
conda activate elasticflow
# install dependencies
cd ElasticFlow
python -m pip install --upgrade pip
pip install -r requirements.txt
# make gRPC
cd scheduler
make
```

### Reproduction Steps

1. Run the experiments.
```Bash
cd scheduler
```
- Figure 8(a): `source run_fig8a.sh`. This takes about 30 minutes.
- Figure 8(b): `source run_fig8b.sh`. This might take a few days to finish the simulation of all of the traces!
- Figure 9: `bash run_fig9.sh`. This takes about 7 minutes. Note that the results might be a bit different because the trace used is randomly generated. 
- Figure 10: `source run_fig10.sh`. This takes about 2 minutes.  Note that the results might be a bit different because the trace used is randomly generated.
- Figure 11: `source run_fig11.sh`. This takes about 10 minutes. Note that the results might be a bit different because the trace used is randomly generated.

All the logs and results will saved be in the `<repo>/plot_figure/logs/` directory.

2. Plot the figures
Please refer to `<repo>/plot_figures/README.md`


## Reproduce testbed results

Note: Due to the execution scripts of testbed experiments are highly related to intracompany platform, we only demonstrate the functionality and provide the reproduction steps on the hardware devices we use. Please adjust to your platform if you would like to execute the testbed experiment.

### Hardware

The testbed experiments require up to 16 VMs, each with 8 A100 GPUs, 96 CPU cores, 900 GB RAM, and eight NVIDIA Mellanox HDR InfiniBand HCAs. 
NVMe is required for dataset and DL model checkpoint storage to speed up the I/O process. 
At least 160G NVMe storage is needed on each node for the dataset and model checkpoints.
The provided scripts can be executed on Azure Standard_ND96asr_A100 VMs. Please adjust to your platform if you would like to execute the testbed experiment.

### Environment

Run `bash prepare_container.sh`.

### Reproduction Steps
1. Run the experiments.
```Bash
cd scheduler
```
- Figure 7(a): 
- Figure 7(b): 
- Figure 12(a): 
- Figure 12(b): 

2. Plot the figures
Please refer to `<repo>/plot_figures/README.md`
