# Pollux Testbed Experiments
This directory contains the code used for the testbed experiments 
of Pollux baseline in Figure6(a) of the paper. The code is adapted from
[Pollux (OSDI'21) artifact](https://github.com/petuum/adaptdl/tree/osdi21-artifact).
The key files are:

- models/ contains the implementations of each evaluated model listed in Table 1.
- run_workload.py submits jobs according to a workload trace.
- run_monitor.py monitors and logs the cluster state during each cluster scheduling experiment.

## Getting Started
1. Hardware requirements
The testbed experiments requires 4 Azure Standard_ND96asr_A100 nodes, each with 8 A100 GPUs, 96 CPU cores, 900 GB RAM, and eight NVIDIA Mellanox HDR InfiniBand HCAs.
xxx G storage on the `/mnt` directory and xxx nvme is required for dataset storage.
Running the testbed experiment of Pollux takes about one day.
2. Dataset
xxx container image:

The datasets are included in the `/mnt/data`/ directory.

3. Kubernetes configuration

## Reproducing the simulation result in Figure 6(a)