# ElasticFlow-artifact

We provide the artifact for the ASPLOS 2023 paper "ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning", including:

- The main implementation of ElasticFlow.
- Cluster simulation scripts (Sec 6.3 \& 6.4 \& 6.5).
- Testbed experiment scripts (Sec 6.2 \& 6.6).
- Figure plotting scripts.

## Simulation Experiments

### General Simulation Experiments

Please see `ElasticFlow/README.md` for more details.

### Chronus simulation

Please see `chronus/README.md` for more details.

### Pollux simulation

Please see `pollux/pollux_simulator/README.md` for more details.

## Testbed Experiments
Note: Due to the execution scripts of testbed experiments are highly related to intracompany platform, we only demonstrate the functionality and provide the reproduction steps on the hardware devices we use. Please adjust to your platform if you would like to execute the testbed experiment.

The testbed experiments require 16 nodes, each with 8 A100 GPUs, 96 CPU cores, 900 GB RAM, and eight NVIDIA Mellanox HDR InfiniBand HCAs. 
You may use the Azure Standard_ND96asr_A100 VMs for reproduction.

### General Testbed Experiments
Please see `ElasticFlow/README.md` for more details.

### Chronus Testbed Experiments
Please see `chronus/README.md` for more details.

### Pollux Testbed Experiments
As the Pollux baseline is implemented on k8s, we do not interage Pollux in the ElasticFlow system for comparison. We use the open-sourced artifact from the [Pollux repo](https://github.com/petuum/adaptdl/tree/osdi21-artifact) for testbed experiments. 

Please see `pollux/pollux_testbed/README.md` for more details.

## Plotting Figures
Please refer to `<repo>/plot_figure/README.md`
