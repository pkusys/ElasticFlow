# ElasticFlow-artifact

We provide the artifact for the ASPLOS 2023 paper "ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning", including:

- The main implementation of ElasticFlow.
- Cluster simulator (Sec 6.3 \& 6.4 \& 6.5).
- Testbed experiment scripts (Sec 6.2 \& 6.6).

## Simulation Experiments

### General Simulation Experiments

Please see `ElasticFlow/README.md` for more details.

### Pollux simulation

Please see `pollux/pollux_simulator/README.md` for more details.

## Testbed Experiments
The testbed experiments of ElasticFlow were conducted in an internal cluster of our corperation. The experiments require 16 nodes, each with 8 A100 GPUs, 96 CPU cores, 900 GB RAM, and eight NVIDIA Mellanox HDR InfiniBand HCAs. 
You may use the Azure Standard_ND96asr_A100 VMs for reproduction.

### General Testbed Experiments
Please see `ElasticFlow/README.md` for more details.

### Testbed Experiments of Pollux
As the Pollux baseline is implemented on k8s, we do not interage Pollux in the ElasticFlow system for comparison. We use the open-sourced artifact from the [Pollux repo](https://github.com/petuum/adaptdl/tree/osdi21-artifact) for testbed experiments. 

Please see `pollux/pollux_testbed/README.md` for more details.
