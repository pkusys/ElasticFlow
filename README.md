# ElasticFlow-artifact

We provide the artifact for the ASPLOS 2023 paper "ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning", including:

- The main implementation of \sysname.
- Testbed experiment scripts (Sec 6.2 \& 6.6).
- Cluster simulator (Sec 6.3 \& 6.4 \& 6.5).

## Testbed Experiments

### General Testbed Experiments
Please see `ElasticFlow/README.md` for more details.

### Testbed Experiments of Pollux
As the Pollux baseline is implemented on k8s, we do not interage Pollux in the ElasticFlow system for comparison. We use the open-sourced artifact from the [Pollux repo](https://github.com/petuum/adaptdl/tree/osdi21-artifact) for testbed experiments. 

Please see `pollux/pollux_testbed/README.md` for more details.


## Simulation Experiments

### General Simulation Experiments
Please see `ElasticFlow/README.md` for more details.

### Pollux simulation

`pollux/pollux_simulator` contains Instructions for reproducing the experiment 
of Pollux shown in Figure 12(a).

Please see `pollux/pollux_simulator/README.md` for more details.
