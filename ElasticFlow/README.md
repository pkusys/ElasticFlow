# ElasticFlow Experiments

## Contents
- `ElasticFlow/` contains code for simulation and is adapted from Tiresias.
	- `elastic-training-executor/` contains testbed training code for testbed experiments. It is not needed in simulation experiments.
	- `scheduler/` contains
		- `cluster_spec/` contains configuration files for cluster, e.g., the number of nodes, the number of GPU per node.
		- `runtime/` contains the gRPC source code for communication between the scheduler, master, worker, and trainer. 

calc.py computes metrics, e.g., avg. JCT, Makespan, and 99th JCT.
cluster.py, switch.py, and node.py contain implementations of the cluster.
jobs.py and model.py contain information of the jobs.
flags.py contains the argument definition method.
log.py and utils.py contain auxiliary functions.
matching.py contains the implementation of the matching algorithm for Muri.
run_sim.py contains the implementation of different scheduling policies.

## Environment

## Reproduce simulation results

## Reproduce testbed results