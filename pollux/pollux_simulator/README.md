# Pollux Cluster Simulator

This directory contains the code used for the simulator-based experiments 
of Pollux baseline in Figure8(a) of the paper. The code is adapted from
[Pollux (OSDI'21) artifact](https://github.com/petuum/adaptdl/tree/osdi21-artifact).

The contents are summarized as follows:

- **traces/** contains the system throughput and statistical efficiency
  measurements collected according to the "Simulator construction" paragraph.
- **applications.py** contains the code that parses the collected traces of
  each training job type, as well as helpers to interpolate to GPU placement
  and batch size configurations which were not directly measured.
- **pollux.py**, **speedup.py**, **goodput.py**, and **utils.py** contain the
  implementation of the Pollux scheduling policy and Pollux agent.

## Getting Started

We suggest using a fresh virtualenv or conda environment to install the
dependencies:

```
$ conda create -n pollux python=3.8
$ conda activate pollux
$ python3 -m pip install -r requirements.txt
```

## Reproducing the simulation result in Figure 8(a)

Use the following command to run the Pollux policy on the 195-job traces, which is the
one used for Figure 8(a). The simulator should print out logs for every 60s
of simulated time, along with some information on the active and running jobs.
At the end, it will print out the average completion time as well as the DDL 
satisfactory ratio of the jobs in the workload. 

```
$ python3 simulator.py --policy pollux elasticVpollux_p.csv >fig8_a_pollux.out
...
...
...
Average JCT: 71114.34871794871
DDL satisfactory ratio: 0.5025641025641026
```
The simulation takes about a day.

## Plotting figures
Please refer to `<repo>/plot_figure/.README.md`

