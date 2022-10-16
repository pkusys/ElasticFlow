# ElasticFlow Experiments

## Contents
- `ElasticFlow/` contains code for simulation and is adapted from [Tiresias](https://github.com/SymbioticLab/Tiresias).
	- `chronus-scheduler/` contains the implementation of Chronus scheduling algorithm provided by [this repo](https://github.com/S-Lab-System-Group/ChronusArtifact).
	- `elastic-training-executor/` contains testbed training code for testbed experiments. It is not needed in simulation experiments.
	- `scheduler/` contains the implementation of ElasticFlow scheduling algorithm and some baseline algorithms.
		- `cluster_spec/` contains configuration files for cluster, e.g., the number of nodes, the number of GPU per node.
		- `overhead_measurement_traces/` contains the traces for measuring the overheads of the implementation of ElasticFlow. It is not needed in simulation.
		- `runtime/` contains the gRPC source code for communication between the scheduler, master, worker, and trainer. 
		- `throughputs_A100/` contains the profiled throughputs for DL models on A100 GPUs.
		- `throughputs_T4/` contains the profiled throughputs for DL models on T4 GPUs (provided by [adaptdl](https://github.com/petuum/adaptdl/tree/osdi21-artifact)).
		- `trace_generator/` contains the script for generating job traces.
	- `prepare_container.sh` is the script for preparing the docker container for testbed experiments. It is not needed in simulation.
	- `topo.xml` is needed for testbed experiments. It is not needed in simulation.
	

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
cd chronus-scheduler
make
cd ../scheduler
make
```
Note that the Chronus baseline relies on the Gurobi optimizer. Please refer to the [official website](https://www.gurobi.com) for downloading, environment configuration, and installing.

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

2. Plot the figures.

Please refer to `<repo>/plot_figure/README.md`


## Reproduce testbed results

Note: Due to the execution scripts of testbed experiments are highly related to intracompany platform, we only demonstrate the functionality and provide the reproduction steps on the hardware devices we use. Please adjust to your platform if you would like to execute the testbed experiment.

### Hardware

The testbed experiments require up to 16 VMs, each with 8 A100 GPUs, 96 CPU cores, 900 GB RAM, and eight NVIDIA Mellanox HDR InfiniBand HCAs. 
NVMe is required for dataset and DL model checkpoint storage to speed up the I/O process. 
At least 160G NVMe storage is needed on each node for the dataset and model checkpoints.
The provided scripts can be executed on Azure Standard_ND96asr_A100 VMs. Please adjust to your platform if you would like to execute the testbed experiment.

### Environment

1. Get the docker container.

Run `bash prepare_container.sh`.

Note that the docker size is 21.5GB, make sure that there is enough disk space. There should be a `/mnt` directory. NVMe disk is required for this script.

2. Get the dataset.

The datasets and model configuration files can be downloaded from [this link](https://drive.google.com/file/d/1gxFg842sYH6JNqCkKtYf7DfkFAunkh_n/view?usp=sharing). 

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


### Reproduction Steps
First, enter the scheduler directory:
```Bash
cd /workspace/ElasticFlow-artifact/ElasticFlow/scheduler
```
Then, run the experiments in container.
1. Figure 6(a):

On the master node, run the master server:
```Bash
python python master.py -p 6888 -n 4
```
On each of four worker nodes (the master node can be included in the worker nodes), run the worker:
```Bash
python worker.py -i <master_ip> -P 6888 -p 9000 -n 8 -A <master_ip> -g 6889 -w 32 -r ../elastic-training-executor/ -x /opt/conda/bin/python3.8 --dynamic_requests=True --scheduler_port=6890 --scheduler_addr=<master_ip>
```
Then, wait for a few seconds, if you see messages such as `trainer 0 idles ... ...`, it means the trainer processes have been successfully started. Then, you can start the scheduler on the master node:
```Bash
python scheduler.py --cluster_spec=cluster_specs/n4g8.csv --print --scheme=<placement_algo> --trace_file=../traces_for_ElasticFlow/25job_endtoend_trace.csv  --schedule=<scheduling_algo> --log_path=../../plot_figure/logs/figure6a/<scheduling_algo> --simulation=False --scheduling_slot=240 --restart_threshold=70 --gpu_type=A100
```
For ElasticFlow algorithm, `<placement_algo>` should be `elastic` and `<scheduling_algo>` should be `ef-accessctrl`
For EDF algorithm, `<placement_algo>` should be `elastic` and `<scheduling_algo>` should be `edf`
For Gandiva algorithm, `<placement_algo>` should be `gandiva` and `<scheduling_algo>` should be `gandiva`
For Tiresias algorithm, `<placement_algo>` should be `elastic` and `<scheduling_algo>` should be `dlas-gpu`
For Themis algorithm, `<placement_algo>` should be `elastic` and `<scheduling_algo>` should be `themis`


For Chronus scheduler, you should run the scheduler in the `chronus-scheduler` directory:
```Bash
cd ../chronus-scheduler/utils
# convert trace
python convert_ef_trace_to_chronus.py -t ../../traces_for_ElasticFlow/25job_endtoend_trace.csv -o ../../traces_for_chronus/25job_endtoend_trace.csv
python get_name_list.py -t ../../traces_for_chronus/25job_endtoend_trace.csv -o ../../traces_for_chronus/25job_endtoend_trace.lst
# run scheduler
cd ..
python main.py --schedule=time-aware-with-lease --trace=../traces_for_chronus/25job_endtoend_trace.csv --save_log_dir=../../plot_figure/logs/figure6a/chronus --ident=chronus --aggressive=True   --mip_objective=adaptive --placement=local_search --profile=True --check_time_interval=240 --disable_turn_off=True --num_node_p_switch=4 --lease_term_interval=240 --name_list=../traces_for_chronus/25job_endtoend_trace.lst --num_gpu_p_node=8 --gpu_type=A100
```

It takes a few hours for each setting.

2. Figure 6(b) & Figure 7: 

On the master node, run the master server:
```Bash
python python master.py -p 6888 -n 16
```
On each of four worker nodes (the master node can be included in the worker nodes), run the worker:
```Bash
python worker.py -i <master_ip> -P 6888 -p 9000 -n 8 -A <master_ip> -g 6889 -w 128 -r ../elastic-training-executor/ -x /opt/conda/bin/python3.8 --dynamic_requests=True --scheduler_port=6890 --scheduler_addr=<master_ip>
```
Then, wait for a few seconds, if you see messages such as `trainer 0 idles ... ...`, it means the trainer processes have been successfully started. Then, you can start the scheduler on the master node:
```Bash
python scheduler.py --cluster_spec=cluster_specs/n16g8.csv --print --scheme=<placement_algo> --trace_file=../traces_for_ElasticFlow/195job_endtoend_trace.csv  --schedule=<scheduling_algo> --log_path=../../plot_figure/logs/figure6b/<scheduling_algo> --simulation=False --scheduling_slot=240 --restart_threshold=70 --gpu_type=A100
```

For Chronus scheduler, you should run the scheduler in the `chronus-scheduler` directory:
```Bash
cd ../chronus-scheduler/utils
# convert trace
python convert_ef_trace_to_chronus.py -t ../../traces_for_ElasticFlow/195job_endtoend_trace.csv -o ../../traces_for_chronus/195job_endtoend_trace.csv
python get_name_list.py -t ../../traces_for_chronus/195job_endtoend_trace.csv -o ../../traces_for_chronus/195job_endtoend_trace.lst
# run scheduler
cd ..
python main.py --schedule=time-aware-with-lease --trace=../traces_for_chronus/195job_endtoend_trace.csv --save_log_dir=../../plot_figure/logs/figureba/chronus --ident=chronus --aggressive=True   --mip_objective=adaptive --placement=local_search --profile=True --check_time_interval=240 --disable_turn_off=True --num_node_p_switch=16 --lease_term_interval=240 --name_list=../traces_for_chronus/195job_endtoend_trace.lst --num_gpu_p_node=8 --gpu_type=A100
```

It takes a few hours for each setting.

3. Figure 12(a): 

On the master node, run the master server:
```Bash
python master.py -p 6888 -t overhead_measurement_traces/<model>_profile.txt -d 1
```
On each of eight worker nodes (the master node can be included in the worker nodes), run the worker:
```Bash
python worker.py -i <master_ip> -P 6888 -p 9000 -n 8 -A <master_ip> -g 6889 -w 64 -r ../elastic-training-executor/ -x /opt/conda/bin/python3.8 
```
The time for running the whole trace is the profiling overhead for each model.

4. Figure 12(b): 

On the master node, run the master server:
```Bash
python master.py -p 6888 -t overhead_measurement_traces/<model>_<case>_overhead.txt -d 1
```
On each of two worker nodes (the master node can be included in the worker nodes), run the worker:
```Bash
python worker.py -i <master_ip> -P 6888 -p 9000 -n 8 -A <master_ip> -g 6889 -w 16 -r ../elastic-training-executor/ -x /opt/conda/bin/python3.8 
```
The time for running the whole trace is the scaling/migration overhead for each case.


### Plotting figures
For plotting figures, please refer to `<repo>/plot_figure/README.md`
