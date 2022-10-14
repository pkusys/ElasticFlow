#!/bin/bash
# Figure 9
echo "trace generation"
python3 ../trace_generator/generate_trace.py --num_jobs=1000 --lam=1500 --output_file=../traces_for_ElasticFlow/1000job_1500lam.csv

schedule=("edf" "edf-accessctrl" "ef" "ef-accessctrl" )
setups=("n8g8" "n16g8" "n32g8")
placement="elastic"

echo "running..."

for setup in ${setups[@]};do
    cluster_spec="cluster_specs/${setup}.csv"
    job_file="../traces_for_ElasticFlow/1000job_1500lam.csv"
    log_folder="../../plot_figure/logs/figure9"
    mkdir ${log_folder}
    for s in ${schedule[@]};do
        log_name="${log_folder}/${s}"
        mkdir $log_name
        python3 scheduler.py --cluster_spec=${cluster_spec} --print --scheme=${placement} --trace_file=${job_file}  --schedule=${s} --log_path=${log_name} --simulation=True --scheduling_slot=240 --gpu_type=A100&
    done
done