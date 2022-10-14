#!/bin/bash
# Figure 10
echo "trace generation"
python3 ../trace_generator/generate_trace.py --num_jobs=100 --lam=1500 --min_ddl=1.5 --seed=1 --output_file=../traces_for_ElasticFlow/100jobs_1500lam.csv

schedule=("edf" "gandiva" "dlas-gpu" "themis" "ef-accessctrl" )
setups=("n16g8" )

echo "running..."

for setup in ${setups[@]};do
    cluster_spec="cluster_specs/${setup}.csv"
    job_file="../traces_for_ElasticFlow/100jobs_1500lam.csv"
    #job_file="/Users/gudiandian/Desktop/github/ElasticFlow-artifact/ElasticFlow/trace_generator/1000job_1500lam.csv"
    log_folder="../../plot_figure/logs/figure10"
    mkdir ${log_folder}
    for s in ${schedule[@]};do
        if [ $s = "gandiva" ]; then
            placement="gandiva"
        else
            placement="elastic"
        fi
        log_name="${log_folder}/${s}"
        mkdir $log_name
        python3 scheduler.py --cluster_spec=${cluster_spec} --print --scheme=${placement} --trace_file=${job_file}  --schedule=${s} --log_path=${log_name} --simulation=True --scheduling_slot=240 --gpu_type=A100&
    done
done