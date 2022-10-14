#!/bin/bash
# Figure 8(a)
schedule=("edf" "gandiva" "dlas-gpu" "themis" "ef-accessctrl" )
#placement=("elastic" "gandiva" "elastic" "elastic" "elastic")
setups=("n16g4")
jobs=("elasticVpollux_e")

echo "running..."

for setup in ${setups[@]};do
    cluster_spec="cluster_specs/${setup}.csv"
    for job in ${jobs[@]};do
        job_file="../traces_for_ElasticFlow/${job}.csv"
        #job_file="test.csv"
        log_folder="../../plot_figure/logs/figure8a"
        mkdir ${log_folder}
        for s in ${schedule[@]};do
            if [ $s = "gandiva" ]; then
                placement="gandiva"
            else
                placement="elastic"
            fi
            log_name="${log_folder}/${s}"
            mkdir $log_name
            python3 scheduler.py --cluster_spec=${cluster_spec} --print --scheme=${placement} --trace_file=${job_file}  --schedule=${s} --log_path=${log_name} --simulation=True --scheduling_slot=60 --gpu_type=T4&
        done
    done
done