#!/bin/bash
# Figure 8(a)
schedule=("edf" "gandiva" "dlas-gpu" "themis" "ef-accessctrl" )
#placement=("elastic" "gandiva" "elastic" "elastic" "elastic")
jobs=("trace1" "trace2" "trace3" "trace4" "trace5" "trace6" "trace7" "trace8" "trace9" "trace10" "philly_103959")

echo "running..."

for job in ${jobs[@]};do
    if [ $job = "trace1" ] || [ $job = "trace2" ] || [ $job = "trace3" ] || [ $job = "trace5" ] || [ $job = "trace6" ]; then
        setup="n8g8"
    elif [ $job = "trace4" ] || [ $job = "trace7" ] || [ $job = "trace8" ] || [ $job = "trace9" ]; then
        setup="n16g8"
    else
        setup="n4g8"
    fi
    cluster_spec="cluster_specs/${setup}.csv"
    job_file="../traces_for_ElasticFlow/${job}.csv"
    #job_file="test.csv"
    log_folder="../../plot_figure/logs/figure8b"
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