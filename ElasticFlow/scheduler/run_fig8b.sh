#!/bin/bash
# Figure 8(a)
schedule=("edf" "gandiva" "dlas-gpu" "themis" "ef-accessctrl" )
#placement=("elastic" "gandiva" "elastic" "elastic" "elastic")
jobs=("cluster_1" "cluster_2" "cluster_3" "cluster_4" "cluster_5" "cluster_6" "cluster_7" "cluster_8" "cluster_9" "cluster_10" "philly_103959")

echo "running..."

for job in ${jobs[@]};do
    if [ $job = "cluster_1" ] || [ $job = "cluster_2" ] || [ $job = "cluster_3" ] || [ $job = "cluster_5" ] || [ $job = "cluster_6" ]; then
        setup="n8g8"
    elif [ $job = "cluster_4" ] || [ $job = "cluster_7" ] || [ $job = "cluster_8" ] || [ $job = "cluster_9" ]; then
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
        log_name="${log_folder}/${s}_${job}"
        mkdir $log_name
        python3 scheduler.py --cluster_spec=${cluster_spec} --print --scheme=${placement} --trace_file=${job_file}  --schedule=${s} --log_path=${log_name} --simulation=True --scheduling_slot=240 --gpu_type=A100&
    done
done

cd ../chronus-scheduler/utils
for job in ${jobs[@]};do
    if [ $job = "cluster_1" ] || [ $job = "cluster_2" ] || [ $job = "cluster_3" ] || [ $job = "cluster_5" ] || [ $job = "cluster_6" ]; then
        num_node=8
    elif [ $job = "cluster_4" ] || [ $job = "cluster_7" ] || [ $job = "cluster_8" ] || [ $job = "cluster_9" ]; then
        num_node=16
    else
        num_node=4
    fi
    # get trace and namelist
    job_file="../../traces_for_ElasticFlow/${job}.csv"
    chronus_job_file="../../traces_for_chronus/${job}.csv"
    chronus_namelist_file="../../traces_for_chronus/${job}.lst"
    python3 convert_ef_trace_to_chronus.py -t ${job_file} -o ${chronus_job_file}
    python3 get_name_list.py -t ${chronus_job_file} -o ${chronus_namelist_file}
    cd ..
    chronus_job_file="../traces_for_chronus/${job}.csv"
    chronus_namelist_file="../traces_for_chronus/${job}.lst"
    save_log_dir="../../plot_figure/logs/figure8b/chronus_${job}"
    python3 main.py --schedule=time-aware-with-lease --trace=${chronus_job_file} --save_log_dir=${save_log_dir} --ident=chronus --aggressive=True --mip_objective=adaptive --placement=local_search --profile=True --check_time_interval=240 --disable_turn_off=True --num_node_p_switch=${num_node} --lease_term_interval=240 --name_list=${chronus_namelist_file} --simulation=True --gpu_type=A100 --num_gpu_p_node=8&
    cd utils
done
cd ../../scheduler
