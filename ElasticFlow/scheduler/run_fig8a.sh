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

cd ../chronus-scheduler/utils
# get trace and namelist
python3 convert_ef_trace_to_chronus.py -t ../../traces_for_ElasticFlow/elasticVpollux_e.csv -o ../../traces_for_chronus/elasticVpollux_e.csv
python3 get_name_list.py -t ../../traces_for_chronus/elasticVpollux_e.csv -o ../../traces_for_chronus/elasticVpollux_e.lst
cd ..
python3 main.py --schedule=time-aware-with-lease --trace=../traces_for_chronus/elasticVpollux_e.csv --save_log_dir=../../plot_figure/logs/figure8a/chronus --ident=chronus --aggressive=True --mip_objective=adaptive --placement=local_search --profile=True --check_time_interval=60 --disable_turn_off=True --num_node_p_switch=16 --lease_term_interval=240 --name_list=../traces_for_chronus/elasticVpollux_e.lst --simulation=True --gpu_type=T4 --num_gpu_p_node=4
cd ../scheduler
