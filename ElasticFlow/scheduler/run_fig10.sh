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

cd ../chronus-scheduler/utils
# get trace and namelist
python3 convert_ef_trace_to_chronus.py -t ../../traces_for_ElasticFlow/100jobs_1500lam.csv -o ../../traces_for_chronus/100jobs_1500lam.csv
python3 get_name_list.py -t ../../traces_for_chronus/100jobs_1500lam.csv -o ../../traces_for_chronus/100jobs_1500lam.lst
cd ..
python3 main.py --schedule=time-aware-with-lease --trace=../traces_for_chronus/100jobs_1500lam.csv --save_log_dir=../../plot_figure/logs/figure10/chronus --ident=chronus --aggressive=True --mip_objective=adaptive --placement=local_search --profile=True --check_time_interval=240 --disable_turn_off=True --num_node_p_switch=16 --lease_term_interval=240 --name_list=../traces_for_chronus/100jobs_1500lam.lst --simulation=True --gpu_type=A100 --num_gpu_p_node=8
cd ../scheduler