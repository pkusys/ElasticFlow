#!/bin/bash
# Figure 11
echo "trace generation"
best_effort_percentages=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
for best_effort_percentage in ${best_effort_percentages[@]};do
    python3 ../trace_generator/generate_trace.py --num_jobs=200 --lam=900 --output_file=../traces_for_ElasticFlow/200jobs_900lam_${best_effort_percentage}.csv --best_effort_percentage=${best_effort_percentage}
done

schedule=("edf" "gandiva" "dlas-gpu" "themis" "ef-accessctrl" )
setups=("n8g8" )

echo "running..."

for setup in ${setups[@]};do
    cluster_spec="cluster_specs/${setup}.csv"
    for best_effort_percentage in ${best_effort_percentages[@]};do
        job_file="../traces_for_ElasticFlow/200jobs_900lam_${best_effort_percentage}.csv"
        log_folder="../../plot_figure/logs/figure11"
        mkdir ${log_folder}
        for s in ${schedule[@]};do
            if [ $s = "gandiva" ]; then
                placement="gandiva"
            else
                placement="elastic"
            fi
            log_name="${log_folder}/${s}_${best_effort_percentage}"
            mkdir $log_name
            python3 scheduler.py --cluster_spec=${cluster_spec} --print --scheme=${placement} --trace_file=${job_file}  --schedule=${s} --log_path=${log_name} --simulation=True --scheduling_slot=240 --gpu_type=A100&
        done
    done
done

cd ../chronus-scheduler/utils
for best_effort_percentage in ${best_effort_percentages[@]};do
    # get trace and namelist
    job_file="../../traces_for_ElasticFlow/200jobs_900lam_${best_effort_percentage}.csv"
    chronus_job_file="../../traces_for_chronus/200jobs_900lam_${best_effort_percentage}.csv"
    chronus_namelist_file="../../traces_for_chronus/200jobs_900lam_${best_effort_percentage}.lst"
    python3 convert_ef_trace_to_chronus.py -t ${job_file} -o ${chronus_job_file}
    python3 get_name_list.py -t ${chronus_job_file} -o ${chronus_namelist_file}
    cd ..
    chronus_job_file="../traces_for_chronus/200jobs_900lam_${best_effort_percentage}.csv"
    chronus_namelist_file="../traces_for_chronus/200jobs_900lam_${best_effort_percentage}.lst"
    save_log_dir="../../plot_figure/logs/figure11/chronus_${best_effort_percentage}"
    python3 main.py --schedule=time-aware-with-lease --trace=${chronus_job_file} --save_log_dir=${save_log_dir} --ident=chronus --aggressive=True --mip_objective=adaptive --placement=local_search --profile=True --check_time_interval=240 --disable_turn_off=True --num_node_p_switch=8 --lease_term_interval=240 --name_list=${chronus_namelist_file} --simulation=True --gpu_type=A100 --num_gpu_p_node=8&
    cd utils
done
cd ../../scheduler