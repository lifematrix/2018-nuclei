#!/usr/bin/env bash

model_name=unet-3-256
log_root=log/${model_name}


datetime=`date +"%Y%m%d-%H%M%S"`
#datetime=20180403-173913

#if (test $# -ge 1); then
#    echo "Continue interrupted train, from logdir: "$1
#    model_log_dir=$1
#else
#    model_log_dir=${model_log_root}/${datetime}
#    mkdir -p ${model_log_dir}
#fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../aivis:$PYTHONPATH

general_log_file=log/${datetime}_${model_name}.txt
log_dir=${log_root}/${datetime}

task_dir=$(dirname $0)
echo "task_dir: $task_dir"

echo "log file: "${general_log_file}
touch ${general_log_file}
#
#read -r -d '' cmd<<EOF
# python code/train_yaml.py train
#        --net ${task_dir}/net.yaml              \
#        --solver ${task_dir}/train.yaml             \
#        --logdir ${log_dir}
#EOF


read -r -d '' cmd<<EOF
 python -m aivis.tools.dnnapp train
        --net ${task_dir}/net.yaml              \
        --solver ${task_dir}/train.yaml             \
        --logdir ${log_dir}
EOF

function trap_ctrlc()
{
    echo "Ctrl-C is caught ..."
    echo "Log file is: ${general_log_file}"
    echo "The shell will exit. You can view log file to monitor process."
}

trap "trap_ctrlc" 2

if [[ $1 = "--withlog" ]]; then
    echo "withlog"
    exec setsid ${cmd}  1>${general_log_file} 2>&1 &
    tail -f ${general_log_file}
else
    echo ${cmd}
    exec ${cmd}
fi


