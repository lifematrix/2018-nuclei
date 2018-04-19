#!/usr/bin/env bash

model_name=diabetic.dr_resnet
log_root=log/${model_name}


datetime=`date +"%Y%m%d-%H%M%S"`
#if (test $# -ge 1); then
#    echo "Continue interrupted train, from logdir: "$1
#    model_log_dir=$1
#else
#    model_log_dir=${model_log_root}/${datetime}
#    mkdir -p ${model_log_dir}

    # clean log, train from starch
echo "tensorflow log dir: "${log_dir}
echo "clean log dir, to train from scratch"
    #rm -f ${model_log_dir}/*
#fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=`pwd`/code/package:$PYTHONPATH

# general_log_file=log/${datetime}_${model_name}.txt
# tf_log_dir=${tf_log_root}/20171229-180358
# log_dir=${log_root}/${datetime}
log_dir=${log_root}/20180309-104219


echo "log file: "${general_log_file}
touch ${general_log_file}

task_dir=$(dirname $0)
echo "task_dir: $task_dir"

python code/train_yaml.py validate        \
    --net $task_dir/net.yaml              \
    --solver $task_dir/val.yaml             \
    --checkpoint $log_dir/best/model.ckpt-326349

# --lr_policy Interval::[[10,0.0001],[50,0.001],[50,0.0004],[50,0.0002],[50,0.00005],[50,0.00001],[20,0.000005]]
# [[10,0.0001],[100,0.001],[100,0.0001],[50,0.00004],[50,0.00001],[50,0.000001],[30,0.0000002]]


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


