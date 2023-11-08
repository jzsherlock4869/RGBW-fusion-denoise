#!/bin/bash

opt=$1
gpu=$2
basename=`basename $opt`
expid=$(echo $basename | awk '{ string=substr($0,1,3); print string; }')

echo "started task, exp ${expid} on GPU no. ${gpu}"
echo $basename

CUDA_VISIBLE_DEVICES=$gpu nohup python -u train_rgbw.py -opt $1 > logs/tr_${expid}_${gpu}.log 2>&1 &
