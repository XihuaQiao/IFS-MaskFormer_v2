#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
num=$2
port=$(python get_free_port.py)
echo ${port}

weight=$3
out=$4

extra=$5

alias exp="python train.py --config-file configs/voc/step1.yaml --dist-url tcp://127.0.0.1:${port} --num-gpus ${num}"
shopt -s expand_aliases

gen_par="MODEL.WEIGHTS ${weight} OUTPUT_DIR ${out}"

for ns in 5; do
  for is in 0 1 2; do
    inc_par="DATASETS.SHOT ${ns} DATASETS.iSHOT ${is}"
      exp ${inc_par} ${gen_par} ${extra}
  done
done
