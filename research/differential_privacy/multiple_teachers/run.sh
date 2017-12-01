#!/bin/sh
for i in `seq 1 $1`;
do
    sbatch -p  titanx-short --gres=gpu:1 run_single.sh $1 $((i -1))
done

