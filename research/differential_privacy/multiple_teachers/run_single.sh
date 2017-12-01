#!/bin/sh
echo $1 $2
python train_teachers.py --dataset=mnist --nb_teachers=$1 --teacher_id=$2 --train_dir=train_dir --data_dir=data_dir --max_steps=500 --mem=8000

