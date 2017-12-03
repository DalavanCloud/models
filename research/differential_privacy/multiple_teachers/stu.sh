#!/bin/sh
echo 'data' $1 'channel' $2 'test_name' $3
python train_student.py --d_stu=$2 --nb_teachers=50 --dataset=$1 --dataset_teacher=mnist \
--stdnt_share=1000 --train_dir=train_dir --data_dir=data_dir --teachers_dir=train_dir \
--max_steps=3000 --teachers_max_steps=3000 --test_name=$3