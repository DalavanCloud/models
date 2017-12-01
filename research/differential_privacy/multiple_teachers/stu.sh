#!/bin/sh
python train_student.py --d_stu=$1 --nb_teachers=50 --dataset=svhn --dataset_teacher=mnist --stdnt_share=1000 --train_dir=train_dir --data_dir=data_dir --teachers_dir=train_dir --max_steps=3000 --teachers_max_steps=3000
