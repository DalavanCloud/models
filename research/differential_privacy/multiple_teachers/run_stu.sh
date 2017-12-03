#!/bin/sh

test_name='digit.test-MNIST-l2t' 
sbatch -p  titanx-short --gres=gpu:2 --output=$test_name.out stu.sh digit -1 $test_name 