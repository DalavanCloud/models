#!/bin/sh

test_name='digit.test-MNIST-l2t' 
test_names=('all-digits-MNIST-gammaC' 'all-digits-MNIST-gammaC-sh-asp' 'all-digits-MNIST-gammaC-sh' 'digit.test-MNIST')

#for t in "${test_names[@]}"
#do
#    sbatch -p  titanx-short --gres=gpu:2 --output=$t.out stu.sh digit -1 $t
#done

data_names=('svhn' 'cifar10')

for d in "${data_names[@]}"
do
    for i in `seq 0 3`
    do
      test_name=$d'_'$i
      echo $test_name
      sbatch -p  titanx-short --gres=gpu:2 --output=$test_name.out stu.sh $d $i $test_name 
    done
done