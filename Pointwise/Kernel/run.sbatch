#!/bin/bash
#SBATCH -J PointwiseKernelTest
#SBATCH -p ty_normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o PointwiseKernelOutput
#SBATCH -e PointwiseKernelError
#SBATCH --gres=dcu:1
#SBATCH --mem=20G
#SBATCH --exclusive

module switch compiler/dtk/23.04

echo "Pointwise Kernel Test Start"
echo "....................."
batchNumberOptions = (1 8 16 32 64 128)
parameterList = (576 7 160)

for((i = 0; i < ${#parameterList[@]}; i += 3)) do
    for batchnumber in ${batchNumberOptions[@]}; do 
            echo "InputBatchNumber: ${batchnumber}, InputChannel: ${parameterList[i]}, InputHeight: ${parameterList[i+1]}, OutputChannel: ${parameterList[i+2]}"
            cli = "./build/kernel ${batchnumber} ${parameterList[i]} ${parameterList[i+1]} ${parameterList[i+2]}"
            time ${cli}
            echo "..............................................................."
            sleep 1
    done
done
echo "Finish!"