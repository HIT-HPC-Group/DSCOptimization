#!/bin/bash
#SBATCH -J DepthwiseKernelTest
#SBATCH -p ty_normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o DepthwiseKernelOutput
#SBATCH -e DepthwiseKernelError
#SBATCH --gres=dcu:1
#SBATCH --mem=20G
#SBATCH --exclusive

module switch compiler/dtk/23.04

echo "Depthwise Kernel Test Start"
echo "....................."
batchNumberOptions=(1 8 16 32 64 128)
parameterList=(32 112 112 3 1 144 56 56 3 1 192 28 28 3 1 240 28 28 5 1 384 14 14 3 1 480 14 14 3 1 480 14 14 5 1 576 14 14 3 1 672 14 14 5 1 960 7 7 3 1 1152 7 7 3 1 1152 7 7 5 1 96 112 112 3 2 144 56 56 3 2 144 56 56 5 2 192 28 28 3 2 240 28 28 3 2 576 14 14 3 2 672 14 14 5 2)
for((i = 0; i < ${#parameterList[@]}; i += 5)) do
    for batchnumber in ${batchNumberOptions[@]}; do 
            echo "InputBatchNumber: ${batchnumber}, InputChannel: ${parameterList[i]}, InputHeight: ${parameterList[i+1]}, InputWidth: ${parameterList[i+2]}, FilterHeight: ${parameterList[i+3]}, Stride: ${parameterList[i+4]}"
            cli="./build/kernel ${batchnumber} ${parameterList[i]} ${parameterList[i+1]} ${parameterList[i+3]} ${parameterList[i+4]}"
            time ${cli}
            echo "..............................................................."
            sleep 1
    done
done
echo "Finish!"