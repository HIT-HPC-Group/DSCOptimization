import torch
from torch import nn
from OptimizedDepthwiseLayer import OptimizedDepthwiseLayer
from OptimizedPointwiseLayer import OptimizedPointwiseLayer
from OriginalDepthwiseLayer import OriginalDepthwiseLayer
from OriginalPointwiseLayer import OriginalPointwiseLayer
import pandas as pd
import numpy as np

def test(inputBatchNumber, mode="baseline", loopTime=10):
    # All depthwise convolution layer configs (17)
    # # Input Channel, Input Height/Width, Input Width, Fitler Height/Width, Stride
    depthwiseLayerConfigs = [[32, 112, 112, 3, 1], [96, 112, 112, 3, 2], [144, 56, 56, 3, 1], [144, 56, 56, 3, 2],
                             [192, 28, 28, 3, 1], [192, 28, 28, 3, 1], [192, 28, 28, 3, 2], [384, 14, 14, 3, 1],
                             [384, 14, 14, 3, 1], [384, 14, 14, 3, 1], [384, 14, 14, 3, 1], [576, 14, 14, 3, 1],
                             [576, 14, 14, 3, 1], [576, 14, 14, 3, 2], [960, 7, 7, 3, 1], [960, 7, 7, 3, 1], [960, 7, 7, 3, 1],]

    # All pointwise convolution layer configs (34)
    # Input Channel, Input Height(Width), OutputChannel
    pointwiseLayerConfigs = [[32, 112, 16], [16, 112, 96], [96, 56, 24], [24, 56, 144], [144, 56, 24], [24, 56, 144],
                             [144, 28, 32], [32, 28, 192], [192, 28, 32], [32, 28, 192], [192, 28, 32], [32, 28, 192],
                             [192, 14, 64], [64, 14, 384], [384, 14, 64], [64, 14, 384], [384, 14, 64], [64, 14, 384],
                             [384, 14, 64], [64, 14, 384], [384, 14, 96], [96, 14, 576], [576, 14, 96], [96, 14, 576],
                             [576, 14, 96], [96, 14, 576], [576, 7, 160], [160, 7, 960], [960, 7, 160], [160, 7, 960],
                             [960, 7, 160], [160, 7, 960], [960, 7, 320], [320, 7, 1280],]

    depthwiseForwardTime = 0
    for depthwise in depthwiseLayerConfigs:
        inputChannel = depthwise[0]
        inputHeight = depthwise[1]
        inputWidth = depthwise[2]
        filterHeight = depthwise[3]
        stride = depthwise[4]

        if(filterHeight == 3):
            paddingHeight = paddingWidth = 1
        elif(filterHeight == 5):
            paddingHeight = paddingWidth = 2
        else:
            paddingHeight = paddingWidth = 0

        # Determine the output size
        outputBatchNumber = inputBatchNumber
        outputChannel = inputChannel
        outputHeight = int((inputHeight + paddingHeight * 2 - filterHeight) / stride + 1)
        outputWidth = int((inputWidth + paddingWidth * 2 - filterHeight) / stride + 1)

        # Randomly create input data and output data
        inputData = torch.randn(inputBatchNumber, inputChannel, inputHeight, inputWidth).to(cuda_device)

        if mode == "baseline" or mode == "onlypointwise":
            depthwiselayer = OriginalDepthwiseLayer(inputChannel, outputChannel, filterHeight, stride).to(cuda_device)
        else:
            depthwiselayer = OptimizedDepthwiseLayer(inputChannel, outputChannel, filterHeight, stride).to(cuda_device)

        # Measure performane
        time = 0
        with torch.no_grad():
            for _ in range(loopTime):
                starter.record()
                depthwiselayer(inputData)
                ender.record()
                torch.cuda.synchronize()
                time += starter.elapsed_time(ender)
        depthwiseForwardTime += time / loopTime

    pointwiseForwardTime = 0
    for pointwise in pointwiseLayerConfigs:
        inputChannel = pointwise[0]
        inputHeight = pointwise[1]
        inputWidth = pointwise[1]
        outputChannel = pointwise[2]

        # Determine the output size
        outputBatchNumber = inputBatchNumber
        outputHeight = inputHeight
        outputWidth = inputWidth

        # Randomly create input data and output data
        inputData = torch.randn(inputBatchNumber, inputChannel, inputHeight, inputWidth).to(cuda_device)

        if mode == "baseline" or mode == "onlydepthwise":
            pointwiselayer = OriginalPointwiseLayer(inputChannel, outputChannel).to(cuda_device)
        else:
            pointwiselayer = OptimizedPointwiseLayer(inputChannel, outputChannel).to(cuda_device)

        # Measure performane
        time = 0
        with torch.no_grad():
            for _ in range(loopTime):
                starter.record()
                pointwiselayer(inputData)
                ender.record()
                torch.cuda.synchronize()
                time += starter.elapsed_time(ender)
        pointwiseForwardTime += time / loopTime

    return depthwiseForwardTime + pointwiseForwardTime

# All possible batch numbers
batchNumberOptions = [64, 1, 8, 16, 32, 64]

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
loop = 100
starter = torch.cuda.Event(enable_timing = True)
ender = torch.cuda.Event(enable_timing = True)

# Warm Up
print("Start warm up.")
for batchNumber in batchNumberOptions:
    test(batchNumber, mode="baseline", loopTime=10)
print("Finish warm up.")

# Test
columns = [
    "Input Batch Size",
    "Baseline (us)",
    "Only Depthwise (us)", "Faster (%)", "Speed Up (x)",
    "Only Pointwise (us)", "Faster (%)", "Speed Up (x)",
    "All Optimized (us)", "Faster (%)", "Speed Up (x)",
]

resultTable = pd.DataFrame(columns = columns)

for batchNumber in batchNumberOptions:
    result = []

    baselineResult = test(batchNumber, mode = "baseline", loopTime=loop)
    result.append("%.2f" % baselineResult)

    onlyDepthwiseResult = test(batchNumber, mode = "onlydepthwise", loopTime=loop)
    result.append("%.2f" % onlyDepthwiseResult)
    faster = 100 * (baselineResult - onlyDepthwiseResult) / baselineResult
    speedup = baselineResult / onlyDepthwiseResult
    result.append("%.2f" % faster)
    result.append("%.2f" % speedup)

    onlyPointwiseResult = test(batchNumber, mode = "onlypointwise", loopTime=loop)
    result.append("%.2f" % onlyPointwiseResult)
    faster = 100 * (baselineResult - onlyPointwiseResult) / baselineResult
    speedup = baselineResult / onlyPointwiseResult
    result.append("%.2f" % faster)
    result.append("%.2f" % speedup)

    alloptimizedResult = test(batchNumber, mode = "alloptimized", loopTime=loop)
    result.append("%.2f" % alloptimizedResult)
    faster = 100 * (baselineResult - alloptimizedResult) / baselineResult
    speedup = baselineResult / alloptimizedResult
    result.append("%.2f" % faster)
    result.append("%.2f" % speedup)

    resultTable = pd.DataFrame(
        np.insert(resultTable.values, len(resultTable.index),
        values=[batchNumber,
                result[0],
                result[1], result[2], result[3],
                result[4], result[5], result[6],
                result[7], result[8], result[9],], axis = 0),columns = columns)

resultTable.to_csv("DCU_MobilenetAblation_Result.csv")