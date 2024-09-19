import os
import time
import pandas as pd
import numpy as np
import re
# All batch size
batchSizeList = [1, 8, 16, 32, 64]

# All layer configurations in
# MobileNet V2 and EfficientNet B0 (18 in total)
# MNasNet and ShuffleNet V2 (12 in total)
# Parameter Order: "Input Channel", "Input Height/Width", "Filter Height/Width", "Stride", 
paramList = [
    [32, 112, 112, 3, 1],
    [144, 56, 56, 3, 1],
    [192, 28, 28, 3, 1],
    [240, 28, 28, 5, 1],
    [384, 14, 14, 3, 1],
    [480, 14, 14, 3, 1],
    [480, 14, 14, 5, 1],
    [576, 14, 14, 3, 1],
    [672, 14, 14, 5, 1],
    [960, 7, 7, 3, 1],
    [1152, 7, 7, 3, 1],
    [1152, 7, 7, 5, 1],
    [96, 112, 112, 3, 2],
    [144, 56, 56, 3, 2],
    [144, 56, 56, 5, 2],
    [192, 28, 28, 3, 2],
    [240, 28, 28, 3, 2],
    [576, 14, 14, 3, 2],
    [672, 14, 14, 5, 2],

    [72, 56, 56, 3, 1],
    [120, 28, 28, 5, 1],
    [24, 28, 28, 3, 1],
    [48, 14, 14, 3, 1],
    [96, 7, 7, 3, 1],
    [48, 112, 112, 3, 2],
    [72, 56, 56, 5, 2],
    [576, 14, 14, 5, 2],
    [24, 56, 56, 3, 2],
    [48, 28, 28, 3, 2],
    [96, 14, 14, 3, 2]
]

loopTime = 3

# Create table
columns = [
    "Input Channel", "Input Height/Width", "Filter Height/Width", "Stride", 
    "Input Batch = 1 - Kernel (us)", "Input Batch = 1 - MIOpen (us)", "Faster (%)", "Speed Up", "Input Batch = 1 - Kernel Hit (%)", "Input Batch = 1 - MIOpen Hit (%)",
    "Input Batch = 8 - Kernel (us)", "Input Batch = 8 - MIOpen (us)", "Faster (%)", "Speed Up", "Input Batch = 8 - Kernel Hit (%)", "Input Batch = 8 - MIOpen Hit (%)",
    "Input Batch = 16 - Kernel (us)", "Input Batch = 16 - MIOpen (us)", "Faster (%)", "Speed Up", "Input Batch = 16 - Kernel Hit (%)", "Input Batch = 16 - MIOpen Hit (%)",
    "Input Batch = 32 - Kernel (us)", "Input Batch = 32 - MIOpen (us)", "Faster (%)", "Speed Up", "Input Batch = 32 - Kernel Hit (%)", "Input Batch = 32 - MIOpen Hit (%)",
    "Input Batch = 64 - Kernel (us)", "Input Batch = 64 - MIOpen (us)", "Faster (%)", "Speed Up","Input Batch = 64 - Kernel Hit (%)", "Input Batch = 64 - MIOpen Hit (%)",
]

resultTable = pd.DataFrame(columns = columns)

os.system("rm -rf DCU_Depthwise_CacheHit_Result.txt")
# Profile Kernels
for param in paramList:
    result = []
    for batchSize in batchSizeList:
        kernelTime = 0
        miopenTime = 0
        kernelhit = 0
        miopenhit = 0
        for i in range(loopTime):
            print("Profiling Input Batch: " + str(batchSize) + ", " + 
                  "Input Channel: " + str(param[0]) + ", " + 
                  "Input Height: " + str(param[1]) + ", " +
                  "Filter Height: " + str(param[3]) + ", " +
                  "Stride: " + str(param[4]) + " " + "for " + str(i + 1) + " time.")
            cli = "hipprof --pmc ./build/kernel" + " " + str(batchSize) + " " + str(param[0]) + " " + str(param[1]) + " " + str(param[3]) + " " + str(param[4]) + " >> hitresult.txt"
            os.system(cli)

            # Find the corresponding profiling process output file.
            profprocid = 0
            with open("hitresult.txt", "r") as f1:
                lines = f1.readlines()
                for line in lines:
                    if "HIP_PROF:process id" in line:
                        match = re.search(r"'(\d+)'", line)
                        profprocid = match.group(1) if match else None
                        break
            os.system("rm -rf hitresult.txt")

            # Find the cache hit rate from the profiling result file
            pmc_file = "pmc_results_" + str(profprocid) + ".txt"
            with open(pmc_file, "r") as f2:
                lines = f2.readlines()
                for i in range(0, len(lines)):
                    if "Filter" in lines[i] and "Input" in lines[i] and "Stride" in lines[i]:
                        kernelTime += float(lines[i + 14].split()[-1].split('(')[0])
                        kernelhit += float(lines[i + 22].split()[-1].split('(')[0])
                    if "naive" in lines[i] or "MIOpenConvUni" in lines[i] or "miopen" in lines[i]:
                        miopenTime += float(lines[i + 14].split()[-1].split('(')[0])
                        miopenhit += float(lines[i + 22].split()[-1].split('(')[0])
            os.system("rm -rf " + pmc_file)

        kernelTime = 1000000 * kernelTime / loopTime
        miopenTime = 1000000 * miopenTime / loopTime
        result.append(kernelTime)
        result.append(miopenTime)

        faster = 100 * (miopenTime - kernelTime) / miopenTime
        result.append("%.3f" % faster)
        speedup = miopenTime / kernelTime
        result.append("%.3f" % speedup)

        result.append(kernelhit / loopTime)
        result.append(miopenhit / loopTime)

    resultTable = pd.DataFrame(
        np.insert(resultTable.values, len(resultTable.index), 
        values = [param[0], param[1], param[3], param[4], 
        result[0], result[1], result[2], result[3], result[4], result[5], 
        result[6], result[7], result[8], result[9], result[10], result[11], 
        result[12], result[13], result[14], result[15], result[16], result[17], 
        result[18], result[19], result[20], result[21], result[22], result[23], 
        result[24], result[25], result[26], result[27], result[28], result[29]], axis = 0), 
        columns = columns)

# Output table
resultTable.to_csv("DCU_Depthwise_CacheHit_Result.csv")