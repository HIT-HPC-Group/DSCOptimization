/*
Pointwise Convolution Kernel
InputBatch_8_Input_28x28_InChannel_120_OutChannel_40

Grid:
    gridDim.x = (8 * 40 * 28 * 28) / (4 * 4 * 40);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 4
outputChannelPerWarp = 40
channelGroupSize = 8
horizontalRepeat = 4
verticalRepeat = 1

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 4 * 40 output data.
Each warp is responsible for generating 4 * 40 output data.

DCU: 8 120 28 40
Kernel: TODO: ms
miopen: TODO: ms
*/

__global__ void InputBatch_8_Input_28x28_InChannel_120_OutChannel_40(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[4 * 4 * 8];
    __shared__ float inputSharedBuffer2[4 * 4 * 8];

    __shared__ float filterSharedBuffer1[1 * 40 * 8];
    __shared__ float filterSharedBuffer2[1 * 40 * 8];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0, filterTemp2 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0, filterOperand4 = 0, filterOperand5 = 0;

    // to hold intermediate result
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0, input1filter4 = 0, input1filter5 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0, input2filter4 = 0, input2filter5 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0, input3filter4 = 0, input3filter5 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0, input4filter4 = 0, input4filter5 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 49 * 94080 + (blockIdx.x % 49) / 7 * 112 + (blockIdx.x % 7) / 1 * 4;
    if(threadIdx.x < 4 * 4 * 8 - 0 * 256) {
        inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 16 * 784 + ((threadIdx.x + 0 * 256) % 16) / 4 * 28 + (threadIdx.x + 0 * 256) % 4];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 1) * 4800;
    filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 120 + ((threadIdx.x + 0 * 256) % 8)];
    if(threadIdx.x < 1 * 40 * 8 - 1 * 256) {
        filterSharedBuffer1[threadIdx.x + 1 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 120 + ((threadIdx.x + 1 * 256) % 8)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 8) / (2 * 8); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 28 * 28 * 8;
        if(threadIdx.x < 4 * 4 * 8 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 16 * 784 + ((threadIdx.x + 0 * 256) % 16) / 4 * 28 + (threadIdx.x + 0 * 256) % 4];
        }

        blockLoadFilterStartIdx += 8;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 120 + ((threadIdx.x + 0 * 256) % 8)];
        if(threadIdx.x < 1 * 40 * 8 - 1 * 256) {
            filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 120 + ((threadIdx.x + 1 * 256) % 8)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 3];

        filterOperand1 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 4 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 4 * 4 * 8 - 0 * 256) {
            inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        }

        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        if(threadIdx.x < 1 * 40 * 8 - 1 * 256) {
            filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 28 * 28 * 8;
        if(threadIdx.x < 4 * 4 * 8 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 16 * 784 + ((threadIdx.x + 0 * 256) % 16) / 4 * 28 + (threadIdx.x + 0 * 256) % 4];
        }

        blockLoadFilterStartIdx += 8;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 120 + ((threadIdx.x + 0 * 256) % 8)];
        if(threadIdx.x < 1 * 40 * 8 - 1 * 256) {
            filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 120 + ((threadIdx.x + 1 * 256) % 8)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 3];

        filterOperand1 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 4 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 4 * 4 * 8 - 0 * 256) {
            inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        }

        filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        if(threadIdx.x < 1 * 40 * 8 - 1 * 256) {
            filterSharedBuffer1[threadIdx.x + 1 * 256] = filterTemp2;
        }
        __syncthreads();
    }
    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 8) * 4 * 4 + 3];

    filterOperand1 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 4 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;
    input1filter2 += inputOperand1 * filterOperand2;
    input1filter3 += inputOperand1 * filterOperand3;
    input1filter4 += inputOperand1 * filterOperand4;
    input1filter5 += inputOperand1 * filterOperand5;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;
    input2filter4 += inputOperand2 * filterOperand4;
    input2filter5 += inputOperand2 * filterOperand5;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;
    input3filter4 += inputOperand3 * filterOperand4;
    input3filter5 += inputOperand3 * filterOperand5;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;
    input4filter4 += inputOperand4 * filterOperand4;
    input4filter5 += inputOperand4 * filterOperand5;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (8 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 8);
        input1filter2 += __shfl_down(input1filter2, offset, 8);
        input1filter3 += __shfl_down(input1filter3, offset, 8);
        input1filter4 += __shfl_down(input1filter4, offset, 8);
        input1filter5 += __shfl_down(input1filter5, offset, 8);

        input2filter1 += __shfl_down(input2filter1, offset, 8);
        input2filter2 += __shfl_down(input2filter2, offset, 8);
        input2filter3 += __shfl_down(input2filter3, offset, 8);
        input2filter4 += __shfl_down(input2filter4, offset, 8);
        input2filter5 += __shfl_down(input2filter5, offset, 8);

        input3filter1 += __shfl_down(input3filter1, offset, 8);
        input3filter2 += __shfl_down(input3filter2, offset, 8);
        input3filter3 += __shfl_down(input3filter3, offset, 8);
        input3filter4 += __shfl_down(input3filter4, offset, 8);
        input3filter5 += __shfl_down(input3filter5, offset, 8);

        input4filter1 += __shfl_down(input4filter1, offset, 8);
        input4filter2 += __shfl_down(input4filter2, offset, 8);
        input4filter3 += __shfl_down(input4filter3, offset, 8);
        input4filter4 += __shfl_down(input4filter4, offset, 8);
        input4filter5 += __shfl_down(input4filter5, offset, 8);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 49 * 31360 + (blockIdx.x % 49) / 7 * 112 + (blockIdx.x % 7) / 1 * 4 + (blockIdx.x % 1) * 31360;

    if(laneID % 8 == 0) {
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter1;

        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter2;

        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter3;

        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter4;

        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 31360 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter5;
    }
}
