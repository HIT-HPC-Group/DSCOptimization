/*
Pointwise Convolution Kernel
InputBatch_1_Input_14x14_InChannel_480_OutChannel_96

Grid:
    gridDim.x = (1 * 96 * 14 * 14) / (4 * 7 * 4);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 7
outputChannelPerWarp = 4
channelGroupSize = 32
horizontalRepeat = 2
verticalRepeat = 2

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 7 * 4 output data.
Each warp is responsible for generating 7 * 4 output data.

DCU: 1 480 14 96
Kernel: TODO: ms
miopen: TODO: ms
*/
template <typename scalar_t>
__global__ void InputBatch_1_Input_14x14_InChannel_480_OutChannel_96(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[2 * 7 * 32];
    __shared__ float inputSharedBuffer2[2 * 7 * 32];

    __shared__ float filterSharedBuffer1[2 * 4 * 32];
    __shared__ float filterSharedBuffer2[2 * 4 * 32];

    // to hold loaded operands temp.
    float inputTemp1 = 0, inputTemp2 = 0;
    float filterTemp1 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0;
    float filterOperand1 = 0, filterOperand2 = 0;

    // to hold intermediate result
    float input1filter1 = 0, input1filter2 = 0;

    float input2filter1 = 0, input2filter2 = 0;

    float input3filter1 = 0, input3filter2 = 0;

    float input4filter1 = 0, input4filter2 = 0;

    float input5filter1 = 0, input5filter2 = 0;

    float input6filter1 = 0, input6filter2 = 0;

    float input7filter1 = 0, input7filter2 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 168 * 94080 + (blockIdx.x % 168) / 24 * 28 + (blockIdx.x % 24) / 12 * 7;
    inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 14 * 196 + ((threadIdx.x + 0 * 256) % 14) / 7 * 14 + (threadIdx.x + 0 * 256) % 7];
    if(threadIdx.x < 2 * 7 * 32 - 1 * 256) {
        inputSharedBuffer1[threadIdx.x + 1 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 14 * 196 + ((threadIdx.x + 1 * 256) % 14) / 7 * 14 + (threadIdx.x + 1 * 256) % 7];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 12) * 3840;
    filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 32) * 480 + ((threadIdx.x + 0 * 256) % 32)];
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 32) / (2 * 32); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 14 * 14 * 32;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 14 * 196 + ((threadIdx.x + 0 * 256) % 14) / 7 * 14 + (threadIdx.x + 0 * 256) % 7];
        if(threadIdx.x < 2 * 7 * 32 - 1 * 256) {
            inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 14 * 196 + ((threadIdx.x + 1 * 256) % 14) / 7 * 14 + (threadIdx.x + 1 * 256) % 7];
        }

        blockLoadFilterStartIdx += 32;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 32) * 480 + ((threadIdx.x + 0 * 256) % 32)];

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 6];

        filterOperand1 = filterSharedBuffer1[(warpID % 2) * 128 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 2) * 128 + laneID + 1 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;

        // Copy Temp Registers to shared buffer 2
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        if(threadIdx.x < 2 * 7 * 32 - 1 * 256) {
            inputSharedBuffer2[threadIdx.x + 1 * 256] = inputTemp2;
        }

        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 14 * 14 * 32;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 14 * 196 + ((threadIdx.x + 0 * 256) % 14) / 7 * 14 + (threadIdx.x + 0 * 256) % 7];
        if(threadIdx.x < 2 * 7 * 32 - 1 * 256) {
            inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 14 * 196 + ((threadIdx.x + 1 * 256) % 14) / 7 * 14 + (threadIdx.x + 1 * 256) % 7];
        }

        blockLoadFilterStartIdx += 32;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 32) * 480 + ((threadIdx.x + 0 * 256) % 32)];

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 6];

        filterOperand1 = filterSharedBuffer2[(warpID % 2) * 128 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 2) * 128 + laneID + 1 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;

        // Copy Temp Registers to shared buffer 1
        inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        if(threadIdx.x < 2 * 7 * 32 - 1 * 256) {
            inputSharedBuffer1[threadIdx.x + 1 * 256] = inputTemp2;
        }

        filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        __syncthreads();
    }
    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 32) * 7 * 2 + 6];

    filterOperand1 = filterSharedBuffer1[(warpID % 2) * 128 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 2) * 128 + laneID + 1 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;
    input1filter2 += inputOperand1 * filterOperand2;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;

    input5filter1 += inputOperand5 * filterOperand1;
    input5filter2 += inputOperand5 * filterOperand2;

    input6filter1 += inputOperand6 * filterOperand1;
    input6filter2 += inputOperand6 * filterOperand2;

    input7filter1 += inputOperand7 * filterOperand1;
    input7filter2 += inputOperand7 * filterOperand2;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 32);
        input1filter2 += __shfl_down(input1filter2, offset, 32);

        input2filter1 += __shfl_down(input2filter1, offset, 32);
        input2filter2 += __shfl_down(input2filter2, offset, 32);

        input3filter1 += __shfl_down(input3filter1, offset, 32);
        input3filter2 += __shfl_down(input3filter2, offset, 32);

        input4filter1 += __shfl_down(input4filter1, offset, 32);
        input4filter2 += __shfl_down(input4filter2, offset, 32);

        input5filter1 += __shfl_down(input5filter1, offset, 32);
        input5filter2 += __shfl_down(input5filter2, offset, 32);

        input6filter1 += __shfl_down(input6filter1, offset, 32);
        input6filter2 += __shfl_down(input6filter2, offset, 32);

        input7filter1 += __shfl_down(input7filter1, offset, 32);
        input7filter2 += __shfl_down(input7filter2, offset, 32);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 168 * 18816 + (blockIdx.x % 168) / 24 * 28 + (blockIdx.x % 24) / 12 * 7 + (blockIdx.x % 12) * 1568;

    if(laneID % 32 == 0) {
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 6] = input7filter1;

        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 784 + (laneID / 32) * outputHeight * outputWidth + 6] = input7filter2;
    }
}
