/*
Pointwise Convolution Kernel
InputBatch_1_Input_28x28_InChannel_24_OutChannel_24

Grid:
    gridDim.x = (1 * 24 * 28 * 28) / (14 * 4 * 8);
Block:
    blockDim.x = 64 * 14;

warpNumPerBlock = 14
outputWidthPerWarp = 4
outputChannelPerWarp = 8
channelGroupSize = 8
horizontalRepeat = 14
verticalRepeat = 1

One thread block contains 14 warps, 14 * 64 = 896 threads.
Each thread block is responsible for generating 14 * 4 * 8 output data.
Each warp is responsible for generating 4 * 8 output data.

DCU: 1 24 28 24
Kernel: TODO: ms
miopen: TODO: ms
*/
template <typename scalar_t>
__global__ __launch_bounds__(1024) void InputBatch_1_Input_28x28_InChannel_24_OutChannel_24(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[14 * 4 * 8];
    __shared__ float inputSharedBuffer2[14 * 4 * 8];

    __shared__ float filterSharedBuffer1[1 * 8 * 8];
    __shared__ float filterSharedBuffer2[1 * 8 * 8];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0;
    float filterOperand1 = 0;

    // to hold intermediate result
    float input1filter1 = 0;

    float input2filter1 = 0;

    float input3filter1 = 0;

    float input4filter1 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 42 * 18816 + (blockIdx.x % 42) / 21 * 392 + (blockIdx.x % 21) / 3 * 4;
    if(threadIdx.x < 14 * 4 * 8 - 0 * 896) {
        inputSharedBuffer1[threadIdx.x + 0 * 896] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 896) / 56 * 784 + ((threadIdx.x + 0 * 896) % 56) / 4 * 28 + (threadIdx.x + 0 * 896) % 4];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 3) * 192;
    if(threadIdx.x < 1 * 8 * 8 - 0 * 896) {
        filterSharedBuffer1[threadIdx.x + 0 * 896] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 896) / 8) * 24 + ((threadIdx.x + 0 * 896) % 8)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 8) / (2 * 8); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 28 * 28 * 8;
        if(threadIdx.x < 14 * 4 * 8 - 0 * 896) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 896) / 56 * 784 + ((threadIdx.x + 0 * 896) % 56) / 4 * 28 + (threadIdx.x + 0 * 896) % 4];
        }

        blockLoadFilterStartIdx += 8;
        if(threadIdx.x < 1 * 8 * 8 - 0 * 896) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 896) / 8) * 24 + ((threadIdx.x + 0 * 896) % 8)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 3];

        filterOperand1 = filterSharedBuffer1[(warpID % 1) * 64 + laneID + 0 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;

        input2filter1 += inputOperand2 * filterOperand1;

        input3filter1 += inputOperand3 * filterOperand1;

        input4filter1 += inputOperand4 * filterOperand1;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 14 * 4 * 8 - 0 * 896) {
            inputSharedBuffer2[threadIdx.x + 0 * 896] = inputTemp1;
        }

        if(threadIdx.x < 1 * 8 * 8 - 0 * 896) {
            filterSharedBuffer2[threadIdx.x + 0 * 896] = filterTemp1;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 28 * 28 * 8;
        if(threadIdx.x < 14 * 4 * 8 - 0 * 896) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 896) / 56 * 784 + ((threadIdx.x + 0 * 896) % 56) / 4 * 28 + (threadIdx.x + 0 * 896) % 4];
        }

        blockLoadFilterStartIdx += 8;
        if(threadIdx.x < 1 * 8 * 8 - 0 * 896) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 896) / 8) * 24 + ((threadIdx.x + 0 * 896) % 8)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 3];

        filterOperand1 = filterSharedBuffer2[(warpID % 1) * 64 + laneID + 0 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;

        input2filter1 += inputOperand2 * filterOperand1;

        input3filter1 += inputOperand3 * filterOperand1;

        input4filter1 += inputOperand4 * filterOperand1;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 14 * 4 * 8 - 0 * 896) {
            inputSharedBuffer1[threadIdx.x + 0 * 896] = inputTemp1;
        }

        if(threadIdx.x < 1 * 8 * 8 - 0 * 896) {
            filterSharedBuffer1[threadIdx.x + 0 * 896] = filterTemp1;
        }
        __syncthreads();
    }
    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 1) % 14) * 4 + (laneID % 8) * 4 * 14 + 3];

    filterOperand1 = filterSharedBuffer1[(warpID % 1) * 64 + laneID + 0 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;

    input2filter1 += inputOperand2 * filterOperand1;

    input3filter1 += inputOperand3 * filterOperand1;

    input4filter1 += inputOperand4 * filterOperand1;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (8 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 8);

        input2filter1 += __shfl_down(input2filter1, offset, 8);

        input3filter1 += __shfl_down(input3filter1, offset, 8);

        input4filter1 += __shfl_down(input4filter1, offset, 8);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 42 * 18816 + (blockIdx.x % 42) / 21 * 392 + (blockIdx.x % 21) / 3 * 4 + (blockIdx.x % 3) * 6272;

    if(laneID % 8 == 0) {
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 6272 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 6272 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 6272 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 6272 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter1;
    }
}
