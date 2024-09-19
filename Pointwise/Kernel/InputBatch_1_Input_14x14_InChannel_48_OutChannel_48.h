/*
Pointwise Convolution Kernel
InputBatch_1_Input_14x14_InChannel_48_OutChannel_48

Grid:
    gridDim.x = (1 * 48 * 14 * 14) / (4 * 7 * 8);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 7
outputChannelPerWarp = 8
channelGroupSize = 8
horizontalRepeat = 2
verticalRepeat = 2

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 7 * 8 output data.
Each warp is responsible for generating 7 * 8 output data.

DCU: 1 48 14 48
Kernel: TODO: ms
miopen: TODO: ms
*/

__global__ void InputBatch_1_Input_14x14_InChannel_48_OutChannel_48(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[2 * 7 * 8];
    __shared__ float inputSharedBuffer2[2 * 7 * 8];

    __shared__ float filterSharedBuffer1[2 * 8 * 8];
    __shared__ float filterSharedBuffer2[2 * 8 * 8];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0;
    float filterOperand1 = 0;

    // to hold intermediate result
    float input1filter1 = 0;

    float input2filter1 = 0;

    float input3filter1 = 0;

    float input4filter1 = 0;

    float input5filter1 = 0;

    float input6filter1 = 0;

    float input7filter1 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 42 * 9408 + (blockIdx.x % 42) / 6 * 28 + (blockIdx.x % 6) / 3 * 7;
    if(threadIdx.x < 2 * 7 * 8 - 0 * 256) {
        inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 14 * 196 + ((threadIdx.x + 0 * 256) % 14) / 7 * 14 + (threadIdx.x + 0 * 256) % 7];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 3) * 768;
    if(threadIdx.x < 2 * 8 * 8 - 0 * 256) {
        filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 48 + ((threadIdx.x + 0 * 256) % 8)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 8) / (2 * 8); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 14 * 14 * 8;
        if(threadIdx.x < 2 * 7 * 8 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 14 * 196 + ((threadIdx.x + 0 * 256) % 14) / 7 * 14 + (threadIdx.x + 0 * 256) % 7];
        }

        blockLoadFilterStartIdx += 8;
        if(threadIdx.x < 2 * 8 * 8 - 0 * 256) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 48 + ((threadIdx.x + 0 * 256) % 8)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 6];

        filterOperand1 = filterSharedBuffer1[(warpID % 2) * 64 + laneID + 0 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;

        input2filter1 += inputOperand2 * filterOperand1;

        input3filter1 += inputOperand3 * filterOperand1;

        input4filter1 += inputOperand4 * filterOperand1;

        input5filter1 += inputOperand5 * filterOperand1;

        input6filter1 += inputOperand6 * filterOperand1;

        input7filter1 += inputOperand7 * filterOperand1;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 2 * 7 * 8 - 0 * 256) {
            inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        }

        if(threadIdx.x < 2 * 8 * 8 - 0 * 256) {
            filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 14 * 14 * 8;
        if(threadIdx.x < 2 * 7 * 8 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 14 * 196 + ((threadIdx.x + 0 * 256) % 14) / 7 * 14 + (threadIdx.x + 0 * 256) % 7];
        }

        blockLoadFilterStartIdx += 8;
        if(threadIdx.x < 2 * 8 * 8 - 0 * 256) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 48 + ((threadIdx.x + 0 * 256) % 8)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 6];

        filterOperand1 = filterSharedBuffer2[(warpID % 2) * 64 + laneID + 0 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;

        input2filter1 += inputOperand2 * filterOperand1;

        input3filter1 += inputOperand3 * filterOperand1;

        input4filter1 += inputOperand4 * filterOperand1;

        input5filter1 += inputOperand5 * filterOperand1;

        input6filter1 += inputOperand6 * filterOperand1;

        input7filter1 += inputOperand7 * filterOperand1;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 2 * 7 * 8 - 0 * 256) {
            inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        }

        if(threadIdx.x < 2 * 8 * 8 - 0 * 256) {
            filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        }
        __syncthreads();
    }
    // load next group of Cnum channels
    blockLoadInputStartIdx += 14 * 14 * 8;
    if(threadIdx.x < 2 * 7 * 8 - 0 * 256) {
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 14 * 196 + ((threadIdx.x + 0 * 256) % 14) / 7 * 14 + (threadIdx.x + 0 * 256) % 7];
    }

    blockLoadFilterStartIdx += 8;
    if(threadIdx.x < 2 * 8 * 8 - 0 * 256) {
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 48 + ((threadIdx.x + 0 * 256) % 8)];
    }

    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 6];

    filterOperand1 = filterSharedBuffer1[(warpID % 2) * 64 + laneID + 0 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;

    input2filter1 += inputOperand2 * filterOperand1;

    input3filter1 += inputOperand3 * filterOperand1;

    input4filter1 += inputOperand4 * filterOperand1;

    input5filter1 += inputOperand5 * filterOperand1;

    input6filter1 += inputOperand6 * filterOperand1;

    input7filter1 += inputOperand7 * filterOperand1;

    // Copy Temp Registers to shared buffer 2
    if(threadIdx.x < 2 * 7 * 8 - 0 * 256) {
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
    }

    if(threadIdx.x < 2 * 8 * 8 - 0 * 256) {
        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
    }
    __syncthreads();

    // Exchange shared buffer 1 and shared buffer 2 and repeat
    // Copy operands from shared buffer 2 into Operands Registers
    inputOperand1 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 0];
    inputOperand2 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 1];
    inputOperand3 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 2];
    inputOperand4 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 3];
    inputOperand5 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 4];

    inputOperand6 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 5];
    inputOperand7 = inputSharedBuffer2[((warpID / 2) % 2) * 7 + (laneID % 8) * 7 * 2 + 6];

    filterOperand1 = filterSharedBuffer2[(warpID % 2) * 64 + laneID + 0 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;

    input2filter1 += inputOperand2 * filterOperand1;

    input3filter1 += inputOperand3 * filterOperand1;

    input4filter1 += inputOperand4 * filterOperand1;

    input5filter1 += inputOperand5 * filterOperand1;

    input6filter1 += inputOperand6 * filterOperand1;

    input7filter1 += inputOperand7 * filterOperand1;

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

        input5filter1 += __shfl_down(input5filter1, offset, 8);

        input6filter1 += __shfl_down(input6filter1, offset, 8);

        input7filter1 += __shfl_down(input7filter1, offset, 8);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 42 * 9408 + (blockIdx.x % 42) / 6 * 28 + (blockIdx.x % 6) / 3 * 7 + (blockIdx.x % 3) * 3136;

    if(laneID % 8 == 0) {
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 1568 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 1568 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 1568 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 1568 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 1568 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 1568 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 7 + (warpID % 2) * 1568 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter1;
    }
}
