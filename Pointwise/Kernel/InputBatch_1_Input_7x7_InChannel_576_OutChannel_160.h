/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_576_OutChannel_160

Grid:
    gridDim.x = (1 * 160 * 7 * 7) / (4 * 7 * 2);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 7
outputChannelPerWarp = 2
channelGroupSize = 64
horizontalRepeat = 1
verticalRepeat = 4

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 7 * 2 output data.
Each warp is responsible for generating 7 * 2 output data.
*/

__global__ void InputBatch_1_Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[1 * 7 * 64];
    __shared__ float inputSharedBuffer2[1 * 7 * 64];

    __shared__ float filterSharedBuffer1[4 * 2 * 64];
    __shared__ float filterSharedBuffer2[4 * 2 * 64];

    // to hold loaded operands temp.
    float inputTemp1 = 0, inputTemp2 = 0;
    float filterTemp1 = 0, filterTemp2 = 0;

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
    int blockLoadInputStartIdx = blockIdx.x / 140 * 28224 + (blockIdx.x % 140) / 20 * 7 + (blockIdx.x % 20) / 20 * 7;
    inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 7 * 49 + ((threadIdx.x + 0 * 256) % 7) / 7 * 7 + (threadIdx.x + 0 * 256) % 7];
    if(threadIdx.x < 1 * 7 * 64 - 1 * 256) {
        inputSharedBuffer1[threadIdx.x + 1 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 7 * 49 + ((threadIdx.x + 1 * 256) % 7) / 7 * 7 + (threadIdx.x + 1 * 256) % 7];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 20) * 4608;
    filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 64) * 576 + ((threadIdx.x + 0 * 256) % 64)];
    filterSharedBuffer1[threadIdx.x + 1 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 64) * 576 + ((threadIdx.x + 1 * 256) % 64)];
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 64) / (2 * 64); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 64;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 7 * 49 + ((threadIdx.x + 0 * 256) % 7) / 7 * 7 + (threadIdx.x + 0 * 256) % 7];
        if(threadIdx.x < 1 * 7 * 64 - 1 * 256) {
            inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 7 * 49 + ((threadIdx.x + 1 * 256) % 7) / 7 * 7 + (threadIdx.x + 1 * 256) % 7];
        }

        blockLoadFilterStartIdx += 64;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 64) * 576 + ((threadIdx.x + 0 * 256) % 64)];
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 64) * 576 + ((threadIdx.x + 1 * 256) % 64)];

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 6];

        filterOperand1 = filterSharedBuffer1[(warpID % 4) * 128 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 4) * 128 + laneID + 1 * 64];

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
        if(threadIdx.x < 1 * 7 * 64 - 1 * 256) {
            inputSharedBuffer2[threadIdx.x + 1 * 256] = inputTemp2;
        }

        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 64;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 7 * 49 + ((threadIdx.x + 0 * 256) % 7) / 7 * 7 + (threadIdx.x + 0 * 256) % 7];
        if(threadIdx.x < 1 * 7 * 64 - 1 * 256) {
            inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 7 * 49 + ((threadIdx.x + 1 * 256) % 7) / 7 * 7 + (threadIdx.x + 1 * 256) % 7];
        }

        blockLoadFilterStartIdx += 64;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 64) * 576 + ((threadIdx.x + 0 * 256) % 64)];
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 64) * 576 + ((threadIdx.x + 1 * 256) % 64)];

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 6];

        filterOperand1 = filterSharedBuffer2[(warpID % 4) * 128 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 4) * 128 + laneID + 1 * 64];

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
        if(threadIdx.x < 1 * 7 * 64 - 1 * 256) {
            inputSharedBuffer1[threadIdx.x + 1 * 256] = inputTemp2;
        }

        filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        filterSharedBuffer1[threadIdx.x + 1 * 256] = filterTemp2;
        __syncthreads();
    }
    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 6];

    filterOperand1 = filterSharedBuffer1[(warpID % 4) * 128 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 4) * 128 + laneID + 1 * 64];

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
    for (int offset = (64 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 64);
        input1filter2 += __shfl_down(input1filter2, offset, 64);

        input2filter1 += __shfl_down(input2filter1, offset, 64);
        input2filter2 += __shfl_down(input2filter2, offset, 64);

        input3filter1 += __shfl_down(input3filter1, offset, 64);
        input3filter2 += __shfl_down(input3filter2, offset, 64);

        input4filter1 += __shfl_down(input4filter1, offset, 64);
        input4filter2 += __shfl_down(input4filter2, offset, 64);

        input5filter1 += __shfl_down(input5filter1, offset, 64);
        input5filter2 += __shfl_down(input5filter2, offset, 64);

        input6filter1 += __shfl_down(input6filter1, offset, 64);
        input6filter2 += __shfl_down(input6filter2, offset, 64);

        input7filter1 += __shfl_down(input7filter1, offset, 64);
        input7filter2 += __shfl_down(input7filter2, offset, 64);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 140 * 7840 + (blockIdx.x % 140) / 20 * 7 + (blockIdx.x % 20) / 20 * 7 + (blockIdx.x % 20) * 392;

    if(laneID % 64 == 0) {
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 6] = input7filter1;

        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 98 + (laneID / 64) * outputHeight * outputWidth + 6] = input7filter2;
    }
}
