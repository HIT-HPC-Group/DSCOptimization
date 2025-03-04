/*
Pointwise Convolution Kernel
InputBatch_32_Input_112x112_InChannel_32_OutChannel_16

Grid:
    gridDim.x = (32 * 16 * 112 * 112) / (4 * 14 * 16);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 14
outputChannelPerWarp = 16
channelGroupSize = 4
horizontalRepeat = 4
verticalRepeat = 1

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 14 * 16 output data.
Each warp is responsible for generating 14 * 16 output data.
*/

__global__ void InputBatch_32_Input_112x112_InChannel_32_OutChannel_16(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[4 * 14 * 4];
    __shared__ float inputSharedBuffer2[4 * 14 * 4];

    __shared__ float filterSharedBuffer1[1 * 16 * 4];
    __shared__ float filterSharedBuffer2[1 * 16 * 4];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0, inputOperand8 = 0, inputOperand9 = 0, inputOperand10 = 0;
    float inputOperand11 = 0, inputOperand12 = 0, inputOperand13 = 0, inputOperand14 = 0;
    float filterOperand1 = 0;

    // to hold intermediate result
    float input1filter1 = 0;

    float input2filter1 = 0;

    float input3filter1 = 0;

    float input4filter1 = 0;

    float input5filter1 = 0;

    float input6filter1 = 0;

    float input7filter1 = 0;

    float input8filter1 = 0;

    float input9filter1 = 0;

    float input10filter1 = 0;

    float input11filter1 = 0;

    float input12filter1 = 0;

    float input13filter1 = 0;

    float input14filter1 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 224 * 401408 + (blockIdx.x % 224) / 8 * 448 + (blockIdx.x % 8) / 1 * 14;
    if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
        inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 56 * 12544 + ((threadIdx.x + 0 * 256) % 56) / 14 * 112 + (threadIdx.x + 0 * 256) % 14];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 1) * 512;
    if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
        filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 32 + ((threadIdx.x + 0 * 256) % 4)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 4) / (2 * 4); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 112 * 112 * 4;
        if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 56 * 12544 + ((threadIdx.x + 0 * 256) % 56) / 14 * 112 + (threadIdx.x + 0 * 256) % 14];
        }

        blockLoadFilterStartIdx += 4;
        if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 32 + ((threadIdx.x + 0 * 256) % 4)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 6];
        inputOperand8 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 7];
        inputOperand9 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 8];
        inputOperand10 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 9];

        inputOperand11 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 10];
        inputOperand12 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 11];
        inputOperand13 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 12];
        inputOperand14 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 13];

        filterOperand1 = filterSharedBuffer1[(warpID % 1) * 64 + laneID + 0 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;

        input2filter1 += inputOperand2 * filterOperand1;

        input3filter1 += inputOperand3 * filterOperand1;

        input4filter1 += inputOperand4 * filterOperand1;

        input5filter1 += inputOperand5 * filterOperand1;

        input6filter1 += inputOperand6 * filterOperand1;

        input7filter1 += inputOperand7 * filterOperand1;

        input8filter1 += inputOperand8 * filterOperand1;

        input9filter1 += inputOperand9 * filterOperand1;

        input10filter1 += inputOperand10 * filterOperand1;

        input11filter1 += inputOperand11 * filterOperand1;

        input12filter1 += inputOperand12 * filterOperand1;

        input13filter1 += inputOperand13 * filterOperand1;

        input14filter1 += inputOperand14 * filterOperand1;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
            inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        }

        if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
            filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 112 * 112 * 4;
        if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 56 * 12544 + ((threadIdx.x + 0 * 256) % 56) / 14 * 112 + (threadIdx.x + 0 * 256) % 14];
        }

        blockLoadFilterStartIdx += 4;
        if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 32 + ((threadIdx.x + 0 * 256) % 4)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 6];
        inputOperand8 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 7];
        inputOperand9 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 8];
        inputOperand10 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 9];

        inputOperand11 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 10];
        inputOperand12 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 11];
        inputOperand13 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 12];
        inputOperand14 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 13];

        filterOperand1 = filterSharedBuffer2[(warpID % 1) * 64 + laneID + 0 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;

        input2filter1 += inputOperand2 * filterOperand1;

        input3filter1 += inputOperand3 * filterOperand1;

        input4filter1 += inputOperand4 * filterOperand1;

        input5filter1 += inputOperand5 * filterOperand1;

        input6filter1 += inputOperand6 * filterOperand1;

        input7filter1 += inputOperand7 * filterOperand1;

        input8filter1 += inputOperand8 * filterOperand1;

        input9filter1 += inputOperand9 * filterOperand1;

        input10filter1 += inputOperand10 * filterOperand1;

        input11filter1 += inputOperand11 * filterOperand1;

        input12filter1 += inputOperand12 * filterOperand1;

        input13filter1 += inputOperand13 * filterOperand1;

        input14filter1 += inputOperand14 * filterOperand1;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
            inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        }

        if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
            filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        }
        __syncthreads();
    }
    // load next group of Cnum channels
    blockLoadInputStartIdx += 112 * 112 * 4;
    if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 56 * 12544 + ((threadIdx.x + 0 * 256) % 56) / 14 * 112 + (threadIdx.x + 0 * 256) % 14];
    }

    blockLoadFilterStartIdx += 4;
    if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 32 + ((threadIdx.x + 0 * 256) % 4)];
    }

    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 6];
    inputOperand8 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 7];
    inputOperand9 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 8];
    inputOperand10 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 9];

    inputOperand11 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 10];
    inputOperand12 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 11];
    inputOperand13 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 12];
    inputOperand14 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 13];

    filterOperand1 = filterSharedBuffer1[(warpID % 1) * 64 + laneID + 0 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;

    input2filter1 += inputOperand2 * filterOperand1;

    input3filter1 += inputOperand3 * filterOperand1;

    input4filter1 += inputOperand4 * filterOperand1;

    input5filter1 += inputOperand5 * filterOperand1;

    input6filter1 += inputOperand6 * filterOperand1;

    input7filter1 += inputOperand7 * filterOperand1;

    input8filter1 += inputOperand8 * filterOperand1;

    input9filter1 += inputOperand9 * filterOperand1;

    input10filter1 += inputOperand10 * filterOperand1;

    input11filter1 += inputOperand11 * filterOperand1;

    input12filter1 += inputOperand12 * filterOperand1;

    input13filter1 += inputOperand13 * filterOperand1;

    input14filter1 += inputOperand14 * filterOperand1;

    // Copy Temp Registers to shared buffer 2
    if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
    }

    if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
    }
    __syncthreads();

    // Exchange shared buffer 1 and shared buffer 2 and repeat
    // Copy operands from shared buffer 2 into Operands Registers
    inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 0];
    inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 1];
    inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 2];
    inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 3];
    inputOperand5 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 4];

    inputOperand6 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 5];
    inputOperand7 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 6];
    inputOperand8 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 7];
    inputOperand9 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 8];
    inputOperand10 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 9];

    inputOperand11 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 10];
    inputOperand12 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 11];
    inputOperand13 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 12];
    inputOperand14 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 13];

    filterOperand1 = filterSharedBuffer2[(warpID % 1) * 64 + laneID + 0 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;

    input2filter1 += inputOperand2 * filterOperand1;

    input3filter1 += inputOperand3 * filterOperand1;

    input4filter1 += inputOperand4 * filterOperand1;

    input5filter1 += inputOperand5 * filterOperand1;

    input6filter1 += inputOperand6 * filterOperand1;

    input7filter1 += inputOperand7 * filterOperand1;

    input8filter1 += inputOperand8 * filterOperand1;

    input9filter1 += inputOperand9 * filterOperand1;

    input10filter1 += inputOperand10 * filterOperand1;

    input11filter1 += inputOperand11 * filterOperand1;

    input12filter1 += inputOperand12 * filterOperand1;

    input13filter1 += inputOperand13 * filterOperand1;

    input14filter1 += inputOperand14 * filterOperand1;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (4 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 4);

        input2filter1 += __shfl_down(input2filter1, offset, 4);

        input3filter1 += __shfl_down(input3filter1, offset, 4);

        input4filter1 += __shfl_down(input4filter1, offset, 4);

        input5filter1 += __shfl_down(input5filter1, offset, 4);

        input6filter1 += __shfl_down(input6filter1, offset, 4);

        input7filter1 += __shfl_down(input7filter1, offset, 4);

        input8filter1 += __shfl_down(input8filter1, offset, 4);

        input9filter1 += __shfl_down(input9filter1, offset, 4);

        input10filter1 += __shfl_down(input10filter1, offset, 4);

        input11filter1 += __shfl_down(input11filter1, offset, 4);

        input12filter1 += __shfl_down(input12filter1, offset, 4);

        input13filter1 += __shfl_down(input13filter1, offset, 4);

        input14filter1 += __shfl_down(input14filter1, offset, 4);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 224 * 200704 + (blockIdx.x % 224) / 8 * 448 + (blockIdx.x % 8) / 1 * 14 + (blockIdx.x % 1) * 200704;

    if(laneID % 4 == 0) {
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 7] = input8filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 8] = input9filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 9] = input10filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 10] = input11filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 11] = input12filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 12] = input13filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 13] = input14filter1;
    }
}
