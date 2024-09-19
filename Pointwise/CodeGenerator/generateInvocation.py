first1 = True
first2 = True
first3 = True
lastBatch = 0
lastHeight = 0

with open('parameters.txt','r') as r:
    with open('invocation.txt', 'w') as w:
        while True:
            text = r.readline().strip()
            if not text:
                break
            slist = text.split(' ')
            ilist = []
            for i in slist:
                ilist.append(int(i))
            if ilist[3] != lastBatch:
                lastBatch = ilist[3]
                if first1 == True:
                    first1 = False
                    w.write('\t'+'if(inputBatchNumber == '+slist[3]+') {'+'\n')
                else:
                    w.write('\t\t\t'+'}'+'\n')
                    w.write('\t\t'+'}'+'\n')
                    w.write('\t'+'} else if(inputBatchNumber == '+slist[3]+') {'+'\n')
                first2 = True
                first3 = True
            if ilist[1] != lastHeight:
                lastHeight = ilist[1]
                if first2 == True:
                    first2 = False
                    w.write('\t\t'+'if(inputHeight == '+slist[1]+') {'+'\n')
                else:
                    w.write('\t\t\t'+'}'+'\n')
                    w.write('\t\t'+'} else if(inputHeight == '+slist[1]+') {'+'\n')
                first3 = True
            if first3 == True:
                first3 = False
                w.write('\t\t\t'+'if(inputChannel == '+slist[0]+' && outputChannel == '+slist[2]+') {'+'\n')
            else:
                w.write('\t\t\t'+'} else if(inputChannel == '+slist[0]+' && outputChannel == '+slist[2]+') {'+'\n')
            w.write('\t\t\t\t'+'dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / ('+slist[4]+' * '+slist[5]+' * '+slist[6]+'));'+'\n')
            w.write('\t\t\t\t'+'dim3 blockSize('+slist[4]+' * 64);'+'\n')
            w.write('\t\t\t\t'+'hipEventRecord(start);'+'\n')
            w.write('\t\t\t\t'+'InputBatch_'+slist[3]+'_Input_'+slist[1]+'x'+slist[1]+'_InChannel_'+slist[0]+'_OutChannel_'+slist[2]+'<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,'+'\n')
            w.write('\t\t\t\t\t'+'inputBatchNumber, inputChannel, inputHeight, inputWidth,'+'\n')
            w.write('\t\t\t\t\t'+'filterOutChannel, filterInChannel, filterHeight, filterWidth,'+'\n')
            w.write('\t\t\t\t\t'+'outputBatchNumber, outputChannel, outputHeight, outputWidth);'+'\n')
            w.write('\t\t\t\t'+'hipEventRecord(stop);'+'\n')
            w.write('\t\t'+'hipEventSynchronize(stop);'+'\n')
            w.write('\t\t'+'hipEventElapsedTime(&elapsedTime, start, stop);'+'\n')
            w.write('\t\t'+'kernelTime = elapsedTime;'+'\n')
        w.write('\t\t\t'+'}'+'\n')
        w.write('\t\t'+'}'+'\n')
        w.write('\t'+'}'+'\n')
