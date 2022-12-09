#!/bin/bash
# nvcc script
cd ../Cuda/; nvcc ./Simulator2/main.cu ./Simulator2/simulation_runner.cu ./Simulator2/common/thread_pool.cu ./Simulator2/allocations/*.cpp ./UPPAALXMLParser/*.cpp ./Simulator2/results/*.cu ./Simulator2/visitors/*.cpp -O3 --dopt=on --use_fast_math -Xptxas -O3 -o c.out -lcuda -lnvrtc
python3 ../Analysis/file_assembler.py ./Simulator2/
mv ./Simulator2/kernal.cu ./