#!/bin/bash
# nvcc script
#This script is the bane of my fucking existens. This project is developed on windows, and run and tested on linux. Every time i transfer this script to linux, \r characters are added at the end.
#I do not wanna deal with this shit anymore, so now its just one line. Have fun understanding it!
cd ../Cuda/; nvcc ./main.cu ./simulation_runner.cu ./common/thread_pool.cu ./allocations/*.cpp ./UPPAALXMLParser/*.cpp ./results/*.cu ./visitors/*.cpp -O3 --dopt=on --use_fast_math -Xptxas -O3 -o c.out -lcuda -lnvrtc; python3 ../Analysis/kernal_assembler.py ./; echo "Please make sure 'kernal.cu' is in the same folder as 'c.out' when running."; echo "Use -h to open help menu."; echo "GLHF"