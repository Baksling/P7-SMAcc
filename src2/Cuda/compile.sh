#!/bin/bash
# nvcc script
nvcc main.cu ./Simulator/*.cu ./Simulator/writers/*.cu ./UPPAALTreeParser/*.cu ./UPPAALTreeParser/*.cpp ./Domain/*.cu ./Visitors/*.cu ./Domain/expressions/*.cu -rdc=true -dopt=on -O3  -o ./a.out