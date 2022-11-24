#!/bin/bash
# nvcc script
nvcc main.cu ./Simulator/*.cu ./Simulator/writers/*.cu ./UPPAALTreeParser/*.cu ./UPPAALTreeParser/*.cpp ./Domain/*.cu ./Visitors/*.cu ./Domain/expressions/*.cu -rdc=true -lineinfo -o ./b.out