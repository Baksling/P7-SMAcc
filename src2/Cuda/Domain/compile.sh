#!/bin/bash
# nvcc script
nvcc main.cu ./Simulator/*.cu ./UPPAALTreeParser/*.cu ./UPPAALTreeParser/*.cpp ./Domain/*.cu ./Visitors/*.cu ./Domain/expressions/*.cu -rdc=true -o ./a.out