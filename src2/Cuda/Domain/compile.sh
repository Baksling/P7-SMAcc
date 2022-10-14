#!/bin/bash
nvcc ../Simulator/*.cu ./*.cu ../UPPAALTreeParser/*.cu ../UPPAALTreeParser/*.cpp ./UpdateExpressions/*.cu -o ./a.out -rdc=true
