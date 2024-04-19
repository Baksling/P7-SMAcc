# SMAcc: Statistical Model checking Accelerated

How to run the SMAcc test suite.

## Hardware
Tested on Ubuntu 22.04 with CUDA toolkit version 11.8 (later versions should work).
The host machine should be equipped with a NVIDIA GPU, prefferable from the Turing line-up or later.
The host needs to have Docker installed.
The host machine needs to have the NVIDIA Container Toolkit installed to be able to pass the GPU into the container.

To get closer to optimal results, we suggest opening the dockerfile and changing the BLOCKS environment variable, although the default will be able to run.
We have tested it on a NVIDIA Tesla T4, RTX 3070 and A100, where we found good results with a block configuration of 40, 46, and 64 respectivly (written as 40,256 ; 46,256 ; 64,256 respectivly).
Empirically, changing the BLOCKS parameter to the number of SMs on the GPU seems to give okay results.

## Run experiment suite

To run the experiments, run the "run_full.sh" script in order to run the full test suite, as seen in the paper (takes around 18-24 hours). In the paper, the experiments are shown as an average of 10 samples, however, this script only runs 1 sample. 

Alternatively, run the "run_lite.sh" script in order to run a shorter experiment suite (takes around 5-15 minutes).

The full suite includes all the models shown in the paper, while the run_lite suite only samples the smallest version of each model.

Example:
    sudo bash run_lite.sh

Please make sure the dockerfile and the run files in the same folder.

## Results
After the experiments have concluded, a folder named "smacc_output" should have been created in the current directory.
In the folder, a bunch of .tsv files and .png files are created. 
The png files are the plots and tables seen in paper, while the .tsv files are the raw data used to generate the plots.
the file "all_results.tsv" is a file with all the results contained in one.

The plots show the number of components used in the model on the x axis, with time on the y axis.
The comparison table shows the fastest processing time on the CPU and GPU and the performance ratio between the two (GPU_time / CPU_time).
The power ratio (from the paper) is calculated by taking the speedup multiplied with the CPU power divided by the GPU power ((speedup * CPU_wattage) / GPU_wattage). 

CPU power consumption is assumed to be 450W and GPU power consumption is assumed to be 250W, as those were the values for the hardware used in the paper. 


## Run from source
To compile from source, navigate to the /src2/Cuda/ folder and run the compile.sh script (you must be located in this folder). 
This will generate a c.out file and a kernal.cu file. 
The kernal.cu file must be located at your current path when calling the c.out file in order for JIT compilation to work. 
Call c.out -h to see the possible arguments.
All of the following commands assume you are located at src2/cuda/.

Example of running SMAcc on a single model:

    ./c.out -m ./UPPAALXMLParser/XmlFiles/agentBaseCovid_1000_10.0.xml -w c -b 40,256 -n 10240 -x 100t -u 1000 -z

Example of running auto tests script (lite version):

    python3 ../Analysis/auto_test_runner.py -p ./c.out -d ALL -m ./UPPAALXMLParser/XmlFiles -o ./output/ -t 100 --timeout 3600 -b 40,256 --saveplots ./output/ --lite 1


Example of running accuracy tests:

    python3 ../Analysis/integration_test.py -s ./c.out -f ./UPPAALXMLParser/XmlFiles/tests/ -y 1 -p 100 -a 1000000 -z 1 -op ./tests.tsv
