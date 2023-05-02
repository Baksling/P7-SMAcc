# SMACC: statistical model checking Accelerated

How to run the smacc test suite.

## Hardware
Tested on ubuntu 22.04 with cuda toolkit version 11.8 (later version should work)
The host machine should be equipped with a Nvidia GPU, prefferable from the turing line-up or later.
The host needs docker installed.
The host machine needs to have the Nvidia Container Toolkit installed to be able to pass the GPU into the container.

To get closer to optimal results, we suggest opening the dockerfile and changing the BLOCKS environment variable, although the default will run.
We have tested it on a Nvidia Tesla T4, RTX 3070 and A100, where we found good results with a block configuration of 40, 46 and 64 respectivly.
Empirically, changing the BLOCKS parameter to the number of SMs seems to give okay results.

## How to run

To run the experiments, run the "run_full.sh" script in order to run the full test suite, as seen in the paper (takes around 18-24 hours)
Alternatively, run the "run_lite.sh" srcipt in order to run a shorter experiment suite (takes around 5-15 minutes).

Example:
sudo bash run_lite.sh

Please make sure the dockerfile and the run files in the same folder.

## Results
After the results have concluded, a folder named "smacc_output" should have been created in the current directory.
In the folder, a bunch of .tsv files and .png files are created. 
The png files are the plots and tables seen in paper, while the .tsv files are the raw data used to generate the plots.
the file all_results.tsv is a file with all the results contained in one.

The plots show the number of components used in the model on the x axis, with time on y axis.
The comparison table shows the fastest processing time on the CPU and GPU and the performance ratio between the two (GPU_time / CPU_time)
The power ratio (from the paper) is calculated by taking the speedup times CPU power divided by GPU power ((speedup * CPU_wattage) / GPU_wattage). 