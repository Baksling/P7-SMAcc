#!/bin/bash/

rm -f -r ./smacc_output/
mkdir ./smacc_output/
sudo docker image rm smacc 2> /dev/null
sudo docker build -t smacc --no-cache --build-arg USER=smacc_user --build-arg UID=1001 .
sudo docker run --rm -it -e LITE=0 -e DEVICE=ALL -v ./smacc_output/:/Smacc/output/ --gpus=all smacc 