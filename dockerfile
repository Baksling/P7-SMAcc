FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEVICE=ALL
ENV TIMEOUT=3600
ENV BLOCKS=40,256
ENV LITE=0
ENV N_SAMPLES=10

RUN apt-get update -y && apt upgrade -y
RUN apt-get install python3 python3-pip -y
RUN apt-get install git -y

# ARG USER
# ARG UID

# RUN adduser --uid $UID --ingroup dip --gecos "" --disabled-password $USER 
# RUN echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# ENV HOME="/home/$USER"

WORKDIR /Smacc

RUN git clone https://github.com/Baksling/P7-SMAcc.git .
RUN pip install -r /Smacc/src2/requirements.txt

RUN mkdir /Smacc/output
WORKDIR /Smacc/src2/Cuda/
RUN bash ./compile.sh
RUN cp /Smacc/src2/Cuda/kernal.cu /Smacc/kernal.cu 
WORKDIR /Smacc

CMD python3 /Smacc/src2/Analysis/auto_test_runner.py \
    -p /Smacc/src2/Cuda/c.out -d ${DEVICE} \
    -m /Smacc/src2/Cuda/UPPAALXMLParser/XmlFiles/ \
    -b ${BLOCKS} -o /Smacc/output/ \ 
    --timeout ${TIMEOUT} --saveplots /Smacc/output/ --lite ${LITE} \
    --n_samples ${N_SAMPLES}
