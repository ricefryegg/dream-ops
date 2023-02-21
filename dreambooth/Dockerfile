FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive 

# Bitsandbytes needs the softlink
RUN ln -s /usr/local/cuda/lib64/libcudart.so.11.8.89 /usr/local/cuda/lib64/libcudart.so

RUN apt-get update && apt-get install -y \
    git \
    libjpeg-dev \
    python3-opencv \
    python3-pip \
    wget \
    zlib1g-dev

# avoid rebuilding system package layer everytime
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
COPY --from=base / /

ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /dreambooth
WORKDIR /dreambooth
