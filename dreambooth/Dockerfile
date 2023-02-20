FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Bitsandbytes needs the softlink
RUN ln -s /usr/local/cuda/lib64/libcudart.so.11.8.89 /usr/local/cuda/lib64/libcudart.so

RUN apt-get update && apt-get install -y \
    git \
    libjpeg-dev \
    python3-pip \
    wget \
    zlib1g-dev

RUN mkdir -p /app
WORKDIR /app
ADD requirements.txt /app
RUN pip install -r requirements.txt

WORKDIR /dreambooth
RUN wget \
    https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py \
    https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py
