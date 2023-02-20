docker run --rm -it \
    --gpus all --ipc=host \
    -v ./dreambooth:/dreambooth \
    dreambooth:cu11.8 bash
