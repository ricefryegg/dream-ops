docker run --rm -it \
    --hostname `hostname` \
    --gpus all --ipc=host \
    -v .:/dreambooth \
    -v "$(dirname "$(pwd)")/models:/models" \
    dreambooth:cu11.8 bash
