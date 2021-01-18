Usage:
# Make sure you fill out SCENE_DATASET_PATH to where you've
# stored the mp3d scene_datasets
export SCENE_DATASET_PATH=/local/scratch/sm2558/scene_datasets
docker run \
    --gpus all \
    --shm-size 32g \
    -p 8265:8265 \
    -v ${SCENE_DATASET_PATH}:/root/scene_datasets:ro \
    -ti docker_ray-habitat:latest bash

    --mount source=${SCENE_DATASET_PATH},destination=/root/scene_datasets,readonly \ 

# Make sure to update repo
cd /root/vnav
git pull
python3 src
