# Graph Convolution Memory for Reinforcement Learning

## Install
Getting CUDA/python/conda/habitat/ray working together is a project in itself. We run everything in docker to make our setup reproduceable anywhere.

### Host Setup
We have tested everything using `Docker version 20.10.2, build 2291f61`, `NVidia Driver Version: 460.27.04`, and `CUDA Version: 11.2` so if you run into issues try using these versions on your host. After installing CUDA, follow the [NVidia guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to install docker and nvidia-docker2.

Unfortunately CUDA is required for Habitat, so you must follow this step. Once the host is set up, continue with the docker container creation. We also do not store the scene models in the docker container, as they are huge and it's not legal for us to package them in the container. Download the `habitat` task of the matterport3D dataset as shown [here](https://github.com/facebookresearch/habitat-lab#data). Then extract it, and use the extracted directory as `SCENE_DATASET_PATH` in the container setup.

### Docker Container Setup
```
#!/bin/bash

cd vnav
# Build the image -- this takes a while, get yourself a coffee
docker build docker -t ray_habitat:latest

# Launch a container
# Make sure you fill out SCENE_DATASET_PATH to where you've
# stored the mp3d scene_datasets
export SCENE_DATASET_PATH=/local/scratch/sm2558/scene_datasets
docker run \
    --gpus all \
    --shm-size 32g \
    -p 8299:8265 # Ray \
    -p 5000:5000 # Render server \
    -p 5050:5050 # Visdom \
    -p 6099:6006 # Tensorboard \
    -v ${SCENE_DATASET_PATH}:/root/scene_datasets \
    -ti ray_habitat:latest bash

# Now we should be in the container
```

## Execution
Once in the container, make sure the repo is up to date, then run!

```
#!/bin/bash

# Ensure CUDA is working as expected
nvidia-smi
# Make sure to update source repo in the container
cd /root/vnav
git pull
# Run!
python3 src/start.py src/cfg/memory.py
```

