FROM nvidia/cudagl:11.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git python3 python3-pip vim
RUN git clone https://github.com/facebookresearch/habitat-sim /root/habitat-sim
RUN git clone https://github.com/facebookresearch/habitat-lab /root/habitat-lab

# Build habitat-sim
RUN apt install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev libbullet-dev cmake ninja-build g++
# Make image smaller by not caching downloaded pip pkgs
ARG PIP_NO_CACHE_DIR=1
RUN cd /root/habitat-sim && pip3 install -r requirements.txt && python3 setup.py install --headless  --with-cuda --bullet

# Install pytorch for example, and ensure sim works with all our required pkgs
ARG TORCH=1.8.1
ARG TORCHVISION=0.9.1
ARG CUDA=cu111
# Pytorch and torch_geometric w/ deps
RUN pip3 install torch==${TORCH}+${CUDA} \
    torchvision==${TORCHVISION}+${CUDA} \ 
    -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    torch_geometric
#RUN cd /root/habitat-sim && python3 examples/example.py

# Build habitat-lab and install habitat-baselines deps
RUN cd /root/habitat-lab && pip3 install -e . \
    && pip3 install lmdb webdataset ifcfg

# Ray rllib
RUN pip3 install gputil https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

# Detectron
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# Install our deps; webserver for viewing observation and faster PIL
RUN pip3 uninstall -y pillow && pip3 install flask pillow-simd Flask-SocketIO visdom dnc

# Copy token for gitlab clone
RUN echo "6RJQmn5AezFsn7Nsqu7N" > /root/token && git clone https://oauth2:$(cat /root/token)@gitlab.developers.cam.ac.uk/cst/prorok-lab/vnav /root/vnav

# Download test and object-goal navigation episodes
RUN apt install -y unzip && \
    # test
    curl http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip --output habitat-test-scenes.zip \
    && unzip habitat-test-scenes.zip -d /root/habitat-lab \
    # objectnav
	&& curl https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip \
		--output objectnav_mp3d_v1.zip \ 
	&& mkdir -p /root/habitat-lab/data/datasets/objectnav/mp3d/v1/ \
	&& unzip objectnav_mp3d_v1.zip -d /root/habitat-lab/data/datasets/objectnav/mp3d/v1/ \
	&& rm objectnav_mp3d_v1.zip

# Link matterport models to mounted docker volume
RUN ln -s /root/scene_datasets/mp3d /root/habitat-lab/data/scene_datasets

# Dev tools
RUN pip3 install pre-commit && cd /root/vnav && pre-commit && pre-commit autoupdate && pre-commit install
