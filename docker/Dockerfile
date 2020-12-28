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
ARG TORCH=1.7.0
ARG TORCHVISION=0.8.1
ARG CUDA=cu110
# Pytorch and torch_geometric w/ deps
RUN pip3 install torch==${TORCH}+${CUDA} \
    torchvision==${TORCHVISION}+${CUDA} \ 
    -f https://download.pytorch.org/whl/torch_stable.html \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    torch_geometric
#RUN cd /root/habitat-sim && python3 examples/example.py

# Build habitat-lab
RUN cd /root/habitat-lab && pip3 install -e .

# Ray rllib
RUN pip3 install ray[rllib]

# Copy token for gitlab clone
RUN echo "6RJQmn5AezFsn7Nsqu7N" > /root/token && git clone https://oauth2:$(cat /root/token)@gitlab.developers.cam.ac.uk/cst/prorok-lab/vnav /root/vnav
