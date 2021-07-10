# Graph Convolutional Memory for Reinforcement Learning
This is the code used for the paper [Graph Convolutional Memory for Reinforcement Learning](https://arxiv.org/abs/2106.14117). This repo is intended to aid in reproducability of the paper. If you are interested in using graph convolutional memory in your own project, I suggest you use my `graph-conv-memory` library available [here](https://github.com/smorad/graph-conv-memory).  

## Description
Graph convolutional memory (GCM) is graph-structured memory that may be applied to reinforcement learning to solve POMDPs, replacing LSTMs or attention mechanisms.

## Quickstart
If you are interested in apply GCM for your problem, you must install dependencies `torch` and `torch_geometric`. If you are using `ray rllib` to train, use the `RayObsGraph` model as so (running from the project root directory):

```
import torch
import torch_geometric

from ray import tune
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole

from models.ray_graph import RayObsGraph
from models.edge_selectors.temporal import TemporalBackedge

our_gnn = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGraphConv(32, 32), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGraphConv(32, 32), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
ray_cfg = {
   "env": StatelessCartPole, # Replace this with your desired env
   "framework": "torch",
   "model": {
      "custom_model": RayObsGraph,
      "custom_model_config": {
         "gnn_input_size": 32,
         "gnn_output_size": 32,
         "gnn": our_gnn,
         "edge_selectors": TemporalBackedge([1])
      }
   }
}
tune.run("PPO", config=ray_cfg)
```

If you are not using `ray rllib`, use the model like so:

```
import torch
import torch_geometric
from models.gcm import DenseGCM
from models.edge_selectors.temporal import TemporalBackedge

our_gnn = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGraphConv(YOUR_OBS_SIZE, 32), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGraphConv(32, 32), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
gcm = DenseGCM(our_gnn, edge_selectors=TemporalBackedge([1]), graph_size=128)

# Create initial state
edges = torch.zeros(
    (1, 128, 128), dtype=torch.float
)
nodes = torch.zeros((1, 128, YOUR_OBS_SIZE))
weights = torch.zeros(
    (1, 128, 128), dtype=torch.float
)
num_nodes = torch.tensor([0], dtype=torch.long)
m_t = [nodes, edges, weights, num_nodes]

for t in train_timestep:
   state, m_t = gcm(obs[t], m_t)
   # Do what you will with the state
   # likely you want to use it to get action/value estimate
   action_logits = logits(state)
   state_value = vf(state)
```
See `src/models/edge_selectors` for different kinds of priors.
     

## Full Install
Getting CUDA/python/conda/habitat/ray working together is a project in itself. We run everything in docker to make our setup reproduceable anywhere. The full install will install all our code as used for our various experiments. You only need to do this if you are rerunning our experiments.

### Host Setup
We have tested everything using `Docker version 20.10.2, build 2291f61`, `NVidia Driver Version: 460.27.04`, and `CUDA Version: 11.2` so if you run into issues try using these versions on your host. After installing CUDA, follow the [NVidia guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to install docker and nvidia-docker2.

Unfortunately CUDA is required for Habitat, so you must follow this step if you want to run the navigation experiment. Once the host is set up, continue with the docker container creation. We also do not store the scene models in the docker container, as they are huge and it's not legal for us to package them in the container. Download the `habitat` task of the matterport3D dataset as shown [here](https://github.com/facebookresearch/habitat-lab#data). Then extract it, and use the extracted directory as `SCENE_DATASET_PATH` in the container setup.

### Docker Container Setup
```
#!/bin/bash

cd vnav
# Build the image -- this takes a while, get yourself a coffee
docker build docker -t ray_habitat:latest

# Launch a container
# Make sure you fill out SCENE_DATASET_PATH to where you've
# stored the mp3d scene_datasets (navigation problem only)
# We cannot share these, you need to sign a waiver with mp3d first
export SCENE_DATASET_PATH=/path_to/scene_datasets
# port description:
# 8265 ray
# 5000 navigation renders
# 5050 visdom
# 6006 tensorboard
docker run \
    --gpus all \
    --shm-size 32g \
    -p 8299:8265 \
    -p 5000:5000 \
    -p 5050:5050 \
    -p 6099:6006 \
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

### Rerunning Experiments
You can rerun our experiments with the following commands:
```
python3 src/start.py src/cfg/cartpole.py # Cartpole experiment
python3 src/start.py src/cfg/memory.py # Memory experiment
python3 src/start.py src/cfg/nav.py # Navigation experiment
```

Which will populate `$HOME/ray_results/<EXPERIMENT_ID>` with tensorboard data as well as CSV and JSON files containing the training info.
