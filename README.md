# Graph Convolution Memory for Reinforcement Learning

## Description
Graph convolutional memory (GCM) is graph-structured memory that may be applied to reinforcement learning to solve POMDPs, replacing LSTMs or attention mechanisms.

## Quickstart
If you are interested in apply GCM for your problem, you must `torch` and `torch_geometric` as dependencies. If you are using `ray rllib` to train, use the `RayObsGraph` model as so:

```
from ray import tune
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
   ...
   "framework": "torch",
   "model": : {
      "custom_model": RayObsGraph,
      "custom_model_config": {
         "gnn_input_size": 32,
         "gnn_output_size": 32,
         "gnn": our_gnn,
         "edge_selectors": TemporalBackedge([1])
      }
   }
}
tune.run("PPO", ray_cfg)
```

If you are not using `ray rllib`, use the model like so:

```
from models.gam import DenseGAM
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
gam = DenseGAM(our_gnn, edge_selectors=TemporalBackedge([1]), graph_size=128)

# Create initial state
edges = torch.zeros(
    (1, 128, 128), dtype=dtype
)
nodes = torch.zeros((1, 128, YOUR_OBS_INPUT_SIZE))
weights = torch.zeros(
    (1, 128, 128), dtype=dtype
)
num_nodes = torch.tensor(0, dtype=torch.long).reshape(1,1)
m_t = [nodes, edges, weights, num_nodes]

for t in train_timestep:
   state, m_t = gam(obs[t], m_t)
   # Do what you will with the state
   action_logits = logits(state)
   state_value = vf(state)
```
See `src/models/edge_selectors` for different kinds of priors.
     

## Full Install
Getting CUDA/python/conda/habitat/ray working together is a project in itself. We run everything in docker to make our setup reproduceable anywhere. The full install will install all our code as used for our various experiments. You only need to do this if you are rerunning our experiments.

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

