from cfg import graph_attention
from ray.tune import grid_search
from models.ray_graph import RayObsGraph
from torch_geometric.nn import GATConv

models = [
    # Base model
    {},
    # Framestack
    # {"num_framestacks": 4},
    # LSTM
    {"use_lstm": True, "max_seq_len": 32, "lstm_cell_size": 256},
    # Ours
    {
        "max_seq_len": 32,
        "custom_model": RayObsGraph,
        "custom_model_config": {
            "graph_size": 32,
            "gcn_output_size": 256,
            "gcn_hidden_size": 256,
        },
    },
    # Ours with attention
    {
        "max_seq_len": 32,
        "custom_model": RayObsGraph,
        "custom_model_config": {
            "graph_size": 32,
            "gcn_output_size": 256,
            "gcn_hidden_size": 256,
            "gcn_conv_type": GATConv,
            "gcn_num_attn_heads": 8,
        },
    },
]


CFG = graph_attention.CFG
CFG["ray"]["model"] = grid_search(models)

CFG["tune"] = CFG.get("tune", {})
CFG["tune"].update(
    {
        "goal_metric": {"metric": "episode_reward_mean", "mode": "max"},
        "stop": {"info/num_steps_trained": 10e6},
    },
)
