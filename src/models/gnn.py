import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing


class EgoGNNModel(MessagePassing):
    """
    The model takes as input a euclidean graph. Each node contains a pose relative to
    the current pose and the observation from said pose.

    The output of the model is logits for discrete actions, and logit for creating
    a node from the current observation
    """

    def __init__(self, F_in, F_out):
        super(EdgeConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.mlp = Sequential(Linear(2 * F_in, F_out), ReLU(), Linear(F_out, F_out))

    def forward(self, x, edge_index):
        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)  # shape [N, F_out]

    def message(self, x_i, x_j):
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # shape [E, 2 * F_in]
        return self.mlp(edge_features)  # shape [E, F_out]
