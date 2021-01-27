from scipy.spatial.transform import Rotation
import numpy as np
import networkx as nx
from typing import Type


class ObsNode:
    # Each node holds an observation
    def __init__(self, obs):
        self.obs = obs


class PoseEdge:
    # Each edge holds a 6DOF transform
    def __init__(self, pos: Type[np.array], rot: Type[Rotation]):
        self.pos = pos
        self.rot = rot


class PoseGraph(nx.Graph):
    id_ctr = 0

    def __init__(self):
        # Init world frame
        pass

    def add_node(self, node, **attr):
        node.id = self.id_ctr
        self.id_ctr += 1
        return super().add_node(node, attr)

    def add_nodes_from(nodes_for_adding, **attr):
        for n in nodes_for_adding:
            n.id = self.id_ctr
            self.id_ctr += 1
        return super().add_nodes_from(nodes_for_adding, attr)
