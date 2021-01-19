import multiprocessing
import numpy as np
import networkx as nx
import ray
from ray.rllib.agents import ppo
from server.render import app

from rayenv import NavEnv


class GraphNavEnv(NavEnv):
    graph = nx.Graph()
    node_ctr = 0
    max_edge_dist = 0.3

    def add_node(self, obs, info):
        '''Add a node using the current agent position
        and observation'''
        edges = []
        for n_idx in self.graph.nodes():
            cmp_data = self.graph.nodes[n_idx]['data']
            if np.linalg.norm(cmp_data['pose']['r'] - info['pose2d']['r']) < self.max_edge_dist:
                edges += [(self.node_ctr, n_idx)]
        
        node_data = {'pose': info['pose2d'], 'obs': obs}
        self.graph.add_node(self.node_ctr, data=node_data)
        self.graph.add_edges_from(edges)
        self.node_ctr += 1

    def step(self, action):
        obs, reward, done, info = super().step(action) 
        print('done is', done)
        self.add_node(obs, info)
        return obs, reward, done, info

    def get_info(self, obs):
        info = super().get_info(obs)
        agent_state = self._env.sim.agents[0].state
        pose3d = {'r': agent_state.position, 'q': agent_state.rotation}
        info['pose3d'] = pose3d
        # In habitat, y is up vector
        info['pose2d'] = {
            'r': np.array((agent_state.position[0], agent_state.position[2])),
            'theta': agent_state.rotation.copy()
        }
        return info
        
    def reset(self):
        obs = super().reset()
        # Per-episode graph, place root node
        self.graph.clear()
        self.node_ctr = 0
        info = self.get_info(obs)
        self.add_node(obs, info)
        return obs
