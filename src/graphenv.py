import numpy as np
import networkx as nx
import ray
from ray.rllib.agents import ppo

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


if __name__ == '__main__':
    ray.init(dashboard_host='0.0.0.0', local_mode=True)
    hab_cfg_path = "/root/vnav/cfg/objectnav_mp3d.yaml" 
    ray_cfg = {'env_config': {'path': hab_cfg_path}, 
            # For debug
            'num_workers': 0,
            'num_gpus': 1,
            # For prod
            #'num_gpus_per_worker': 0.5,
            'framework': 'torch'}
    trainer = ppo.PPOTrainer(env=GraphNavEnv, config=ray_cfg)
    # Can access envs here: trainer.workers.local_worker().env
    #ray.util.pdb.set_trace()
    while True:
        print(trainer.train())
