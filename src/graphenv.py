import multiprocessing
import os

import numpy as np
import networkx as nx
import ray
import cv2
from ray.rllib.agents import ppo
from habitat.utils.visualizations import maps
from habitat.utils.visualizations import utils

from server.render import RENDER_ROOT, CLIENT_LOCK
from rayenv import NavEnv


class GraphNavEnv(NavEnv):
    graph = nx.Graph()
    node_map = None
    node_ctr = 0
    max_edge_dist = 0.3

    def add_node(self, obs, info):
        """Add a node using the current agent position
        and observation"""
        edges = []
        for n_idx in self.graph.nodes():
            cmp_data = self.graph.nodes[n_idx]["data"]
            if (
                np.linalg.norm(cmp_data["pose"]["r"] - info["pose2d"]["r"])
                < self.max_edge_dist
            ):
                edges += [(self.node_ctr, n_idx)]

        if info["top_down_map"]:
            map_pose = info["top_down_map"]["agent_map_coord"]
            map_angle = info["top_down_map"]["agent_angle"]
        else:
            map_pose = None
            map_angle = None

        node_data = {
            "pose": info["pose2d"],
            "obs": obs,
            # pose tfed for visualization}
            "map_pose": map_pose,
            "map_angle": map_angle,
        }
        self.graph.add_node(self.node_ctr, data=node_data)
        self.graph.add_edges_from(edges)
        self.node_ctr += 1

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Only visualize if someone is viewing via webbrowser
        if CLIENT_LOCK.exists():
            if self.visualize_lvl >= 2 and info.get("top_down_map") is not None:
                self.add_node_to_map(info)
                self.emit_debug_graph(info)
            self.add_node(obs, info)
        return obs, reward, done, info

    def add_node_to_map(self, info):
        """Draw current position as a node to the node_map"""
        if self.node_map is None:
            self.node_map = maps.colorize_topdown_map(
                info["top_down_map"]["map"],
            )

        if not info["top_down_map"]:
            return

        pose = info["top_down_map"]["agent_map_coord"]
        cv_pose = (pose[1], pose[0])
        self.node_map = cv2.circle(
            self.node_map, cv_pose, radius=10, color=(0, 69, 255), thickness=3
        )

    def emit_debug_graph(self, info):
        img = self.node_map.copy()

        if img.shape[0] > img.shape[1]:
            img = np.rot90(img, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = img.shape
        top_down_height = self.hab_cfg.SIMULATOR.RGB_SENSOR.HEIGHT
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        img = cv2.resize(
            img,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )

        tmp_impath = f"{self.render_dir}/graph.jpg.buf"
        impath = f"{self.render_dir}/graph.jpg"
        _, buf = cv2.imencode(".jpg", img)
        buf.tofile(tmp_impath)
        # We do this so we don't accidentally load a half-written img
        os.replace(tmp_impath, impath)

    def get_info(self, obs):
        info = super().get_info(obs)
        agent_state = self._env.sim.agents[0].state
        pose3d = {"r": agent_state.position, "q": agent_state.rotation}
        info["pose3d"] = pose3d
        # In habitat, y is up vector
        info["pose2d"] = {
            "r": np.array((agent_state.position[0], agent_state.position[2])),
            "theta": agent_state.rotation.copy(),
        }
        return info

    def reset(self):
        obs = super().reset()
        # Per-episode graph, place root node
        self.graph.clear()
        self.node_ctr = 0
        info = self.get_info(obs)
        self.node_map = None
        self.add_node(obs, info)
        return obs
