import habitat
import numpy as np
from gym import spaces

import detectron2
from detectron2.utils.logger import setup_logger
import cv2
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch

setup_logger()

# Ball is coco class 32


@habitat.registry.register_sensor(name="NNSemanticSensor")
class NNSemanticSensor(habitat.Sensor):
    def init_rcnn(self):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        self.predictor = DefaultPredictor(cfg)

    def __init__(self, sim, config, **kwargs):
        self._sim = sim
        self.config = config
        self.init_rcnn()
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "nn_semantic_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return habitat.SensorTypes.SEMANTIC

    def _get_observation_space(self, *args, **kwargs):
        # We don't have access to the full config
        # so just make our dims the same as the color camera
        rgb = self._sim.config.agents[0].sensor_specifications[0]
        assert rgb.uuid == "rgb"
        dims = rgb.resolution
        return spaces.Box(
            low=0,
            high=1,
            shape=(80, *dims),
            dtype=np.float32,
        )

    def get_observation(self, observations, *args, episode, **kwargs):
        rgb = observations["rgb"]
        # Format is:
        # Each semantic category gets a channel
        # Instances share semantic channels
        return self.predictor(rgb)["sem_seg"]
