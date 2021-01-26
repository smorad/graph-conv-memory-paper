import habitat
import numpy as np
from gym import spaces
import habitat_sim
from habitat.core.simulator import SemanticSensor, VisualObservation, Sensor
from typing import List, Any, Union, Optional, cast, Dict

@habitat.registry.register_sensor
class SemanticMaskSensor(SemanticSensor):
    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, sim, config):
        import pdb; pdb.set_trace()
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        self.sim = sim
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.config.HEIGHT, self.config.WIDTH),
            dtype=np.uint32,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)
        layers = np.zeros(len(self.sim.semantic_label_lookup.values()), *obs.shape, dtype=int)
        # 1 channel, pixel == instance_id => n channel, channel == obj_id, pixel == 1
        for inst_id, obj_id in self.sim.semantic_label_lookup.items():
            idxs = np.where(obs.flat == inst_id)
            layers[obj_id].flat[idxs] = 1
        return layers


def check_sim_obs(obs: np.ndarray, sensor: Sensor) -> None:
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )
