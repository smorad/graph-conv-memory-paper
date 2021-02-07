from ray.rllib.agents.callbacks import DefaultCallbacks
import ray


class CustomMetrics(DefaultCallbacks):

    INFO_METRICS = ["distance_to_goal", "success", "spl", "softspl"]

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        info = episode.last_info_for()
        episode.custom_metrics.update({k: info[k] for k in self.INFO_METRICS})
