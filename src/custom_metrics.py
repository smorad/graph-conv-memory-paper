from ray.rllib.agents.callbacks import DefaultCallbacks
import ray


class CustomMetrics(DefaultCallbacks):

    INFO_METRICS = ["distance_to_goal", "success", "spl", "softspl"]

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        info = episode.last_info_for()
        episode.custom_metrics.update({k: info[k] for k in self.INFO_METRICS})


class VAEMetrics(DefaultCallbacks):
    def on_learn_on_batch(self, *, policy, train_batch, **kwargs) -> None:
        pass

    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        m = trainer.get_policy().model
        losses = {
            "vae_mse_loss": m.mse_loss.detach().item(),
            "vae_kld_loss": m.kld_loss.detach().item(),
            "vae_combined_loss": m.combined_loss.detach().item(),
        }
        result["custom_metrics"].update(losses)
