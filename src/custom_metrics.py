from ray.rllib.agents.callbacks import DefaultCallbacks
import ray
import visdom
import numpy as np


class CustomMetrics(DefaultCallbacks):

    INFO_METRICS = ["distance_to_goal", "success", "spl", "softspl"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visdom = visdom.Visdom("http://localhost", port=5050)
        self.train_iters = 0

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        info = episode.last_info_for()
        episode.custom_metrics.update(
            {k: info[k] for k in self.INFO_METRICS if k in info}
        )

    def on_learn_on_batch(self, *, policy, train_batch, result: dict, **kwargs) -> None:
        result["action_dist"] = train_batch["actions"]
        result["act_sum"] = 1.0

    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        m = trainer.get_policy().model

        if not hasattr(m, "visdom_mets"):
            return

        for met_type, met in m.visdom_mets.items():
            for k, imgs in met.items():
                opts = {"caption": k, "title": k}
                if met_type == "heat":
                    self.visdom.heatmap(imgs, opts=opts, win=k)
                elif met_type == "scatter":
                    self.visdom.scatter(imgs, opts=opts, win=k)
                elif met_type == "line":
                    self.visdom.line(
                        Y=imgs,
                        X=np.array([self.train_iters]),
                        opts=opts,
                        win=k,
                        name=k,
                        update="append",
                    )
                else:
                    raise Exception(f"Unknown metric type: {met_type}")

        m.visdom_mets.clear()
        self.train_iters += 1


"""
class GraphMetrics(DefaultCallbacks, CustomMetrics):
    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        m = trainer.get_policy().model
        metrics = {
            "graph_density": m.density,
        }
        result["custom_metrics"].update(metrics)
"""


class VAEMetrics(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visdom = visdom.Visdom("http://localhost", port=5050)
        self.window_map = {}

    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        m = trainer.get_policy().model
        losses = {
            "ae_semantic_loss": m.sem_loss.detach().item(),
            "ae_depth_loss": m.depth_loss.detach().item(),
            "ae_kld_loss": m.kld_loss.detach().item(),
            "ae_combined_loss": m.combined_loss.detach().item(),
        }
        result["custom_metrics"].update(losses)
        for k, imgs in m.visdom_imgs.items():
            if k not in self.window_map:
                win_hash = self.visdom.images(imgs, opts={"caption": k, "title": k})
                self.window_map[k] = win_hash
            else:
                self.visdom.images(
                    imgs, opts={"caption": k, "title": k}, win=self.window_map[k]
                )


class AEMetrics(DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        m = trainer.get_policy().model
        losses = {
            "ae_semantic_loss": m.sem_loss.detach().item(),
            "ae_depth_loss": m.depth_loss.detach().item(),
            "ae_combined_loss": m.combined_loss.detach().item(),
        }
        result["custom_metrics"].update(losses)
