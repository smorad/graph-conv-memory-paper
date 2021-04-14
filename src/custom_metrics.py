from ray.rllib.agents.callbacks import DefaultCallbacks
import ray
import visdom
import numpy as np
import torch
import pickle
import os


def sample_to_input(sample_batch, device):
    """Convert a sample batch to (x, states, seq_lens) for a forward call"""
    sample_batch.pop("infos", None)
    for k in sample_batch:
        sample_batch[k] = torch.from_numpy(sample_batch[k]).to(device=device)

    states = []
    i = 0
    while "state_in_{}".format(i) in sample_batch:
        states.append(sample_batch["state_in_{}".format(i)])
        i += 1
    return (
        sample_batch,
        states,
        torch.from_numpy(sample_batch.get("seq_lens")).to(device=device),
    )


# ret = self.__call__(sample_batch, states, sample_batch.get("seq_lens"))


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
        # input = sample_to_input(train_batch, policy.model.device)
        # print('cb', input[-1].shape)
        # self.writer.add_graph(policy.model, input)

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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class EvalMetrics(CustomMetrics):
    EVAL_METRICS = [
        "latent",
        "gps",
        "map",
        "action_prob",
        "forward_edges",
        "backward_edges",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index: int, **kwargs
    ):
        for k in self.EVAL_METRICS:
            episode.user_data[k] = []

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        super().on_episode_end(worker, base_env, policies, episode, env_index, **kwargs)
        export_dict = {k: np.array(episode.user_data[k]) for k in self.EVAL_METRICS}
        tb_dir = worker.io_context.log_dir
        outdir = f"{tb_dir}/validation"
        os.makedirs(outdir, exist_ok=True)
        with open(f"{outdir}/{episode.episode_id}.pkl", "wb") as f:
            pickle.dump(export_dict, f)

    def on_sample_end(self, *, worker, samples, **kwargs):
        pass

    def on_episode_step(self, *, worker, base_env, episode, env_index: int, **kwargs):
        episode.user_data["gps"].append(episode.last_raw_obs_for()["gps"])
        episode.user_data["latent"].append(episode.last_raw_obs_for()["vae"])
        if episode.length > 0:
            episode.user_data["forward_edges"].append(
                episode.rnn_state_for()[1][:, episode.length - 1]
            )
            episode.user_data["backward_edges"].append(
                episode.rnn_state_for()[1][episode.length - 1, :]
            )
        try:
            probs = softmax(episode.last_pi_info_for()["action_dist_inputs"])
            episode.user_data["action_prob"].append(probs)
        except KeyError:
            pass


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
