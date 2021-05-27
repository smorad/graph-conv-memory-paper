from ray.rllib.agents.callbacks import DefaultCallbacks
from models.ray_graph import RayObsGraph
from typing import Dict
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
    TRAIN_METRICS = ["edge_density", "reg_loss"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.visdom = visdom.Visdom("http://localhost", port=5050)
        self.train_iters = 0

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        info = episode.last_info_for()
        episode.custom_metrics.update(
            {k: info[k] for k in self.INFO_METRICS if k in info}
        )

    def on_learn_on_batch(self, *, policy, train_batch, result: dict, **kwargs) -> None:
        result["action_dist"] = train_batch["actions"]
        result["act_sum"] = 1.0

        # num_params = sum(p.numel() for p in policy.model.parameters() if p.requires_grad)
        # print(f'\n\nPOLICY {policy.model} # OF PARAMS: {num_params}\n\n')
        # input = sample_to_input(train_batch, policy.model.device)
        # print('cb', input[-1].shape)
        # self.writer.add_graph(policy.model, input)

    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        m = trainer.get_policy().model

        tb_mets = {}
        for k in self.TRAIN_METRICS:
            if hasattr(m, k):
                tb_mets[k] = getattr(m, k)

        num_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        result["custom_metrics"]["num_params"] = num_params

        result["custom_metrics"].update(tb_mets)

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
        "compass",
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
        if not isinstance(worker.get_policy().model, RayObsGraph):
            return
        for k in self.EVAL_METRICS:
            episode.user_data[k] = []

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        if not isinstance(worker.get_policy().model, RayObsGraph):
            return
        super().on_episode_end(worker, base_env, policies, episode, env_index, **kwargs)
        export_dict = {
            k: np.array(episode.user_data[k])
            for k in self.EVAL_METRICS
            if k in episode.user_data
        }
        tb_dir = worker.io_context.log_dir
        outdir = f"{tb_dir}/validation"
        os.makedirs(outdir, exist_ok=True)
        with open(f"{outdir}/{episode.episode_id}.pkl", "wb") as f:
            pickle.dump(export_dict, f)

    def on_episode_step(self, *, worker, base_env, episode, env_index: int, **kwargs):
        if not isinstance(worker.get_policy().model, RayObsGraph):
            return
        episode.user_data["gps"].append(episode.last_raw_obs_for()["gps"])
        episode.user_data["compass"].append(episode.last_raw_obs_for()["compass"])
        episode.user_data["latent"].append(episode.last_raw_obs_for()["vae"])
        # Only run for graphs
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


class VAEMetrics(DefaultCallbacks):
    PREFIX = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visdom = visdom.Visdom("http://localhost", port=5050)
        self.window_map = {}

    def get_losses(self, m):
        return {
            "ae_semantic_loss": m.sem_loss.detach().item(),
            "ae_depth_loss": m.depth_loss.detach().item(),
            "ae_kld_loss": m.kld_loss.detach().item(),
            "ae_combined_loss": m.combined_loss.detach().item(),
        }

    def log_images(self, m):
        for k, imgs in m.visdom_imgs.copy().items():
            key = self.PREFIX + k

            if key not in self.window_map:
                win_hash = self.visdom.images(imgs, opts={"caption": key, "title": key})
                self.window_map[key] = win_hash
            else:
                self.visdom.images(
                    imgs, opts={"caption": key, "title": key}, win=self.window_map[key]
                )

    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        model = trainer.get_policy().model
        result["custom_metrics"].update(self.get_losses(model))
        self.log_images(model)


class VAEEvalMetrics(VAEMetrics):
    PREFIX = "evaluation/"
    mets: Dict[str, np.ndarray] = {}

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        episode.custom_metrics.update(self.mets)
        self.mets = {}

    def on_sample_end(self, *, worker, samples, **kwargs):
        model = worker.get_policy().model

        # Push batch thru trained model while in training mode
        # so losses are populated
        # Since we discard the losses from model.custom_loss
        # we do not train the model
        torch_samples, _, _ = sample_to_input(samples, model.device)
        model.from_batch(torch_samples, is_training=False)
        model.custom_loss([torch.tensor([0], device=model.device)], None)
        self.mets = self.get_losses(model)
        self.log_images(model)


class AEMetrics(DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        m = trainer.get_policy().model
        losses = {
            "ae_semantic_loss": m.sem_loss.detach().item(),
            "ae_depth_loss": m.depth_loss.detach().item(),
            "ae_combined_loss": m.combined_loss.detach().item(),
        }
        result["custom_metrics"].update(losses)
