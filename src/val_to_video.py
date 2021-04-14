import sys
import pickle
import os
from typing import Dict, Any
import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import io
import torch

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def generate_ego_graph(gps, obs_imgs, edges, frame):
    cur_gps = gps[frame]
    neigh_frames = edges[frame].nonzero()[0]
    neigh_gps = gps[neigh_frames]
    # Make everything ego-relative
    neigh_gps -= cur_gps
    cur_gps -= cur_gps

    # Img is [B, 1, H, W]
    # we want it [B, H, W, 3] for opencv
    all_imgs = obs_imgs[[frame, *neigh_frames.tolist()]]
    all_imgs = np.tile(np.swapaxes(all_imgs, 1, -1), [1, 1, 1, 3])

    G = nx.Graph()

    all_nodes = np.concatenate(
        (cur_gps.reshape(1, 2), neigh_gps.reshape(-1, 2)), axis=0
    )
    G.add_nodes_from(
        [
            (
                i,
                {
                    "pos": all_nodes[i],
                    "label_pos": all_nodes[i] + np.array([0, -0.08]),
                    "img": all_imgs[i],
                },
            )
            for i in range(all_nodes.shape[0])
        ]
    )
    G.add_edges_from([(0, i) for i in range(1, all_nodes.shape[0])])
    return G


def render_ego_graph(G):
    fig = plt.figure()
    drawpos = nx.get_node_attributes(G, "pos")
    labelpos = nx.get_node_attributes(G, "label_pos")
    imgs = nx.get_node_attributes(G, "img")
    nx.draw(G, drawpos)
    nx.draw_networkx_labels(G, labels=drawpos, pos=labelpos)
    trans = plt.gca().transData.transform
    trans2 = fig.transFigure.inverted().transform

    img_size = 0.1
    p2 = img_size / 2.0
    offset = 0.08
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    for i in imgs:
        xx, yy = trans(drawpos[i])  # figure coordinates
        xa, ya = trans2((xx, yy))  # axes coordinates
        a = plt.axes([xa - p2, ya - p2 + offset, img_size, img_size])
        # a.set_aspect('equal')
        a.imshow(imgs[i])
        a.axis("off")
        # import pdb; pdb.set_trace()
        # plt.figimage(imgs[i], drawpos[i][0], plt.ylim()[1] - drawpos[i][1])
    plt.draw()
    return fig


def load_vae(path="/root/vnav/depth_vae.pt"):
    return torch.load(path).to("cpu").eval()


def latent_to_img(net, latents):
    with torch.no_grad():
        # Format: [B, C, H, W]
        out = net.decode(torch.from_numpy(latents)).numpy()
    return (out * 255).astype(np.uint8)


def get_ego_edges(gps, edges, frame):
    cur_gps = gps[frame]
    neigh_frames = edges[frame].nonzero()[0]
    neigh_gps = gps[neigh_frames]
    # Make everything ego-relative
    neigh_gps -= cur_gps
    cur_gps -= cur_gps


def process(data: Dict[str, np.ndarray], vae: torch.nn.Module, outdir: str):
    depths = latent_to_img(vae, data["latent"])
    for frame in range(data["action_prob"].shape[0]):
        G = generate_ego_graph(data["gps"], depths, data["forward_edges"], frame)
        fig = render_ego_graph(G)
        fig.savefig(f"{outdir}/{frame}.jpg")
        plt.close(fig)
        # ep_imgs += [render_ego_graph(G)]


def main():
    wdir = sys.argv[1] + "/validation"
    files = [f"{wdir}/{f}" for f in os.listdir(wdir) if os.path.isfile(f"{wdir}/{f}")]
    vae = load_vae()
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        outdir = f"/tmp/gviz/{os.path.basename(f)}"
        os.makedirs(outdir, exist_ok=True)
        process(data, vae, outdir)


if __name__ == "__main__":
    main()
