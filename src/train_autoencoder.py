import argparse
import glob
import numpy as np
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.shuffled_input import ShuffledInput
from torch.utils.data import DataLoader
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
#from detectron2.demo.predictor import AsyncPredictor
from detectron2.modeling import build_model

from models.cnn import CNNAutoEncoder


def build_sem_net():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # TODO: Rather than use predictor, we should use the model itself
    # so we can use batch inputs
    predictor = DefaultPredictor(cfg)
    return predictor
    #model = build_model(cfg)
    #model.eval()
    #return model

def reconstruct_rgbd(batch, width=640, height=480, dims=4):
    b = batch.reshape(batch.shape[0], dims, height, width)
    depth = b[:,0,:,:] / 255.0
    rgb = b[:,1:,:,:] / 255.0
    return rgb, depth


def rgb_to_semantic(predictor, batch, num_classes=80):
    # Expecting bgr image [3xHxW]
    res = torch.zeros(batch.shape[0], num_classes, batch.shape)
    for sample_i in range(batch.shape[0]):
        import pdb; pdb.set_trace()
        # RGB to BGR and [HxWx3] to [3xHxW]
        network_in = np.moveaxis(predictor(batch[sample_i][:,:,::-1], 0, -1)
        res.append(out)
    return out



def build_model():
    ae = AutoEncoder()
    criterion = 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


def main():
    rollout_dir = '/root/vnav/data/rgbd_480p'
    rollout_files = glob.glob(f'{rollout_dir}/*.json')
    loocv_reader = JsonReader(rollout_files[0])
    train_reader = ShuffledInput(JsonReader(rollout_files[1:]), n=10)
    batch_idx = 0
    sem_net = build_sem_net()
    while True:
        batch = train_reader.next()['obs']
        rgb, depth = reconstruct_rgbd(batch)
        sem_inputs = rgb_to_semantic(sem_net, rgb)
        #sem = sem_net(rgb)
        # Train
        batch_idx += 1

        if batch_idx % 10 == 0:
            # Val
            pass

        


if __name__ == '__main__':
    main()
