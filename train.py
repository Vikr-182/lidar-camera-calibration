import os
import time
import socket
import torch
from torchsummary import summary
import pytorch_lightning as pl
from stp3.models.stp3 import STP3
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from nuscenes.nuscenes import NuScenes

from stp3.datas.NuscenesData import FuturePredictionDataset
from stp3.config import get_parser, get_cfg
from stp3.datas.dataloaders import prepare_dataloaders
from stp3.trainer import TrainingModule

# import pdb
# pdb.set_trace()


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    model = TrainingModule(cfg.convert_to_dict())

    # nusc = NuScenes(version='v1.0-{}'.format("mini"), dataroot="/mnt/e/datasets/nuscenes/mini", verbose=False)
    # valdata = FuturePredictionDataset(nusc, 0, cfg)
    # valloader = torch.utils.data.DataLoader(
    #     valdata, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False
    # )
    # trainloader = valloader

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            cfg.PRETRAINED.PATH, map_location='cpu'
        )['state_dict']
        state = model.state_dict()
        for k, v in pretrained_model_weights.items():
            if 'decoder' in k and k in state:
                print(k, v.shape)
        pretrained_model_weights = {k: v for k, v in pretrained_model_weights.items() if k in state and 'decoder' not in k}
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')


if __name__ == "__main__":
    main()