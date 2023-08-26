from argparse import ArgumentParser
from PIL import Image
import torch
import os
import matplotlib as mpl
import torch.utils.data
import numpy as np
import json
import torchvision
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pathlib
import datetime
import copy
import pdb
import cv2
# pdb.set_trace()
import time
from stp3.datas.NuscenesData import FuturePredictionDataset
from stp3.trainer import TrainingModule
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.network import preprocess_batch, NormalizeInverse
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import make_contour, generate_instance_colours
from stp3.utils.Optnode_obs_unbatched import *
# from stp3.utils.Optnode_obs import *
from stp3.utils.Optnode import *
from stp3.utils.geometry import *

from stp3.cost import Cost_Function

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from transformers import TextStreamer
# Load model directly
# Use a pipeline as a high-level helper
from transformers import pipeline

from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from lavis.models import load_model_and_preprocess

from nuscenes.utils.data_classes import PointCloud, LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

from stp3.config import get_cfg

device = torch.device('cuda:0')
device2 = torch.device('cuda:0')
to_visualize = True
compute_losses = False

data_path = "/raid/t1/scratch/vikrant.dewangan/v1.0-trainval"
save_path = "/raid/t1/scratch/vikrant.dewangan/datas"

def eval(checkpoint_path, dataroot):
    # save_path = mk_save_dir()
    cfg = get_cfg()
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.LIFT.GT_DEPTH = False
    cfg.DATASET.DATAROOT = data_path
    cfg.DATASET.MAP_FOLDER = data_path
    cfg.DEBUG = True
    cfg.TIME_RECEPTIVE_FIELD = 1  # how many frames of temporal context (1 for single timeframe)
    cfg.N_FUTURE_FRAMES = 1  # how many time steps into the future to predict

    nusc = NuScenes(version='v1.0-{}'.format("trainval"), dataroot=data_path, verbose=False)
    valdata = FuturePredictionDataset(nusc, 1, cfg)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )

    for index, batch in enumerate(tqdm(valloader)):
        for objs in batch['categories_map'][0]:
            if "emergency" in objs[0]['category_name'][0]:
                print(index)
        arr = np.zeros((200, 200, 3))
        whe = np.where(batch['hdmap'].squeeze()[1] > 0)
        arr[whe[0], whe[1]] = np.array([255,255,255])
        # whe = np.where(batch['hdmap'].squeeze()[2,0] > 0)
        # arr[whe[0], whe[1]] = np.array([255,255,0])
        whe = np.where(batch['segmentation'].squeeze() > 0)
        arr[whe[0], whe[1]] = np.array([0,0,255])
        barr = np.copy(arr)

        bev_2d = np.logical_and(barr[:,:,2]>0,barr[:,:,0]==0)
        labels, pts_ = cv2.connectedComponents(bev_2d.astype(np.uint8))

        fig = plt.figure(figsize=(10, 3))

        gs = gridspec.GridSpec(6, 24)

        ax0 = plt.subplot(gs[0:3, 0:6])
        ax0.imshow(batch['unnormalized_images'][0,0,0].numpy())
        ax0.axis('off')

        ax1 = plt.subplot(gs[0:3, 6:12])
        ax1.imshow(batch['unnormalized_images'][0,0,1].numpy())
        ax1.axis('off')

        ax2 = plt.subplot(gs[0:3, 12:18])
        ax2.imshow(batch['unnormalized_images'][0,0,2].numpy())
        ax2.axis('off')

        ax3 = plt.subplot(gs[3:6, 0:6])
        ax3.imshow(batch['unnormalized_images'][0,0,3].numpy())
        ax3.axis('off')

        ax4 = plt.subplot(gs[3:6, 6:12])
        ax4.imshow(batch['unnormalized_images'][0,0,4].numpy())
        ax4.axis('off')

        ax5 = plt.subplot(gs[3:6, 12:18])
        ax5.imshow(batch['unnormalized_images'][0,0,5].numpy())
        ax5.axis('off')

        plt.tight_layout()
        plt.text(200, 250, "{0:0=6d}".format(index), fontsize=12)

        arr = np.zeros((200, 200, 3))
        whe = np.where(batch['hdmap'].squeeze()[1] > 0)
        arr[whe[0], whe[1]] = np.array([255,255,255])
        # whe = np.where(batch['hdmap'].squeeze()[2,0] > 0)
        # arr[whe[0], whe[1]] = np.array([255,255,0])
        whe = np.where(batch['segmentation'].squeeze() > 0)
        arr[whe[0], whe[1]] = np.array([0,0,255])
        for idx in range(1, labels + 1):
            # Create a JSON object for each component
            x, y = np.where(pts_ == idx)

            bevy, bevx = np.where(pts_ == idx)
            bevy = 200-bevy
        
            obj = {
                "object_id": idx,
                "bev_centroid": [(np.mean(bevx).astype(np.int) - 100)/2, (np.mean(bevy).astype(np.int) - 100)/2],
                "matched_coords": [x.tolist(), y.tolist()],
                "bev_area": len(x)/5,
            }
            target_x, target_y = np.mean(x).astype(np.uint8), np.mean(y).astype(np.uint8)
            # target = np.array([((obj['top'] + obj['bottom'])//2 - 100)/2, ((obj['left'] + obj['right'])//2 - 100)/2, 0])
            target = np.array([(target_x - 100)/2, (target_y - 100)/2, 0])
            min_ann_dist = 1e11
            best_ann = {}
            for ann_ind, anns in enumerate(batch["categories_map"][0]):
                dist = np.linalg.norm(anns[1][0][:2] - np.array([(target_x - 100)/2, (target_y - 100)/2]))
                annotation = anns[0]
                if dist < min_ann_dist:
                    min_ann_dist = dist
                    best_ann = annotation
            print(min_ann_dist)
            keys = best_ann.keys()
            for key in keys:
                if type(best_ann[key]) == torch.Tensor:
                    best_ann[key] = best_ann[key].tolist()
                    continue
                for itemind, item in enumerate(best_ann[key]):
                    if type(item) == torch.Tensor:
                        best_ann[key][itemind] = item.tolist()
                    elif type(item) == list:
                        for listind in range(len(item)):
                            if type(item[listind]) == torch.Tensor:
                                best_ann[key][itemind][listind] = best_ann[key][itemind][listind].tolist()
            obj["annotation"] = best_ann
            if "police" in best_ann["category_name"][0]:
                import pdb; pdb.set_trace()
                print(x, y)
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
                arr[x, y] = np.array([255, 0, 0])

        ax6 = plt.subplot(gs[0:6, 18:24])
        ax6.imshow(arr)
        ax6.axis('off')
        plt.savefig('sample_cam_imgs/'+"{0:0=6d}".format(index)+'.png', dpi=300)        
        plt.savefig("arr.png")

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)
    args = parser.parse_args()
    eval(args.checkpoint, args.dataroot)
