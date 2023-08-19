from argparse import ArgumentParser
from PIL import Image
import torch
import os
import matplotlib as mpl
import torch.utils.data
import numpy as np
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
# pdb.set_trace()

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

to_visualize = True
compute_losses = False

font = {'weight' : 'bold',
        'size'   : 35}
mpl.rc('font', **font)                            
colors_seq = [
    [255, 0 , 0],
    [255, 0 , 0],
    [255, 127, 0],
    [255, 255, 0],
    [0, 255, 0],                                
    [148, 0, 211],
]

t_fin = 3
num = 61
problem = OPTNode(t_fin=t_fin, num=num)

def eval(checkpoint_path, dataroot):
    # save_path = mk_save_dir()
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    trainer.cfg.INSTANCE_FLOW.ENABLED = True
    trainer.cfg.INSTANCE_SEG.ENABLED = True    
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cpu')
    trainer.to(device)
    model = trainer.model

    x = np.linspace(0, 19, 20)
    y = np.linspace(-3, 3, 7)
    xv, yv = np.meshgrid(x, y)
    grid = np.dstack((xv, yv))
    grid = np.concatenate((grid), axis=0)

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.LIFT.GT_DEPTH = False
    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.MAP_FOLDER = dataroot
    cfg.DEBUG = True

    n_present = cfg.TIME_RECEPTIVE_FIELD
    n_future = cfg.N_FUTURE_FRAMES

    dataroot = cfg.DATASET.DATAROOT
    nusc = NuScenes(version='v1.0-{}'.format("trainval"), dataroot=dataroot, verbose=False)
    valdata = FuturePredictionDataset(nusc, 0, cfg)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )

    prev_scene_token = None
    scene_cnt = 0
    selected_scenes = [10]
    for index, batch in enumerate(tqdm(valloader)):
        cur_scene_token = batch['scene_token'][0]
        scene_cnt = scene_cnt + 1
        if scene_cnt not in selected_scenes: 
            if prev_scene_token != None:
                if cur_scene_token == prev_scene_token:
                    continue
                else:
                    scene_cnt = 0
        prev_scene_token = cur_scene_token

        fig = plt.figure(figsize=(10, 3))

        gs = gridspec.GridSpec(6, 24)

        ax0 = plt.subplot(gs[0:3, 0:6])
        ax0.imshow(batch['unnormalized_images'][0,2,0].numpy())
        ax0.axis('off')

        ax1 = plt.subplot(gs[0:3, 6:12])
        ax1.imshow(batch['unnormalized_images'][0,2,1].numpy())
        ax1.axis('off')

        ax2 = plt.subplot(gs[0:3, 12:18])
        ax2.imshow(batch['unnormalized_images'][0,2,2].numpy())
        ax2.axis('off')

        ax3 = plt.subplot(gs[3:6, 0:6])
        ax3.imshow(batch['unnormalized_images'][0,2,3].numpy())
        ax3.axis('off')

        ax4 = plt.subplot(gs[3:6, 6:12])
        ax4.imshow(batch['unnormalized_images'][0,2,4].numpy())
        ax4.axis('off')

        ax5 = plt.subplot(gs[3:6, 12:18])
        ax5.imshow(batch['unnormalized_images'][0,2,5].numpy())
        ax5.axis('off')

        ax6 = plt.subplot(gs[0:6, 18:24])
        ax6.imshow(batch['segmentation'][0,2,0].numpy(), cmap='gray')
        ax6.axis('off')

        plt.tight_layout()
        plt.text(200, 250, "{0:0=6d}".format(index), fontsize=12)

        plt.savefig('sample_cam_imgs/'+"{0:0=6d}".format(index)+'.png', dpi=300)



if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)
    args = parser.parse_args()
    eval(args.checkpoint, args.dataroot)
