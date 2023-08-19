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

data_path = "/raid/t1/scratch/vikrant.dewangan/v1.0-trainval"
save_path = "/raid/t1/scratch/vikrant.dewangan/datas"

def eval(checkpoint_path, dataroot):
    # save_path = mk_save_dir()
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    trainer.cfg.INSTANCE_FLOW.ENABLED = True
    trainer.cfg.INSTANCE_SEG.ENABLED = True    
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
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

    dataroot = data_path#cfg.DATASET.DATAROOT
    nusc = NuScenes(version='v1.0-{}'.format("trainval"), dataroot=dataroot, verbose=False)
    valdata = FuturePredictionDataset(nusc, 0, cfg)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=10, pin_memory=True, drop_last=False
    )

    model = torch.nn.DataParallel(model, device_ids=[0])

    for index, batch in enumerate(tqdm(valloader)):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        cam_names = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
        old_arr = np.zeros((200, 200, 3))
        with torch.no_grad():
            output = model(
                image, intrinsics, extrinsics, future_egomotion
            )
            os.makedirs(os.path.join(save_path, str(index)), exist_ok=True)
            for idx in range(6):
                img = batch['unnormalized_images'][0,2,idx].numpy().astype(np.uint8)
                Image.fromarray(img).save(os.path.join(save_path, str(index), cam_names[idx] + ".png"))
            arr = np.zeros((200, 200, 3))
            whe = np.where(batch['hdmap'].squeeze()[2,1] > 0)
            arr[whe[0], whe[1]] = np.array([255,255,255])
            # whe = np.where(batch['hdmap'].squeeze()[2,0] > 0)
            # arr[whe[0], whe[1]] = np.array([255,255,0])
            whe = np.where(batch['segmentation'].squeeze()[2] > 0)
            arr[whe[0], whe[1]] = np.array([0,0,255])
            Image.fromarray(arr.astype(np.uint8)).save(os.path.join(save_path, str(index), "gt_bev.png"))

            pred = torch.argmax(output['segmentation'], dim=2).squeeze()[2].cpu().numpy()
            arr = np.zeros((200, 200, 3))
            whe = np.where(batch['hdmap'].squeeze()[2,1] > 0)
            arr[whe[0], whe[1]] = np.array([255,255,255])
            # whe = np.where(batch['hdmap'].squeeze()[2,0] > 0)
            # arr[whe[0], whe[1]] = np.array([255,255,0])
            whe = np.where(pred > 0)
            arr[whe[0], whe[1]] = np.array([0,0,255])
            Image.fromarray(arr.astype(np.uint8)).save(os.path.join(save_path, str(index), "pred_bev.png"))

            print(np.linalg.norm(arr - old_arr))

            np.save(os.path.join(save_path, str(index), "points.npy"), batch['point_clouds'][2].squeeze()[0]) # 4, N
            np.save(os.path.join(save_path, str(index), "coloring.npy"), batch['point_clouds_labels'][2].squeeze()[0]) # N

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)
    args = parser.parse_args()
    eval(args.checkpoint, args.dataroot)
