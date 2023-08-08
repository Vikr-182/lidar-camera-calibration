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
    nusc = NuScenes(version='v1.0-{}'.format("mini"), dataroot=dataroot, verbose=False)
    valdata = FuturePredictionDataset(nusc, 0, cfg)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=False
    )

    for index, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        labels = trainer.prepare_future_labels(batch)
        trj = labels['gt_trajectory_prev'] # (B, 1, 9, 2)

        with torch.no_grad():
            output = model(
                image, intrinsics, extrinsics, future_egomotion
            )
            # breakpoint()
            print(output.keys())

        seg_prediction = output['segmentation'].detach()
        seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
        pred = seg_prediction.squeeze()
        arr = np.zeros((200, 200, 3))
        whe = np.where(batch['hdmap'].squeeze()[2,1] > 0)
        arr[whe[0], whe[1]] = np.array([255,255,255])
        whe = np.where(torch.argmax(output['hdmap'].squeeze()[2:4], dim=0, keepdim=True)[0] > 0)
        arr[whe[0], whe[1]] = np.array([255,255,255])
        whe = np.where(pred.squeeze()[2] > 0)
        arr[whe[0], whe[1]] = np.array([0,0,255])
        Image.fromarray(arr.astype(np.uint8)).save("grad_data/pred_bev.png")        

        arr = np.zeros((200, 200, 3))
        whe = np.where(batch['hdmap'].squeeze()[2,1] > 0)
        arr[whe[0], whe[1]] = np.array([255,255,255])
        # whe = np.where(batch['hdmap'].squeeze()[2,0] > 0)
        # arr[whe[0], whe[1]] = np.array([255,255,0])
        whe = np.where(batch['segmentation'].squeeze()[2] > 0)
        arr[whe[0], whe[1]] = np.array([0,0,255])
        Image.fromarray(arr.astype(np.uint8)).save("grad_data/sample_bev.png")
        import pdb; pdb.set_trace()

        # if index % 100 == 0:
        #     save(output, labels, batch, n_present, index, save_path)

def save(output, labels, batch, n_present, frame, save_path):
    hdmap = output['hdmap'].detach()
    segmentation = output['segmentation'][:, n_present - 1].detach()
    pedestrian = output['pedestrian'][:, n_present - 1].detach()
    gt_trajs = labels['gt_trajectory']
    images = batch['image']

    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )

    val_w = 2.99
    val_h = 2.99 * (224. / 480.)
    plt.figure(1, figsize=(4*val_w,2*val_h))
    width_ratios = (val_w,val_w,val_w,val_w)
    gs = matplotlib.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    plt.subplot(gs[0, 0])
    plt.annotate('FRONT LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,0].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 1])
    plt.annotate('FRONT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,1].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 2])
    plt.annotate('FRONT RIGHT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,2].cpu()))
    plt.axis('off')

    plt.subplot(gs[1, 0])
    plt.annotate('BACK LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0,n_present-1,3].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[1, 1])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 4].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[1, 2])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 5].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[:, 3])
    showing = torch.zeros((200, 200, 3)).numpy()
    showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255])

    # drivable
    area = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # lane
    area = torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])

    # semantic
    semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
    semantic_index = semantic_seg > 0
    showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])

    pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
    pedestrian_index = pedestrian_seg > 0
    showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])

    plt.imshow(make_contour(showing))
    plt.axis('off')

    bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    dx = np.array([0.5, 0.5])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    plt.xlim((200, 0))
    plt.ylim((0, 200))
    gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
    gt_trajs = (gt_trajs[0, :, :2].cpu().numpy() - bx) / dx
    plt.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=3.0)

    plt.savefig(save_path / ('%04d.png' % frame))
    plt.close()

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)

    args = parser.parse_args()

    eval(args.checkpoint, args.dataroot)
