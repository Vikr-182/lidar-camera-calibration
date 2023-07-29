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
    nworkers = cfg.N_WORKERS
    nusc = NuScenes(version='v1.0-{}'.format("mini"), dataroot=dataroot, verbose=False)
    valdata = FuturePredictionDataset(nusc, 0, cfg)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=False
    )

    plt.figure(figsize=(20, 10))
    for timestep in range(3, 9):

        data = json.load(open("f_dict.json"))
        arr = []
        for k, v in data.items():
            if k != str(timestep):continue
            print("aa")
            for pt in v:
                arr.append(pt)
        arr = np.array(arr)
        d_pred = arr[:, 1]
        error = arr[:, 2]
        r_safe = 0.5 #m
        dd = r_safe - (d_pred - (error))
        func = np.maximum(0, dd)
        dd, func = (list(t) for t in zip(*sorted(zip(dd, func))))
        # plt.scatter(dd, func)
        y, binEdges  = np.histogram(np.array(func), bins=100)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        y[:20] = y[:20]/10
        # plt.plot(bincenters, y/100, '-', c='black', label='0.5')

        data = json.load(open("f_dict_2.json"))
        arr = []
        for k, v in data.items():
            if k != str(timestep):continue
            for pt in v:
                arr.append(pt)
            break
        arr = np.array(arr)
        d_pred = arr[:, 1]
        error = arr[:, 2]
        r_safe = 0.5 #m
        dd = r_safe - (d_pred - (error))
        func = np.maximum(0, dd)
        dd, func = (list(t) for t in zip(*sorted(zip(dd, func))))
        # plt.scatter(dd, func)
        y, binEdges  = np.histogram(np.array(func), bins=100)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        y[:20] = y[:20]/10
        y[20:] = y[20:]/10
        plt.plot(bincenters, y/100, '-', label=f'{timestep}')


        # print(len(bincenters), len(xx), len(y), len(yy))
        # print(bincenters[-1], xx[-1], y[-1], yy[-1])
        # plt.plot(-xx, yy/100, '-', c='black')
        plt.xlim([-2,2])
        plt.yticks([])
        plt.legend()
        plt.xlabel('Constraint Violation (m)')
        plt.ylabel('Probability Density')
        plt.title(f'T={timestep - 3}')
        plt.savefig(f"dirac_{timestep - 3}.png")
        plt.clf()

    offsets = []
    for index, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        command = batch['command']
        trajs = batch['sample_trajectory']
        target_points = batch['target_point']
        labels = trainer.prepare_future_labels(batch)
        n_present = model.receptive_field
        trj = labels['gt_trajectory_prev'] # (B, 1, 9, 2)
        traj_past = trj[:, :3, :2]
        traj_future = trj[:, 3:, :2]
        # print(labels['gt_trajectory_prev'].shape, batch['gt_trajectory_prev'].shape, labels['gt_trajectory'].shape, " ok")
        # print(traj_past.shape, "traj_past")


        bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
        dx = np.array([0.5, 0.5])

        with torch.no_grad():
            output = model(
                image, intrinsics, extrinsics, future_egomotion
            )
            # breakpoint()
        seg_prediction = output['segmentation'].detach()
        seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)

        if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_prediction = output['pedestrian'].detach()
            pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
        else:
            pedestrian_prediction = torch.zeros_like(seg_prediction)

        if cfg.PLANNING.ENABLED:
            occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
            occupancy_label = torch.logical_or(labels['segmentation'], labels['pedestrian'])
            n_future = cfg.N_FUTURE_FRAMES

            """
                Define weights
            """
            SAFETYCOST = 0.1
            HEADWAYCOST = 1.0
            LRDIVIDERCOST = 10.0
            COMFORTCOST = 1.0
            PROGRESSCOST = 0.5
            RULECOST = 5.0
            COSTVOLUME = 100.0
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            """
                bx -> offset
                dx -> scale
            """
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
            save_pts = copy.deepcopy(pts)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++            
            model.planning.cost_function.safetycost.factor = SAFETYCOST
            model.planning.cost_function.headwaycost.factor = HEADWAYCOST
            model.planning.cost_function.lrdividercost.factor = LRDIVIDERCOST
            model.planning.cost_function.comfortcost.factor = COMFORTCOST
            model.planning.cost_function.progresscost.factor = PROGRESSCOST
            model.planning.cost_function.rulecost.factor = RULECOST
            model.planning.cost_function.costvolume.factor = COSTVOLUME

            """
            model planning
                cur_trajs : (N, n_future, 2) -> Expert set of trajectories
                final_traj_prev : (1, n_future, 2) -> Final selectred traj pre GRU refinement
                final_traj : (1, n_future, 2)  -> Final selectred traj post GRU refinement
                sm_cost_fo : (N, n_future)  -> Costs related to time dimension
                sm_cost_fc : (N, 1)  -> Costs related to trajectories
            """
            gt_trajs=labels['gt_trajectory'][:, 1:]
            occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
            _, final_traj, CS, cur_trajs, gt_cost_fc, gt_cost_fo, final_traj_prev, sm_cost_fc, sm_cost_fo = model.planning(
                cam_front=output['cam_front'].detach(),
                trajs=trajs[:, :, 1:],
                gt_trajs=labels['gt_trajectory'][:, 1:],
                cost_volume=output['costvolume'][:, n_present:].detach(),
                semantic_pred=occupancy[:, n_present:].squeeze(2),
                hd_map=output['hdmap'].detach(),
                commands=["NA" for i in command],
                target_points=target_points
            )
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            """
                gt_traj : (n_future, 2) -> GT Trajectory
                final_traj_prev : (n_future, 2) -> Final selectred traj pre GRU refinement
                final_traj : (n_future, 2)  -> Final selectred traj post GRU refinement
            """
            gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
            gt_trajs = (gt_trajs[0, :, :2].detach().cpu().numpy() - bx) / dx
            final_traj[0, :, :1] = final_traj[0, :, :1] * -1
            final_traj = (final_traj[0, :, :2].detach().cpu().numpy() - bx) / dx
            final_traj_prev[0, :, :1] = final_traj_prev[0, :, :1] * -1
            final_traj_prev = (final_traj_prev[0, :, :2].detach().cpu().numpy() - bx) / dx
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++            

            trajs_overall = batch['sample_trajectory']
            """
                loop over n_future timesteps
            """
            print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
            pred = seg_prediction[:, n_present:]
            x = torch.linspace(0, 199, 200)
            y = torch.linspace(0, 199, 200)
            gx, gy = torch.meshgrid((x, y))
            grid = torch.dstack((gx, gy))
            distances = grid - (torch.ones((200, 200, 2)) * 100)
            distances = torch.linalg.norm(distances, dim=2)
            p0 = pred.squeeze()[0]
            d_ = distances * p0
            whe = torch.where(d_ == torch.unique(d_)[1])
            d_pred = torch.tensor([whe[0][0], whe[1][0]]) - torch.tensor([100, 100])

            gt = labels['segmentation'][:, n_present:]
            gt0 = gt.squeeze()[0]
            g_ = distances * gt0
            whe = torch.where(g_ == torch.unique(g_)[1])
            d_gt = torch.tensor([whe[0][0], whe[1][0]]) - torch.tensor([100, 100])
            # import pdb; pdb.set_trace()

            print(d_pred, d_gt)
            offsets.append(d_pred - d_gt)
            print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")

            hdmap_pred_road = torch.argmax(output['hdmap'][:,2:4], dim=1)
            hdmap_pred_lane = torch.argmax(output['hdmap'][:,0:2], dim=1)
            labels_lane = labels['hdmap'][:, 0]
            labels_road = labels['hdmap'][:, 1]
            false_negatives_seg = torch.clamp((occupancy_label.long() - occupancy.long()), 0, 1)
            false_positives_seg = torch.clamp((occupancy.long() - occupancy_label.long()), 0, 1)
            false_positives_road = torch.clamp((hdmap_pred_road - labels_road), 0, 1)
            false_positives_lane = torch.clamp((hdmap_pred_lane - labels_lane), 0, 1)
            true_positives_seg = occupancy.long() - false_positives_seg.long()
            error_map = true_positives_seg + (0.5 * false_negatives_seg) # (B, T, 1, 200, 200)
            mask = error_map == 0 # where it is not vehicle
            distance_array = torch.ones_like(error_map) * 1e11 * (mask)
            
            for num in range(200):
                #up
                distance_array[:,:,:,1:,:] = torch.min(distance_array[:,:,:,1:,:], distance_array[:,:,:,:-1,:] + 1)
                #down
                distance_array[:,:,:,:-1,:] = torch.min(distance_array[:,:,:,:-1,:], distance_array[:,:,:,1:,:] + 1)
                #left
                distance_array[:,:,:,:,1:] = torch.min(distance_array[:,:,:,:,1:], distance_array[:,:,:,:,:-1] + 1)
                #right
                distance_array[:,:,:,:,:-1] = torch.min(distance_array[:,:,:,:,:-1], distance_array[:,:,:,:,1:] + 1)
                #topleft
                distance_array[:,:,:,1:,1:] = torch.min(distance_array[:,:,:,1:,1:], distance_array[:,:,:,:-1,:-1] + torch.sqrt(torch.tensor(2)))
                #topright
                distance_array[:,:,:,1:,:-1] = torch.min(distance_array[:,:,:,1:,:-1], distance_array[:,:,:,:-1,1:] + torch.sqrt(torch.tensor(2)))
                #downleft
                distance_array[:,:,:,:-1,1:] = torch.min(distance_array[:,:,:,:-1,1:], distance_array[:,:,:,1:,:-1] + torch.sqrt(torch.tensor(2)))
                #downright
                distance_array[:,:,:,:-1,:-1] = torch.min(distance_array[:,:,:,:-1,:-1], distance_array[:,:,:,1:,1:] + torch.sqrt(torch.tensor(2)))

            num_centerlines = batch['num_centerlines']
            centerlines = batch['centerlines'].squeeze()[:num_centerlines]
            for cx in centerlines:
                cx = cx * dx
                cx += bx
                cx[:, [1, 0]] = cx[:, [0, 1]]
                cx[:, :1] = cx[:, :1] * -1

                cx[:, :1] = cx[:, :1] * -1
                cx = (cx[:, :2] - bx) / dx

                plt.scatter(cx[:, 0], cx[:, 1], color='grey', s=0.1)

            distribution = torch.normal(mean=0, std=torch.ones((100)))
            plt.scatter(final_traj[:, 0], final_traj[:, 1], label='pred')
            plt.scatter(gt_trajs[:, 0], gt_trajs[:, 1], label='gt')
            ee = error_map.squeeze()[2]
            plt.imshow(ee, cmap='seismic')
            plt.legend(facecolor='white')
            plt.axis('off')
            plt.grid(False)
            plt.savefig('error_map.png')
            plt.clf()



            # corrupt occupancy based on a distribution
            # import pdb; pdb.set_trace()


    print(torch.tensor(offsets).mean(), " mean of")
    torch.save(torch.tensor(offsets), "offsets.pt")

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)

    args = parser.parse_args()

    eval(args.checkpoint, args.dataroot)
