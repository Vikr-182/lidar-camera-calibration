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
    nworkers = cfg.N_WORKERS
    nusc = NuScenes(version='v1.0-{}'.format("mini"), dataroot=dataroot, verbose=False)
    valdata = FuturePredictionDataset(nusc, 1, cfg)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )

    for index, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        command = batch['command']
        trajs = batch['sample_trajectory']
        target_points = batch['target_point']
        if cfg.PLANNING.DENSE:
            centerlines = batch['centerlines']
        labels = trainer.prepare_future_labels(batch)
        n_present = model.receptive_field
        trj = labels['gt_trajectory_prev'] # (B, 1, 9, 2)
        traj_past = trj[:, :3, :2]
        traj_future = trj[:, 3:, :2]
        import pdb; pdb.set_trace()
        print(labels['gt_trajectory_prev'].shape, batch['gt_trajectory_prev'].shape, labels['gt_trajectory'].shape, " ok")
        print(traj_past.shape, "traj_past")


        bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
        dx = np.array([0.5, 0.5])


        ######################################## getting obs ###################################################################
        """
            instance: (B, 9, 200, 200)
            centerness: (B, 9, 1, 200, 200)
            offset: (B, 9, 2, 200, 200)
            flow: (B, 9, 2, 200, 200)
        """
        instance_seg = labels['instance'].detach().cpu().numpy()
        centerness = labels['centerness'].detach().cpu().numpy()
        offset = labels['offset'].detach().cpu().numpy()
        flow = labels['flow'].detach().cpu().numpy()

        unique_ids = torch.unique(labels['instance'])
        unique_ids = unique_ids.detach().cpu().numpy()
        instance_map = dict(zip(unique_ids, unique_ids))
        instance_colors = generate_instance_colours(instance_map)

        """
            instance: (6, 200, 200)
            centerness: (6, 1, 200, 200)
            offset: (6, 2, 200, 200)
            flow: (6, 2, 200, 200)
            we use centerness for now
        """        
        instance = instance_seg[0, n_present - 1:]
        centerness = centerness[0, n_present - 1:, 0]
        offset = centerness[0, n_present - 1:, 0]
        flow = centerness[0, n_present - 1:, 0]

        instance_trajs = {}

        # compute top K obs
        num_obs = cfg.PLANNING.DENSE.NUM_OBS
        for instance_id in unique_ids:
            if instance_id == 0:
                # background
                continue
            # compute across time
            points = []
            time_dict = {}
            for i in range(0, n_future + 1):
                whe = np.where(instance[i] == instance_id)
                if len(whe[0]) == 0:
                    continue
                max_num = np.argmax(centerness[i][whe[0], whe[1]])
                print(i, [whe[0][max_num], whe[1][max_num]])
                points.append([whe[0][max_num], whe[1][max_num]])
            if len(points) == 7: # only select points who are visible for all time-steps
                print("BAZINGA")
                instance_trajs[instance_id] = points

        obs_ids = list(instance_trajs.keys())
        distance_dict = {}
        for obs in obs_ids:
            points = np.array(instance_trajs[obs])
            # distance from last observed time-step, ideally -> lanewise
            distance_dict[obs] = np.linalg.norm(traj_past[0, -1] - points)
        distance_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda item: item[1])}    # sort points by distance
        obs_ids = list(distance_dict.keys())
        obs_cnt = 0
        x_obs = np.ones((num_obs, n_future + 1)) * -5e5
        y_obs = np.ones((num_obs, n_future + 1)) * -5e5
        ids_obs = []
        # select top num_obs obstacles
        for instance_ids in obs_ids:
            if instance_ids == 0: continue
            points = np.array(instance_trajs[instance_ids])
            x_obs[obs_cnt] = points[:, 0]
            y_obs[obs_cnt] = points[:, 1]
            ids_obs.append(instance_ids)
            obs_cnt = obs_cnt + 1
            if obs_cnt >= num_obs:
                break
        obs =  np.dstack((x_obs, y_obs))

        """
            plot just to check
        """
        for obb, instanceids in zip(obs, ids_obs):
            if instance_ids == 0: continue
            ob = copy.deepcopy(obb)
            points = np.array(ob)
            points[:, [0, 1]] = points[:, [1, 0]]
            points = (points * dx) + bx
            points[:, :1] = points[:, :1] * -1
            obs_cnt = obs_cnt + 1
            plt.scatter(points[:, 0], points[:, 1], color=instance_colors[instance_ids]/255)
            plt.plot(points[:, 0], points[:, 1], color=instance_colors[instance_ids]/255)
        gt=labels['gt_trajectory'][0, 1:]
        plt.plot(gt[:, 0], gt[:, 1], color='red')
        plt.scatter(gt[:, 0], gt[:, 1], color='red')
        plt.savefig("output_vis/points.png")
        plt.clf()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        print("OBS ", obs.shape)
        ##################################################################################################################

        with torch.no_grad():
            output = model(
                image, intrinsics, extrinsics, future_egomotion
            )
            # breakpoint()
            print(output.keys())

        seg_prediction = output['segmentation'].detach()
        seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)

        if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_prediction = output['pedestrian'].detach()
            pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
        else:
            pedestrian_prediction = torch.zeros_like(seg_prediction)

        if cfg.PLANNING.ENABLED:
            occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
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
            print("hd map shape", batch["hdmap"].shape)
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


            """
                centerlines and sampling to avoid obstacles
            """
            data_centerlines = batch['centerlines']
            num_centerlines = batch['num_centerlines']
            device = data_centerlines.device
            trajectories = []
            trj = labels['gt_trajectory_prev'] # (B, 1, 9, 2)
            trajs_overall = torch.zeros((1, 0, 7, 2)).to(device)
            for centerlines in data_centerlines[0, 0][:num_centerlines]:
                X_MAX = 30
                X_MIN = 0
                NUM_X = 10
                Y_MAX = 2
                Y_MIN = -2
                NUM_Y = 3
                griddim = (X_MIN, X_MAX, NUM_X, Y_MIN, Y_MAX, NUM_Y)

                """
                    ideally -> do a constant velocity prediction over centerline, get points, and select top obs accordingly
                """
                dense_trajs = extract_trajs(centerlines, seg_prediction=labels['segmentation'], hdmap_prediction=labels['hdmap'], trj=labels['gt_trajectory_prev'], device=image.device, viz=False, labels=labels, griddim=griddim, problem=problem, obs=torch.tensor(obs), avoid_obs=cfg.PLANNING.DENSE.OBS, ind=str(index)) # (B, M, num, 2)
                # dense_trajs = extract_trajs(centerlines, trj=labels['gt_trajectory_prev'], x_obs=x_obs, y_obs=y_obs, device='cpu', viz=False, labels=labels, griddim=griddim, problem=problem) # (B, M, num, 2)
                trajs_overall = dense_trajs

            """
                loop over n_future timesteps
            """
            print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
            for ind in range(0, n_future):
                fig = plt.figure(tight_layout=True, figsize=(20, 15))
                if to_visualize:
                    gs = mpl.gridspec.GridSpec(2, 2)
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax2 = fig.add_subplot(gs[0, 1])
                    ax3 = fig.add_subplot(gs[1, 0])
                    ax4 = fig.add_subplot(gs[1, 1])
                    os.makedirs(f"output_vis/", exist_ok=True)
                    ax4.set_xlim((150, 50));ax4.set_ylim((50, 150))
                    ax1.set_xlim((150, 50));ax1.set_ylim((50, 150))
                    ax2.set_xlim((150, 50));ax2.set_ylim((50, 150))
                    ax3.set_xlim((150, 50));ax3.set_ylim((50, 150))

                    """
                        plot semantics:
                            labels: GT semantics
                                labels['segmentation'][:, n_present:]: output seg vehicle
                                labels['pedestrian'][:, n_present:]: output seg pedestrian
                                labels['hdmap']: output hdmap
                            output:
                                seg_pedestrian: output seg vehicle
                                pedestrian_pedestrian: output seg pedestrian
                                hdmap: output hdmap
                            combined: combined labels
                            combined_output: outputs combined
                    """
                    combined = np.ones((200, 200, 3))
                    seg_labels = labels['segmentation'][:, n_present:][0, ind, 0].detach().cpu().numpy()
                    seg_display = np.ones_like(seg_labels)
                    unique_ids_time = np.unique(instance[ind])
                    combined_outputs = np.ones((200, 200, 3))
                    layers = [
                        labels['hdmap_warped_road'][:, n_present:][0, ind, 0],
                        labels['hdmap_warped_lane'][:, n_present:][0, ind, 0],
                        labels['segmentation'][:, n_present:][0, ind, 0],
                        labels['pedestrian'][:, n_present:][0, ind, 0]
                    ]
                    hdmap = output['hdmap'].detach()
                    hdmap_prediction = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
                    outputs = [
                        torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy(),
                        torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy(),
                        seg_prediction[:, n_present:][0, ind, 0],
                        pedestrian_prediction[:, n_present:][0, ind, 0]
                    ]
                    colors = [
                        [0, 0, 0],
                        [84, 70, 70],
                        [0, 0, 240],
                        [0, 128, 0],
                    ]
                    for layer, color in zip(layers, colors):
                        whe = np.where(layer > 0)
                        combined[whe[0], whe[1]] = np.array(color)/255
                    for layer, color in zip(outputs, colors):
                        whe = np.where(layer > 0)
                        combined_outputs[whe[0], whe[1]] = np.array(color)/255
                    for obb, instance_ids in zip(obs, ids_obs):
                        ob = copy.deepcopy(obb)
                        points = ob
                        points[:, [0, 1]] = points[:, [1, 0]]
                        points = (points * dx) + bx
                        points[:, :1] = points[:, :1] * -1
                        ax1.scatter(points[:, 0], points[:, 1], color=instance_colors[instance_ids]/255)
                        ax1.plot(points[:, 0], points[:, 1], color=instance_colors[instance_ids]/255)
                        ax3.plot(points[:, 0], points[:, 1], color=instance_colors[instance_ids]/255)
                        ax3.scatter(points[:, 0], points[:, 1], color=instance_colors[instance_ids]/255)

                    ax1.imshow(combined)
                    ax2.imshow(combined)
                    ax4.imshow(combined)                                                     
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                                    


                    """
                        visualize trajectory set
                    """
                    for cur_traj in trajs_overall[0]:
                        xy = cur_traj
                        if ind == 0:
                            xy[:, :1] = xy[:, :1] * -1
                        xy = (xy[:, :2].detach().cpu().numpy() - bx) / dx
                        ax1.plot(xy[:, 0], xy[:, 1], color="orange",linewidth=1,alpha=0.35)

                    """
                        plot ego-box
                    """
                    ego_translation =  labels['gt_trajectory'][0, ind + 1] - labels['gt_trajectory'][0, 0]
                    ego_translation_ =  labels['gt_trajectory'][0, ind + 1] - labels['gt_trajectory'][0, ind]
                    ego_rotation = np.arctan2(ego_translation_[1], ego_translation_[0])
                    theta = ego_rotation
                    import math
                    w, h = 1.85, 4.084
                    pts = np.array([
                        [-h / 2. + 0.5, w / 2.],
                        [h / 2. + 0.5, w / 2.],
                        [h / 2. + 0.5, -w / 2.],
                        [-h / 2. + 0.5, -w / 2.],
                    ])
                    save_pts = copy.deepcopy(pts)    
                    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
                    save_pts = np.dot(rot, save_pts.T).T
                    save_pts = save_pts + np.array(ego_translation)[:2]
                    save_pts = (save_pts - bx) / dx
                    # save_pts[:, [0, 1]] = save_pts[:, [1, 0]]
                    ax1.fill(save_pts[:, 0], save_pts[:, 1], '#76b900')
                    ax2.fill(save_pts[:, 0], save_pts[:, 1], '#76b900')
                    ax3.fill(save_pts[:, 0], save_pts[:, 1], '#76b900')
                    ax4.fill(save_pts[:, 0], save_pts[:, 1], '#76b900')                
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



                    """
                        visualize cost volume heatmap of query grid
                    """
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                    """
                        visualize trajectory set
                    """
                    for cur_traj in cur_trajs[0]:
                        xy = cur_traj
                        if ind == 0:
                            xy[:, :1] = xy[:, :1] * -1
                        xy = (xy[:, :2].detach().cpu().numpy() - bx) / dx
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


                    """
                        visualize centerlines
                    """
                    cx = batch['centerlines'][0][0,0]
                    if ind == 0:
                        cx[:, :1] = cx[:, :1] * -1
                    cx = (cx[:, :2].detach().cpu().numpy() - bx) / dx
                    ax1.plot(cx[:, 0], cx[:, 1], linewidth=3, color="grey",linestyle="--", zorder=10)
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                    """
                        plot other agents
                    """


                    """
                        visualize 
                            final_traj_prev:    traj before GRU 
                            final_traj:         traj after GRU
                            gt_trajs:           GT traj
                    """
                    ax1.plot(final_traj_prev[:, 0], final_traj_prev[:, 1], color="red", linewidth=5.0, label="before GRU")
                    ax1.scatter(final_traj_prev[:, 0], final_traj_prev[:, 1], color="red", s=50.0, label="before GRU")
                    ax1.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=5.0, label="gt", color="#760900", zorder=20)
                    ax1.scatter(gt_trajs[:, 0], gt_trajs[:, 1], s=50.0, label="gt", color="#760900", zorder=20)
                    ax3.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=5.0, label="gt", color="#760900", zorder=20)
                    ax3.scatter(gt_trajs[:, 0], gt_trajs[:, 1], s=50.0, label="gt", color="#760900", zorder=20)

                    ax1.set_title("Predicted")
                    ax2.set_title("Cost Volume")

                    save_ind = str(index) + "_" + str(ind)
                    plt.savefig(f"output_vis/{save_ind}.png");plt.clf()

        # if index % 100 == 0:
        #     save(output, labels, batch, n_present, index, save_path)


    results = {}

    scores = metric_vehicle_val.compute()
    results['vehicle_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        scores = metric_pedestrian_val.compute()
        results['pedestrian_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        for i, name in enumerate(hdmap_class):
            scores = metric_hdmap_val[i].compute()
            results[name + '_iou'] = scores[1]

    if cfg.INSTANCE_SEG.ENABLED:
        scores = metric_panoptic_val.compute()
        for key, value in scores.items():
            results['vehicle_'+key] = value[1]

    if cfg.PLANNING.ENABLED:
        for i in range(future_second):
            scores = metric_planning_val[i].compute()
            for key, value in scores.items():
                results['plan_'+key+'_{}s'.format(i+1)]=value.mean()

    for key, value in results.items():
        print(f'{key} : {value.item()}')

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
