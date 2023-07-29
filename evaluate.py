from argparse import ArgumentParser
from PIL import Image
import torch
import torch.utils.data
import numpy as np
import torchvision
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
import pathlib
import matplotlib.pyplot as plt
import os
import datetime
import warnings
warnings.filterwarnings("ignore")
# import pdb
# pdb.set_trace()

from src.datas.NuscenesData import FuturePredictionDataset
from trainer import TrainingModule
from src.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from src.utils.network import preprocess_batch, NormalizeInverse
from src.utils.instance import predict_instance_segmentation_and_trajectories
from src.utils.geometry import calculate_birds_eye_view_parameters, interp_arc
from src.utils.tools import gen_dx_bx
from src.utils.visualisation import make_contour
from skimage.draw import polygon

def get_origin_points(cfg, bx, dx, lambda_=0):
    W = cfg.EGO.WIDTH
    H = cfg.EGO.HEIGHT
    pts = np.array([
        [-H / 2. + 0.5 - lambda_, W / 2. + lambda_],
        [H / 2. + 0.5 + lambda_, W / 2. + lambda_],
        [H / 2. + 0.5 + lambda_, -W / 2. - lambda_],
        [-H / 2. + 0.5 - lambda_, -W / 2. - lambda_],
    ])
    pts = (pts - bx.cpu().numpy()) / (dx.cpu().numpy())
    pts[:, [0, 1]] = pts[:, [1, 0]]
    rr , cc = polygon(pts[:,1], pts[:,0])
    rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)
    return torch.from_numpy(rc)

def get_points(cfg, trajs, lambda_=0):
    '''
    trajs: torch.Tensor<float> (B, N, n_future, 2)
    return:
    List[ torch.Tensor<int> (B, N, n_future), torch.Tensor<int> (B, N, n_future)]
    '''
    dx, bx, _ = gen_dx_bx(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
    dx, bx = dx[:2], bx[:2]

    _,_, bev_dimension = calculate_birds_eye_view_parameters(
        cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
    )

    rc = get_origin_points(cfg, bx, dx, lambda_)
    B, N, n_future, _ = trajs.shape

    trajs = trajs.view(B, N, n_future, 1, 2) / dx
    trajs[:,:,:,:,[0,1]] = trajs[:,:,:,:,[1,0]]
    trajs = trajs + rc

    rr = trajs[:,:,:,:,0].long()

    rr = torch.clamp(rr, 0, bev_dimension[0] - 1)

    cc = trajs[:,:,:,:,1].long()
    cc = torch.clamp(cc, 0, bev_dimension[1] - 1)

    return rr, cc

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def mk_save_dir():
    now = datetime.datetime.now()
    string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
    save_path = pathlib.Path('imgs') / string
    save_path.mkdir(parents=True, exist_ok=False)
    return save_path

def eval(checkpoint_path, dataroot):
    save_path = mk_save_dir()

    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=False)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cpu')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.LIFT.GT_DEPTH = False
    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.MAP_FOLDER = dataroot

    dataroot = cfg.DATASET.DATAROOT
    nworkers = cfg.N_WORKERS
    nusc = NuScenes(version='v1.0-{}'.format("mini"), dataroot=dataroot, verbose=False)
    valdata = FuturePredictionDataset(nusc, 1, cfg)
    print(cfg.BATCHSIZE, nworkers)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )

    n_classes = len(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
    hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
    metric_vehicle_val = IntersectionOverUnion(n_classes).to(device)
    future_second = int(cfg.N_FUTURE_FRAMES / 2)

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        metric_pedestrian_val = IntersectionOverUnion(n_classes).to(device)

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        metric_hdmap_val = []
        for i in range(len(hdmap_class)):
            metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1).to(device))

    if cfg.INSTANCE_SEG.ENABLED:
        metric_panoptic_val = PanopticMetric(n_classes=n_classes).to(device)

    if cfg.PLANNING.ENABLED:
        metric_planning_val = []
        for i in range(future_second):
            metric_planning_val.append(PlanningMetric(cfg, 2*(i+1)).to(device))


    for index, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        command = batch['command']
        trajs = batch['sample_trajectory']
        target_points = batch['target_point']
        print(target_points.shape, target_points)
        B = len(image)
        labels = trainer.prepare_future_labels(batch)
        n_present = model.receptive_field
        with torch.no_grad():
            output = model(
                image, intrinsics, extrinsics, future_egomotion
            )
            for i in range(len(hdmap_class)):
                hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
                hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
                break
            pedestrian_prediction = output['pedestrian'].detach()
            pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)

            seg_prediction = output['segmentation'].detach()
            print(output['segmentation'].shape, " output['segmentation']")
            seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)

            multiple = True

            font = {'weight' : 'bold',
                    'size'   : 35}
            mpl.rc('font', **font)                            

            for ind in range(1):
                pass
                fig = plt.figure(tight_layout=True, figsize=(20, 20))
                if multiple:
                    gs = mpl.gridspec.GridSpec(1, 2)
                    # ax1 = fig.add_subplot(gs[0, 0])
                    # ax4 = fig.add_subplot(gs[0, 1])
                    ax2 = fig.add_subplot(gs[0, 0])
                    ax3 = fig.add_subplot(gs[0, 1])
                    os.makedirs(f"output_vis/", exist_ok=True)
                    combined = np.ones((200, 200, 3))
                    combined_outputs = np.ones((200, 200, 3))
                    layers = [
                        labels['hdmap'][0, 1],
                        labels['hdmap'][0, 0], 
                        labels['segmentation'][:, n_present - 1:][0, ind, 0], 
                        labels['pedestrian'][:, n_present - 1:][0, ind, 0]
                    ]
                    hdmap = output['hdmap'].detach()
                    outputs = [
                        torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy(),
                        torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy(),
                        seg_prediction[:, n_present - 1:][0, ind, 0],
                        pedestrian_prediction[:, n_present - 1:][0, ind, 0]
                    ]
                    colors = [
                        [0, 0, 0],
                        [84, 70, 70],
                        [0, 0, 255],
                        [0, 128, 0],
                    ]
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
                    for layer, color in zip(layers, colors):
                        whe = np.where(layer > 0)
                        combined[whe[0], whe[1]] = np.array(color)/255
                    for layer, color in zip(outputs, colors):
                        whe = np.where(layer > 0)
                        combined_outputs[whe[0], whe[1]] = np.array(color)/255
                    # ax4.imshow(combined)
                    # ax1.set_xlim((200, 0));ax1.set_ylim((0, 200))
                    ax2.set_xlim((200, 0));ax2.set_ylim((0, 200))
                    ax3.set_xlim((200, 0));ax3.set_ylim((0, 200))
                    # ax4.set_xlim((150, 50));ax4.set_ylim((50, 150))
                    # ax1.fill(pts[:, 0], pts[:, 1], '#76b900')                
                    ax2.fill(pts[:, 0], pts[:, 1], '#76b900')
                    ax3.fill(pts[:, 0], pts[:, 1], '#76b900')
                    # ax4.fill(pts[:, 0], pts[:, 1], '#76b900')                
                    # ax1.set_title("Ground Truth")
                    ax2.imshow(combined_outputs)
                    # ax4.imshow(combined_outputs)
                    ax2.set_title("Predicted")
                    ax3.imshow(output["costvolume"].detach()[0, ind].cpu().numpy(), cmap="viridis")
                    ax3.set_title("Cost Volume")

                    font = {'weight' : 'bold',
                            'size'   : 35}

                    mpl.rc('font', **font)                
                    occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
                    _, final_traj, CS, cur_trajs, gt_cost_fc, gt_cost_fo = model.planning(
                        cam_front=output['cam_front'].detach(),
                        trajs=trajs[:, :, 1:],
                        gt_trajs=labels['gt_trajectory'][:, 1:],
                        cost_volume=output['costvolume'][:, n_present:].detach(),
                        semantic_pred=occupancy[:, n_present:].squeeze(2),
                        hd_map=output['hdmap'].detach(),
                        commands=command,
                        target_points=target_points
                    )
                    # for traj_ind, traj in enumerate(cur_trajs[0]):
                    #     ax4.plot(traj.detach().cpu().numpy()[:, 0], traj.detach().cpu().numpy()[:, 1], color="green", linewidth=5.0)
                    #     if ind > 10:
                    #         break
                    CC, KK = torch.topk(CS, k=1, dim=-1, largest=False)
                    occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                                labels['pedestrian'][:, n_present:].squeeze(2))
                    gt_trajs = labels['gt_trajectory']
                    gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
                    gt_trajs = (gt_trajs[0, :, :2].cpu().numpy() - bx) / dx
                    final_traj[0, :, :1] = final_traj[0, :, :1] * -1
                    final_traj = (final_traj[0, :, :2].cpu().numpy() - bx) / dx                
                    ax3.plot(final_traj[:, 0], final_traj[:, 1], color="yellow", linewidth=5.0, label="best selected")
                    ax3.text(final_traj[3, 0], final_traj[3, 1], str(np.round(CC[0,0].detach().cpu().numpy())), color="yellow")
                    ax2.plot(final_traj[:, 0], final_traj[:, 1], color="yellow", linewidth=5.0, label="best selected")
                    ax2.text(final_traj[3, 0], final_traj[3, 1], str(np.round(CC[0,0].detach().cpu().numpy())), color="yellow")
                    # ax4.set_title("Shooted Trajectories")
                    gt_cost = (gt_cost_fc + gt_cost_fo.sum(dim=-1))[0][0]
                    ax3.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=5.0, label="gt", color="#760900")
                    ax3.text(gt_trajs[-1, 0], gt_trajs[-1, 1], str(np.round(gt_cost.detach().cpu().numpy())), color="#760900")
                    ax2.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=5.0, label="gt", color="#760900")
                    ax2.text(gt_trajs[-1, 0], gt_trajs[-1, 1], str(np.round(gt_cost.detach().cpu().numpy())), color="#760900")

                    """
                    for _, closest_lane_poses in enumerate(batch["closest_lane_poses"]):
                        pt = interp_arc(6, closest_lane_poses[0, :, 0].detach().cpu().numpy(), closest_lane_poses[0, :, 1].detach().cpu().numpy())
                        pt = np.concatenate((pt, np.zeros((6, 1))), axis=1)
                        occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
                        _, __, ___, ____, c1_cost_fc, c1_cost_fo = model.planning(
                            cam_front=output['cam_front'].detach(),
                            trajs=trajs[:, :, 1:],
                            gt_trajs=labels['gt_trajectory'][:, 1:],
                            cost_volume=output['costvolume'][:, n_present:].detach(),
                            semantic_pred=occupancy[:, n_present:].squeeze(2),
                            hd_map=output['hdmap'].detach(),
                            commands=command,
                            target_points=target_points
                        )
                        closest_lane_poses[0, :, :1] = closest_lane_poses[0, :, :1] * -1
                        closest_lane_poses = (closest_lane_poses[0, :, :2].cpu().numpy() - bx) / dx
                        ax4.plot(closest_lane_poses[:, 0], closest_lane_poses[:, 1], linewidth=5.0, color="pink",label="centerline", linestyle='dashed')
                        c1_cost = (c1_cost_fc + c1_cost_fo.sum(dim=-1))[0,0].detach().cpu().numpy()
                        rank = len(CS[0]) - np.sum((CS[0].detach().cpu().numpy()  - c1_cost) >= 0)
                        print("rank ", rank)
                        ax4.text(closest_lane_poses[0, 0], closest_lane_poses[0, 1], str(np.round(c1_cost)) + " \n rank:" + str(rank) + " out of 1800", color="pink")
                        pass                
                    """
                    ax3.legend(frameon=True, facecolor="white")
                    save_ind = index + ind
                    plt.savefig(f"output_vis/{save_ind}.png");plt.clf()
                else:
                    gs = mpl.gridspec.GridSpec(1, 1)
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.set_xlim((200, 0));ax1.set_ylim((0, 200))                    
                    os.makedirs(f"output_vis/", exist_ok=True)
                    combined = np.ones((200, 200, 3))
                    combined_outputs = np.ones((200, 200, 3))
                    layers = [
                        labels['hdmap'][0, 1],
                        labels['hdmap'][0, 0], 
                        labels['segmentation'][:, n_present - 1:][0, ind, 0], 
                        labels['pedestrian'][:, n_present - 1:][0, ind, 0]
                    ]
                    hdmap = output['hdmap'].detach()
                    outputs = [
                        torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy(),
                        torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy(),
                        seg_prediction[:, n_present - 1:][0, ind, 0],
                        pedestrian_prediction[:, n_present - 1:][0, ind, 0]
                    ]
                    colors = [
                        [0, 0, 0],
                        [84, 70, 70],
                        [0, 0, 255],
                        [0, 128, 0],
                    ]
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
                    for layer, color in zip(layers, colors):
                        whe = np.where(layer > 0)
                        combined[whe[0], whe[1]] = np.array(color)/255
                    for layer, color in zip(outputs, colors):
                        whe = np.where(layer > 0)
                        combined_outputs[whe[0], whe[1]] = np.array(color)/255                    
                    # ax1.imshow(combined)
                    ax1.fill(pts[:, 0], pts[:, 1], '#76b900')                
                    ax1.set_title("Trajectories")
                    occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
                    _, final_traj, CS, cur_trajs, gt_cost_fc, gt_cost_fo = model.planning(
                        cam_front=output['cam_front'].detach(),
                        trajs=trajs[:, :, 1:],
                        gt_trajs=labels['gt_trajectory'][:, 1:],
                        cost_volume=output['costvolume'][:, n_present:].detach(),
                        semantic_pred=occupancy[:, n_present:].squeeze(2),
                        hd_map=output['hdmap'].detach(),
                        commands=command,
                        target_points=target_points
                    )
                    # print(trajs.shape, "trajs")
                    for traj_ind, traj in enumerate(trajs[0, :, 1:, :]):
                        traj[:, 0] *= -1
                        _trajs = (traj.detach().cpu().numpy()[:, :2] - bx) / dx
                        ax1.plot(_trajs[:, 0], _trajs[:, 1], color="#fad09d", linewidth=1.0, alpha=0.5)
                    CC, KK = torch.topk(CS, k=1, dim=-1, largest=False)
                    occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                                labels['pedestrian'][:, n_present:].squeeze(2))
                    gt_trajs = labels['gt_trajectory']
                    gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
                    gt_trajs = (gt_trajs[0, :, :2].cpu().numpy() - bx) / dx
                    final_traj[0, :, :1] = final_traj[0, :, :1] * -1
                    final_traj = (final_traj[0, :, :2].cpu().numpy() - bx) / dx                
                    ax1.plot(final_traj[:, 0], final_traj[:, 1], color="yellow", linewidth=5.0, label="best selected")
                    # ax1.text(final_traj[3, 0], final_traj[3, 1], str(np.round(CC[0,0].detach().cpu().numpy())), color="yellow")
                    ax1.set_title("Shooted Trajectories")
                    gt_cost = (gt_cost_fc + gt_cost_fo.sum(dim=-1))[0][0]
                    ax1.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=5.0, label="gt", color="#760900")
                    # ax4.text(gt_trajs[-1, 0], gt_trajs[-1, 1], str(np.round(gt_cost.detach().cpu().numpy())), color="#760900")
                    """
                    for _, closest_lane_poses in enumerate(batch["closest_lane_poses"]):
                        pt = interp_arc(6, closest_lane_poses[0, :, 0].detach().cpu().numpy(), closest_lane_poses[0, :, 1].detach().cpu().numpy())
                        pt = np.concatenate((pt, np.zeros((6, 1))), axis=1)
                        occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
                        _, __, ___, ____, c1_cost_fc, c1_cost_fo = model.planning(
                            cam_front=output['cam_front'].detach(),
                            trajs=trajs[:, :, 1:],
                            gt_trajs=labels['gt_trajectory'][:, 1:],
                            cost_volume=output['costvolume'][:, n_present:].detach(),
                            semantic_pred=occupancy[:, n_present:].squeeze(2),
                            hd_map=output['hdmap'].detach(),
                            commands=command,
                            target_points=target_points
                        )
                        closest_lane_poses[0, :, :1] = closest_lane_poses[0, :, :1] * -1
                        closest_lane_poses = (closest_lane_poses[0, :, :2].cpu().numpy() - bx) / dx
                        # ax1.plot(closest_lane_poses[:, 0], closest_lane_poses[:, 1], linewidth=5.0, color="pink",label="centerline", linestyle='dashed')
                        c1_cost = (c1_cost_fc + c1_cost_fo.sum(dim=-1))[0,0].detach().cpu().numpy()
                        rank = len(CS[0]) - np.sum((CS[0].detach().cpu().numpy()  - c1_cost) >= 0)
                        print("rank ", rank)
                        # ax1.text(closest_lane_poses[0, 0], closest_lane_poses[0, 1], str(np.round(c1_cost)) + " \n rank:" + str(rank) + " out of 1800", color="pink")
                    """
                    # ax1.legend(frameon=True, facecolor="white")
                    save_ind = index + ind
                    print("saving image")
                    plt.savefig(f"output_vis/{save_ind}.png");plt.clf()

        # semantic segmentation metric
        seg_prediction = output['segmentation'].detach()
        seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
        metric_vehicle_val(seg_prediction[:, n_present - 1:], labels['segmentation'][:, n_present - 1:])

        if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_prediction = output['pedestrian'].detach()
            pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
            metric_pedestrian_val(pedestrian_prediction[:, n_present - 1:],
                                       labels['pedestrian'][:, n_present - 1:])
        else:
            pedestrian_prediction = torch.zeros_like(seg_prediction)

        if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            for i in range(len(hdmap_class)):
                hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
                hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
                metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])

        if cfg.INSTANCE_SEG.ENABLED:
            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                output, compute_matched_centers=False, make_consistent=True
            )
            metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:],
                                     labels['instance'][:, n_present - 1:])

        if cfg.PLANNING.ENABLED:
            occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
            _, final_traj, CS,_, gt_cost_fc, gt_cost_fo = model.planning(
                cam_front=output['cam_front'].detach(),
                trajs=trajs[:, :, 1:],
                gt_trajs=labels['gt_trajectory'][:, 1:],
                cost_volume=output['costvolume'][:, n_present:].detach(),
                semantic_pred=occupancy[:, n_present:].squeeze(2),
                hd_map=output['hdmap'].detach(),
                commands=command,
                target_points=target_points
            )
            for i in range(future_second):
                cur_time = (i+1)*2
                # metric_planning_val[i](final_traj[:,:cur_time].detach(), labels['gt_trajectory'][:,1:cur_time+1], occupancy[:,:cur_time])

        if index % 1 == 0:
            save(output, labels, batch, n_present, index, save_path)


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
