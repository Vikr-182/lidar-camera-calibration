import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
import io
import copy
from io import BytesIO
import matplotlib
import matplotlib as mpl
import PIL
from PIL import Image
from tqdm import tqdm

from stp3.config import get_cfg
from stp3.models.stp3 import STP3
from stp3.models.stp3_unary import STP3_Un
from stp3.losses import SpatialRegressionLoss, SegmentationLoss, HDmapLoss, DepthLoss
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.geometry import cumulative_warp_features_reverse, cumulative_warp_features, extract_trajs, extract_obs_from_centerness, generate_instance_colours
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import visualise_output
from stp3.utils.data import prepare_future_labels
from stp3.datas.dataloaders import prepare_dataloaders
from stp3.config import get_parser, get_cfg

def hook_fn(m, i, o):
    print(m)
    print("------------Input Grad------------")

    for grad in i:
        try:
            print(grad.shape)
        except AttributeError: 
            print ("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:  
        try:
            print(grad.shape)
        except AttributeError: 
            print ("None found for Gradient")
    print("\n")

data_path = "/raid/t1/scratch/vikrant.dewangan/v1.0-trainval"
save_path = "/raid/t1/scratch/vikrant.dewangan/datas"


def main():
    # trainer args
    gpus = [0, 1, 2, 3]
    accelerator = 'ddp'
    logger = wandb
    device = "cuda:0"
    epochs = 10

    ##############################################

    # setup
    cfg = get_cfg()
    cfg.TIME_RECEPTIVE_FIELD = 1
    cfg.N_FUTURE_FRAMES = 0
    cfg.LOG_DATA = False
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 12
    cfg.LIFT.GT_DEPTH = False
    cfg.DATASET.DATAROOT = data_path
    cfg.DATASET.MAP_FOLDER = data_path
    cfg.DEBUG = True
    cfg.LOG_DATA = True


    if cfg.LOG_DATA: wandb.init(project="stp3-nuscenes-perception", entity="forecasting", name="test_run0")
    n_classes = len(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
    hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS

    # Bird's-eye view extent in meters
    cfg.LIFT.X_BOUND[1] > 0 and cfg.LIFT.Y_BOUND[1] > 0

    ##############################################

    # Model
    model = STP3_Un(cfg)
    model.to(device)
    # model.LSS.register_backward_hook(hook_fn)

    # Optimizer
    params = model.parameters()
    optimizer = torch.optim.Adam(
        params, lr=cfg.OPTIMIZER.LR, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
    )

    print("Initialized model")

    ##############################################

    # Losses
    losses_fn = nn.ModuleDict()

    cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS = [1.0, 5.0]
    # Semantic segmentation
    losses_fn['segmentation'] = SegmentationLoss(
        class_weights=torch.Tensor(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS),
        use_top_k=cfg.SEMANTIC_SEG.VEHICLE.USE_TOP_K,
        top_k_ratio=cfg.SEMANTIC_SEG.VEHICLE.TOP_K_RATIO,
        future_discount=cfg.FUTURE_DISCOUNT,
    ).to(device)
    model.segmentation_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    metric_vehicle_val = IntersectionOverUnion(n_classes)

    # HD map
    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        losses_fn['hdmap'] = HDmapLoss(
            class_weights=torch.Tensor(cfg.SEMANTIC_SEG.HDMAP.WEIGHTS),
            training_weights=cfg.SEMANTIC_SEG.HDMAP.TRAIN_WEIGHT,
            use_top_k=cfg.SEMANTIC_SEG.HDMAP.USE_TOP_K,
            top_k_ratio=cfg.SEMANTIC_SEG.HDMAP.TOP_K_RATIO,
        ).to(device)
        metric_hdmap_val = []
        for i in range(len(hdmap_class)):
            metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1))
        model.hdmap_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        metric_hdmap_val = nn.ModuleList(metric_hdmap_val)
    print("Initialized losses")

    ##############################################

    # prepare datalaoder
    trainloader, valloader = prepare_dataloaders(cfg)
    metric_vehicle_val = IntersectionOverUnion(n_classes)
    ##############################################

    # train loop
    losses = []
    for epoch in range(epochs):
        for batchind, batch in enumerate(tqdm(trainloader)):
            image = batch['image'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            future_egomotion = batch['future_egomotion'].to(device)

            # Warp labels
            labels = {}
            labels['segmentation'] = batch['segmentation'].long().contiguous()
            labels['hdmap'] = batch['hdmap'].long().contiguous()
            # labels = prepare_future_labels(batch, model)

            # Forward pass
            output = model(
                image, intrinsics, extrinsics, future_egomotion
            )

            # compute losses
            loss = {}

            # segmentation
            segmentation_factor = 1 / (2 * torch.exp(model.segmentation_weight))
            loss['segmentation'] = segmentation_factor * losses_fn['segmentation'](
                output['segmentation'].to(device), labels['segmentation'].to(device), model.receptive_field
            )

            for key, value in loss.items():
                if cfg.LOG_DATA: wandb.log({'step_train_loss_' + key: value}) 

            loss = sum(loss.values())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batchind % 50 == 49:
                pred = torch.argmax(output['segmentation'], dim=2).cpu()[0,0];plt.imshow(pred);plt.savefig("pred.png");plt.clf()
                label = labels['segmentation'].cpu()[0,0,0];plt.imshow(label);plt.savefig("label.png");plt.clf()
                # import pdb; pdb.set_trace()
            if batchind % 100 == 0:
                print(loss)
                torch.save(model.state_dict(), f"model_{batchind}.ckpt")

if __name__ == "__main__":
    main()