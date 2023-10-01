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
from stp3.datas.reducedNuscenesData import FuturePredictionDataset
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

from segment_anything import sam_model_registry, SamPredictor

# ========================================
#             SAM Initialization
# ========================================
def init_sam(model_type = "vit_h", checkpoint="/home/t1/vikrant.dewangan/llm-bev/MiniGPT-4/sam_vit_h_4b8939.pth", device="cuda:0"):
    sam = sam_model_registry[model_type](checkpoint="/home/t1/vikrant.dewangan/llm-bev/MiniGPT-4/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor
# ========================================

def find_bounding_box(image):
    # Find the coordinates of non-zero (foreground) pixels along each channel
    foreground_pixels = np.array(np.where(np.any(image != 0, axis=2)))

    # Calculate the bounding box coordinates
    min_y, min_x = np.min(foreground_pixels, axis=1)
    max_y, max_x = np.max(foreground_pixels, axis=1)

    return min_y, min_x, max_y, max_x

def find_bounding_box_masks(image):
    # Find the coordinates of non-zero (foreground) pixels along each channel
    foreground_pixels = np.array(np.where(image != 0))

    # Calculate the bounding box coordinates
    min_y, min_x = np.min(foreground_pixels, axis=1)
    max_y, max_x = np.max(foreground_pixels, axis=1)

    return min_y, min_x, max_y, max_x

def crop_around_bounding_box(image, masks, get_black=False):
    if get_black:
        min_y, min_x, max_y, max_x = find_bounding_box(masks)
    else:
        min_y, min_x, max_y, max_x = find_bounding_box_masks(masks)

    # Crop the image using the bounding box coordinates
    cropped_image = image[min_y:max_y+1, min_x:max_x+1, :]

    return cropped_image

def kClosest(points, target, K):
    # Convert the points and target lists to NumPy arrays
    points = np.array(points)
    target = np.array([target])
    # Calculate the distance between each point and the target using NumPy broadcasting
    distances = np.sqrt(np.sum((points[:, :2] - target[:, :2])**2, axis=1))
    # Get the indices of the K closest points using argsort
    closest_indices = np.argsort(distances)[:K]
    # Get the K closest points
    closest_points = points[closest_indices]
    return closest_points.tolist()

def get_image_projection(cam_left, cam_front, cam_right, cam_rear_left, cam_rear, cam_rear_right, arr):
    cam_imgs = [cam_left, cam_front, cam_right, cam_rear_left, cam_rear, cam_rear_right]
    cam_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    calibration_data = json.load(open("calibration.json"))
    min_dist = 1.0
    flag = 0
    cam_img = None
    matched_cam = None
    matched_points = None
    for cam_ind, cam_key in enumerate(cam_keys):
        barr = np.copy(arr)
        ppc = LidarPointCloud(barr.T);
        ppc.translate(np.array(calibration_data[cam_key]['translation']))
        ppc.rotate(np.array(calibration_data[cam_key]['rotation']))
        ddepths_img = ppc.points[2, :]
        points = view_points(ppc.points[:3, :], np.array(calibration_data[cam_key]['camera_intrinsic']), normalize=True)
        
        mask = np.ones(ddepths_img.shape[0], dtype=bool)
        mask = np.logical_and(mask, ddepths_img > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < 1600 - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < 900 - 1)

        if mask.sum() > 0:
            # found
            flag = 1
            points = points[:, mask]
            ddepths_img = ddepths_img[mask]
            cam_img = cam_imgs[cam_ind]
            matched_points = points
            matched_cam = cam_key
            break

    if flag == 0:
        # no point able to back-project, just use front cam
        matched_points = [150, 100] # hard-coded
        cam_img = cam_front
    return cam_img, points, matched_points, matched_cam

def extract_mask(predictor, cam_img, points):
    """
        arr: points to be back-projected: (K closest points)
    """
    predictor.set_image(cam_img)
    input_point = np.array(points.astype(np.int32).T[:, :2])
    input_label = np.array([1 for i in range(len(points[0]))])
    masks, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label,multimask_output=True)
    idxs = (masks[-1].astype(np.uint8) * 200) > 0
    img_copy = np.copy(cam_img)
    # img_copy[~idxs] = 0
    img_copy = crop_around_bounding_box(img_copy, masks[-1])
    return img_copy, masks

device = torch.device('cuda:0')
device2 = torch.device('cuda:1')
# LLaVa
# tokenizer, model_llava, image_processor, context_len = init_llava()
# print("Initializaed LLaVa")

# chat = init_minigp4()

predictor = init_sam(device=device2)
print("Initializaed SAM")
print('Initialization Finished')

to_visualize = True
compute_losses = False

data_path = "/raid/t1/scratch/vikrant.dewangan/v1.0-trainval"
save_path = "/raid/t1/scratch/vikrant.dewangan/datas"

def eval(checkpoint_path, dataroot):
    # save_path = mk_save_dir()
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    trainer.cfg.INSTANCE_FLOW.ENABLED = True
    trainer.cfg.INSTANCE_SEG.ENABLED = True    
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

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
    cfg.DATASET.DATAROOT = data_path
    cfg.DATASET.MAP_FOLDER = data_path
    cfg.DEBUG = True

    nusc = NuScenes(version='v1.0-{}'.format("trainval"), dataroot=data_path, verbose=False)
    valdata = FuturePredictionDataset(nusc, 0, cfg)
    # valdata.indices = valdata.indices[1500:]
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=False
    )

    for indexx, batch in enumerate(tqdm(valloader)):
        cur_scene_token = batch['scene_token'][0]
        unn = batch["unnormalized_images"][0].squeeze()[0]
        arr = np.zeros((200, 200, 3))
        whe = np.where(batch['hdmap'].squeeze()[2,1] > 0)
        arr[whe[0], whe[1]] = np.array([255,255,255])
        whe = np.where(batch['segmentation'].squeeze()[2] > 0)
        arr[whe[0], whe[1]] = np.array([0,0,255])
        np.save("unn.npy", batch["unnormalized_images"].squeeze())
        np.save("points.npy", batch["point_clouds"][-1].squeeze()[0])
        np.save("labels.npy", batch["point_clouds_labels"][-1].squeeze()[0])
        Image.fromarray(arr.astype(np.uint8)).save("sample_bev.png")

        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        cam_names = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

        with torch.no_grad():
            os.makedirs(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index)), exist_ok=True)
            for idx in range(6):
                img = batch['unnormalized_images'][0,2,idx].numpy().astype(np.uint8)
                Image.fromarray(img).save(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), cam_names[idx] + ".png"))

            arr = np.zeros((200, 200, 3))
            whe = np.where(batch['hdmap'].squeeze()[2,1] > 0)
            arr[whe[0], whe[1]] = np.array([255,255,255])
            # whe = np.where(batch['hdmap'].squeeze()[2,0] > 0)
            # arr[whe[0], whe[1]] = np.array([255,255,0])
            whe = np.where(batch['segmentation'].squeeze()[2] > 0)
            arr[whe[0], whe[1]] = np.array([0,0,255])
            barr = np.copy(arr)
            Image.fromarray(arr.astype(np.uint8)).save(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), "gt_bev.png"))

            labels_allowed = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            pts = []
            lidardata = batch['point_clouds'][2].squeeze()[2].numpy()
            for ptind, pt in enumerate(lidardata.T):
                if batch['point_clouds_labels'][2].squeeze()[2][ptind] in labels_allowed:
                    pts.append(pt)
            cts = np.copy(np.array(pts))

            bev_2d = np.logical_and(barr[:,:,2]>0,barr[:,:,0]==0)
            labels, pts_ = cv2.connectedComponents(bev_2d.astype(np.uint8))
            matched_imgs = []
            cam_imgs = []
            for cin in range(6):
                cam_imgs.append(batch['unnormalized_images'][0,2,cin].numpy())

            objects_json = []
            for idx in range(1, labels + 1):
                try:
                    # Create a JSON object for each component
                    
                    x, y = np.where(pts_ == idx)

                    bevy, bevx = np.where(pts_ == idx)
                    bevy = 200-bevy

                    obj = {
                        "object_id": idx,
                        "bev_centroid": [(np.mean(bevx).astype(np.int) - 100)/2, (np.mean(bevy).astype(np.int) - 100)/2],
                        "matched_coords": [x.tolist(), y.tolist()],
                        "bev_area": len(x)/4,
                    }
                    target_x, target_y = np.mean(x).astype(np.uint8), np.mean(y).astype(np.uint8)
                    # target = np.array([((obj['top'] + obj['bottom'])//2 - 100)/2, ((obj['left'] + obj['right'])//2 - 100)/2, 0])
                    target = np.array([(target_x - 100)/2, (target_y - 100)/2, 0])

                    min_ann_dist = 1e11
                    best_ann = {}
                    for ann_ind, anns in enumerate(batch["categories_map"][2]):
                        dist = np.linalg.norm(anns[1][0][:2] - np.array([(target_x - 100)/2, (target_y - 100)/2]))
                        annotation = anns[0]
                        if dist < min_ann_dist:
                            min_ann_dist = dist
                            best_ann = annotation
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
                    try:
                        token_obj = best_ann["token"]
                    except:
                        continue
                    tokens_obj = [token[0] for token in batch["panoptic_mappings_list"][2]]
                    save_token_ind = 0
                    for tokenind, token in enumerate(tokens_obj):
                        if token == token_obj:
                            save_token_ind = tokenind
                    pts = []
                    lidardata = batch['point_clouds'][2].squeeze()[2].numpy()
                    for ptind, pt in enumerate(lidardata.T):
                        if batch['point_clouds_labels'][2].squeeze()[2][ptind] in labels_allowed:
                            pts.append(pt)

                    dts = np.array(pts)
                    bbox = batch["bottom_corners"][2][best_ann["token"][0]].squeeze().T.numpy()
                    
                    # filter out points within bbox
                    min_x = np.min(bbox[:, 0])
                    max_x = np.max(bbox[:, 0])
                    min_y = np.min(bbox[:, 1])
                    max_y = np.max(bbox[:, 1])
                    min_z = np.min(bbox[:, 2])
                    max_z = np.max(bbox[:, 2])
                    mask_x = (dts[:, 0] >= min_x) & (dts[:, 0] <= max_x)
                    mask_y = (dts[:, 1] >= min_y) & (dts[:, 1] <= max_y)
                    mask_z = (dts[:, 2] >= min_z) & (dts[:, 2] <= max_z)
                    mask = mask_x & mask_y & mask_z
                    indices = np.where(mask)
                    dts = dts[indices]
                    max_dist = max(abs(target_x - 100), abs(target_y - 100))
                    elem = int(min(np.ceil(((max_dist + 50)/100) * (len(dts) - 1)), len(dts) - 1))
                    try:
                        z_filter = sorted(dts[:, 2])[elem]
                        minind = np.argmin(np.linalg.norm(dts[:, :3] - np.array([[0.0, 0.0, z_filter]]), axis=1))
                        arr = np.expand_dims(dts[minind], axis=1).T
                    except:
                        continue

                    cam_img, points, matched_point, matched_cam = get_image_projection(batch['unnormalized_images'][0,2,0].numpy(), 
                                                                                batch['unnormalized_images'][0,2,1].numpy(), 
                                                                                batch['unnormalized_images'][0,2,2].numpy(), 
                                                                                batch['unnormalized_images'][0,2,3].numpy(), 
                                                                                batch['unnormalized_images'][0,2,4].numpy(), 
                                                                                batch['unnormalized_images'][0,2,5].numpy(), arr)
                    print(points, matched_cam, matched_point);
                    if matched_cam == None:
                        continue
                    obj["matched_cam"] = matched_cam
                    obj["matched_point"] = matched_point.tolist()
                    # Yes, the car is reversing and the rear park lights are on. This is indicated by the fact that the car is pulling into a parking space and the rear lights are illuminated. This is a common safety measure for drivers to ensure that other vehicles and pedestrians are aware of their presence and movements while reversing.
                    img_cropped, masks = extract_mask(predictor, cam_img, points)
                    matched_imgs.append(img_cropped)

                    user_message = "Given this image is of road scene, give detailed description of the central object in the image."
                    # llm_message = miniGPT4_inference(chat, img_cropped, user_message);
                    # obj['matched_point_cam'] = matched_point.tolist()
                    # obj['matched_cam'] = matched_cam
                    # obj['bev_coords'] = [x.astype(np.int32).tolist(), y.astype(np.int32).tolist()]

                    # conv = reset_conv();
                    # llm_message = llava_inference(model_llava, image_processor, tokenizer, conv, user_message, img_cropped, device);
                    # obj['llm_message_llava'] = llm_message
                    # llm_message = miniGPT4_inference(chat, img_cropped, user_message);
                    # obj['llm_message_minigpt4'] = llm_message
                    objects_json.append(obj)
                except:
                    print(f"SKIPPING {idx}")
                    continue

            for matched_img_ind, matched_img in enumerate(matched_imgs):
                np.save(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), f"{matched_img_ind + 1}_matched_img.npy"), matched_img)
            for cimg_ind, cimg in enumerate(cam_imgs):
                np.save(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), f"{cimg_ind + 1}_cimg.npy"), cimg)
            print("saving, ", os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), "answer_gt.json"))
            with open(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), "answer_gt.json"), "w") as f:
                json.dump(objects_json, f, indent=4)

            print("DONE SAVED");print();print();print();print();

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)
    args = parser.parse_args()
    eval(args.checkpoint, args.dataroot)