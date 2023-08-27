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

# ========================================
#             InsructBLIP-2 Model Initialization
# ========================================
def init_instructblip2(model_name = "blip2_vicuna_instruct", device="cuda:0"):
    model, vis_processors, _ = load_model_and_preprocess(
        name=model_name,
        model_type="vicuna7b",
        is_eval=True,
        device=device,
    )
    return model, vis_processors
# ========================================

# ========================================
#             LLaVa Model Initialization
# ========================================
def init_llava(model_name = "llava", model_path = "/raid/t1/scratch/vikrant.dewangan/LLaVA/ckpt-old/", model_base=None, load_8bit=True, load_4bit=False):
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit)
    return tokenizer, model, image_processor, context_len
# ========================================

# ========================================
#             MiniGPT4 Initialization
# ========================================
def init_minigp4():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--sam-checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="path to sam weights.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    print('Initializing Chat')
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    return chat
# ========================================

# ========================================
#             SAM Initialization
# ========================================
def init_sam(model_type = "vit_h", checkpoint="/home/t1/vikrant.dewangan/llm-bev/MiniGPT-4/sam_vit_h_4b8939.pth", device="cuda:0"):
    from segment_anything import sam_model_registry, SamPredictor


    sam = sam_model_registry[model_type](checkpoint="/home/t1/vikrant.dewangan/llm-bev/MiniGPT-4/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor
# ========================================

def reset_conv(model_name = "llava"):
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    print('Initialization Finished')

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    return conv

def llava_inference(model_llava, image_processor, tokenizer, conv, user_message, image, device="cuda"):
  image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(device)
  if model_llava.config.mm_use_im_start_end:
      inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_message
  else:
      inp = DEFAULT_IMAGE_TOKEN + '\n' + user_message
  conv.append_message(conv.roles[0], inp)
  prompt = conv.get_prompt()
  input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
  stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
  keywords = [stop_str]
  stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
  streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
  output_ids = model_llava.generate(
      input_ids,
      images=image_tensor,
      do_sample=True,
      temperature=0.5,
      max_new_tokens=1024,
      streamer=streamer,
      use_cache=True,
      stopping_criteria=[stopping_criteria])

  outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
  return outputs

def miniGPT4_inference(chat, img_cropped, user_message):
    img_list = []
    chat_state = CONV_VISION.copy()  # Reset chat state to default template
    llm_message = chat.upload_img(Image.fromarray(img_cropped), chat_state, img_list)

    print('Upload done')

    chat.ask(user_message, chat_state)
    llm_message = chat.answer(
            conv=chat_state,
            img_list=img_list,
            # num_beams=num_beams,
            num_beams=1,
            # temperature=temperature,
            temperature=0.7,
            max_new_tokens=300,
            max_length=2000
    )[0]
    return llm_message

def instructblip2_inference(img_cropped, vis_processors, device="cuda", user_message="describe the central object in the scene."):
    image = vis_processors["eval"](Image.fromarray(img_cropped)).unsqueeze(0).to(device)

    samples = {
        "image": image,
        "prompt": user_message,
    }

    output_blip = model_instructblip.generate(
        samples,
        length_penalty=float(1),
        repetition_penalty=float(1),
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.2,
        use_nucleus_sampling=False,
    )

    return output_blip[0]

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

def get_image_projected_points(translation, rotation, camera_intrinsic, arr):
    matched_points = None
    flag = 0
    min_dist = 1.0
    barr = np.copy(arr)
    ppc = LidarPointCloud(barr.T);
    ppc.translate(translation)
    ppc.rotate(rotation)
    ddepths_img = ppc.points[2, :]
    points = view_points(ppc.points[:3, :], np.array(camera_intrinsic), normalize=True)
    
    mask = np.ones(ddepths_img.shape[0], dtype=bool)
    mask = np.logical_and(mask, ddepths_img > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < 1600 - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < 900 - 1)

    if mask.sum() > 0:
        flag = 1
        points = points[:, mask]
        ddepths_img = ddepths_img[mask]
        matched_points = points
    return flag, matched_points

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
device2 = torch.device('cuda:0')
to_visualize = True
compute_losses = False

data_path = "/raid/t1/scratch/vikrant.dewangan/v1.0-trainval"
save_path = "/raid/t1/scratch/vikrant.dewangan/datas"

# LLaVa
tokenizer, model_llava, image_processor, context_len = init_llava()
print("Initializaed LLaVa")

# # InstructBLIP-2
# model_instructblip, vis_processors = init_instructblip2(device=device2)
# print("Initializaed Instruct-BLIP2")

# # MiniGPT-4
# chat = init_minigp4()
# print("Initializaed MiniGPT4")


predictor = init_sam(device=device2)
print("Initializaed SAM")
print('Initialization Finished')

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
    valdata.indices = valdata.indices[18035:]
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )

    for index, batch in enumerate(tqdm(valloader)):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']


        with torch.no_grad():
            output = model(
                image.to(device), intrinsics.to(device), extrinsics.to(device), future_egomotion.to(device)
            )

            pred = torch.argmax(output['segmentation'], dim=2).squeeze()[2].cpu().numpy()
            arr = np.zeros((200, 200, 3))
            whe = np.where(batch['hdmap'].squeeze()[2,1] > 0)
            arr[whe[0], whe[1]] = np.array([255,255,255])
            # whe = np.where(batch['hdmap'].squeeze()[2,0] > 0)
            # arr[whe[0], whe[1]] = np.array([255,255,0])
            whe = np.where(pred > 0)
            arr[whe[0], whe[1]] = np.array([0,0,255])
            barr = np.copy(arr)
            varr = np.copy(arr)

            labels_allowed = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            pts = []
            lidardata = batch['point_clouds'][2].squeeze()[2].numpy()

            bev_2d = np.logical_and(barr[:,:,2]>0,barr[:,:,0]==0)
            labels, pts_ = cv2.connectedComponents(bev_2d.astype(np.uint8))
            matched_imgs = []

            objects_json = []
            objects_chatgpt_json = []
            # front_ids = [9]
            # right_ids = [7, 8, 10]

            interesting_cars = [3,6]
            for idx in tqdm(range(1, labels)):
                # Create a JSON object for each component                    
                x, y = np.where(pts_ == idx)
                # y = 200-y

                obj = {
                    "object_id": idx,
                    "bev_centroid": [(np.mean(x).astype(np.int) - 100)/2, (np.mean(y).astype(np.int) - 100)/2],
                    "matched_coords": [x.tolist(), y.tolist()],
                    "bev_area": len(x)/4,
                }
                
                obj2 = {
                    "object_id": idx,
                    "bev_centroid": [(np.mean(x).astype(np.int) - 100)/2, (np.mean(y).astype(np.int) - 100)/2],
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
                print(min_ann_dist, " min_ann_dist")
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
                pts = []
                lidardata = batch['point_clouds'][2].squeeze()[2].numpy()
                for ptind, pt in enumerate(lidardata.T):
                    if batch['point_clouds_labels'][2].squeeze()[2][ptind] in labels_allowed:
                        pts.append(pt)

                dts = np.array(pts)
                bbox = batch["bottom_corners"][2][best_ann["token"][0]].squeeze().T.numpy()
                print(np.mean(bbox, axis=0), target)
                
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
                print("max_dist: ", max_dist)
                elem = int(min(np.ceil(((max_dist + 50)/100) * (len(dts) - 1)), len(dts) - 1))
                print("elem filter: ", elem, ", len: ", len(dts));
                try:
                    z_filter = sorted(dts[:, 2])[elem]
                    minind = np.argmin(np.linalg.norm(dts[:, :3] - np.array([[0.0, 0.0, z_filter]]), axis=1))
                    arr = np.expand_dims(dts[minind], axis=1).T
                except:
                    continue
                calibration_data = json.load(open("calibration.json"))
                cam_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
                cam_img, points, matched_point, matched_cam = get_image_projection(batch['unnormalized_images'][0,2,0].numpy(), 
                                                                               batch['unnormalized_images'][0,2,1].numpy(), 
                                                                               batch['unnormalized_images'][0,2,2].numpy(), 
                                                                               batch['unnormalized_images'][0,2,3].numpy(), 
                                                                               batch['unnormalized_images'][0,2,4].numpy(), 
                                                                               batch['unnormalized_images'][0,2,5].numpy(), arr)
                print(points, matched_cam, matched_point);
                if matched_cam == None:
                    continue
                if idx in interesting_cars:
                    varr[x, y] = np.array([255, 0, 0])
                    for j in range(6):
                        flag, matched_points = get_image_projected_points(np.array(calibration_data[cam_keys[j]]['translation']), np.array(calibration_data[cam_keys[j]]['rotation']), calibration_data[cam_keys[j]]['camera_intrinsic'], arr)
                        if not flag:
                            continue
                        img_cropped, masks = extract_mask(predictor, batch['unnormalized_images'][0,2,j].numpy(), matched_points)
                        idxs = (masks[-1].astype(np.uint8) * 200) > 0
                        batch['unnormalized_images'][0,2,j].numpy()[idxs] = batch['unnormalized_images'][0,2,j].numpy()[idxs] + (np.array([255, 0, 0]) - batch['unnormalized_images'][0,2,j].numpy()[idxs]) * 0.5
                else:
                    continue

                # Yes, the car is reversing and the rear park lights are on. This is indicated by the fact that the car is pulling into a parking space and the rear lights are illuminated. This is a common safety measure for drivers to ensure that other vehicles and pedestrians are aware of their presence and movements while reversing.
                img_cropped, masks = extract_mask(predictor, cam_img, points)
                matched_imgs.append(img_cropped)
                cam_keys_mapping = {'CAM_FRONT_LEFT':0, 'CAM_FRONT':1, 'CAM_FRONT_RIGHT':2, 'CAM_BACK_LEFT':3, 'CAM_BACK':4, 'CAM_BACK_RIGHT':5}
                user_message = "Given this image is of road scene, describe the central object in the image. Is the indicator on? Which direction is the indicator suggesting that vehicle will go? Answer in terms of direction, left or right."
                conv = reset_conv()
                llm_message_llava = llava_inference(model_llava, image_processor, tokenizer, conv, user_message, img_cropped, device);
                # import pdb; pdb.set_trace()
                obj['llm_message_llava'] = llm_message_llava
                obj2['llm_message_llava'] = llm_message_llava
                img = batch['unnormalized_images'][0,2,cam_keys_mapping[matched_cam]]
                darr = np.copy(barr)
                plt.imshow(img)
                plt.scatter(matched_point[0], matched_point[1], color='red')
                plt.savefig("test.png")
                plt.clf()
                try:
                    darr[x, y] = np.array([255, 0, 0])
                    plt.imshow(darr/256)
                    plt.savefig("testt_bev.png")
                    plt.clf()
                except:
                    pass
                plt.imshow(img_cropped)
                plt.savefig("testt_cropped.png")
                plt.clf()

                user_message = "Is the object facing towards or away from the camera? Reply in one word - towards or away."
                conv = reset_conv()
                llm_message_llava = llava_inference(model_llava, image_processor, tokenizer, conv, user_message, img_cropped, device);
                direction = 1
                d1 = llm_message_llava.lower() == "towards"
                d2 = np.mean(x) < 0
                obj['travel_direction'] = {0:'same', 1: 'opposite'}[d1 ^ d2]
                objects_json.append(obj)
                objects_chatgpt_json.append(obj2)

            with open("answer_gt_shubham2.json", "w") as f:
                json.dump(objects_json, f, indent=4)
            with open("answer_chatgpt_gt_shubham2.json", "w") as f:
                json.dump(objects_chatgpt_json, f, indent=4)
            for matched_img_ind, matched_img in enumerate(matched_imgs):
                np.save(os.path.join(f"imgs/qualitative_shubham2/{matched_img_ind + 1}_matched_img.npy"), matched_img)
                Image.fromarray(matched_img).save(os.path.join(f"imgs/qualitative_shubham2/{matched_img_ind + 1}_matched_img.png"))

            fig = plt.figure(figsize=(7, 3))

            gs = gridspec.GridSpec(6, 18)

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

            plt.tight_layout()
            plt.savefig("6images_shubham.png", dpi=300)

            Image.fromarray(varr.astype(np.uint8)).save("arr_shubham.png")
            import pdb;pdb.set_trace()
            print(" DONE SAVED \n\n")
            exit()
            break
            continue

            pred = torch.argmax(output['segmentation'], dim=2).squeeze()[2].cpu().numpy()
            arr = np.zeros((200, 200, 3))
            whe = np.where(batch['hdmap'].squeeze()[2,1] > 0)
            arr[whe[0], whe[1]] = np.array([255,255,255])
            # whe = np.where(batch['hdmap'].squeeze()[2,0] > 0)
            # arr[whe[0], whe[1]] = np.array([255,255,0])
            whe = np.where(pred > 0)
            arr[whe[0], whe[1]] = np.array([0,0,255])
            barr = np.copy(arr)
            bev_2d = np.logical_and(barr[:,:,2]>0,barr[:,:,0]==0)
            labels, pts_ = cv2.connectedComponents(bev_2d.astype(np.uint8))
            matched_imgs = []

            objects_json = []
            for idx in tqdm(range(1, labels)):
                # Create a JSON object for each component
                y, x = np.where(pts_ == idx)
                y = 200-y
                obj = {
                    "object_id": idx,
                    "bev_centroid": [(np.mean(x).astype(np.int) - 100)/2, (np.mean(y).astype(np.int) - 100)/2],
                    # "matched_coords": [x.tolist(), y.tolist()],
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
                print(min_ann_dist, " min_ann_dist")
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
                pts = []
                lidardata = batch['point_clouds'][2].squeeze()[2].numpy()
                for ptind, pt in enumerate(lidardata.T):
                    if batch['point_clouds_labels'][2].squeeze()[2][ptind] in labels_allowed:
                        pts.append(pt)

                dts = np.array(pts)
                bbox = batch["bottom_corners"][2][best_ann["token"][0]].squeeze().T.numpy()
                print(np.mean(bbox, axis=0), target)
                
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
                print("max_dist: ", max_dist)
                elem = int(min(np.ceil(((max_dist + 50)/100) * (len(dts) - 1)), len(dts) - 1))
                print("elem filter: ", elem, ", len: ", len(dts));
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
                img_cropped = extract_mask(predictor, cam_img, points)
                matched_imgs.append(img_cropped)
                cam_keys_mapping = {'CAM_FRONT_LEFT':0, 'CAM_FRONT':1, 'CAM_FRONT_RIGHT':2, 'CAM_BACK_LEFT':3, 'CAM_BACK':4, 'CAM_BACK_RIGHT':5}
                user_message = "Given this image is of road scene, describe the central object in the image."
                # llm_message_minigpt4 = miniGPT4_inference(chat, img_cropped, user_message)
                # llm_message_instructblip2 = instructblip2_inference(img_cropped, vis_processors, device2)
                conv = reset_conv()
                llm_message_llava = llava_inference(model_llava, image_processor, tokenizer, conv, user_message, img_cropped, device);

                # obj['llm_message'] = llm_message_instructblip2
                # obj['llm_message_minigpt4'] = llm_message_minigpt4
                obj['llm_message_llava'] = llm_message_llava
                # obj['llm_message_instructblip2'] = llm_message_instructblip2
                objects_json.append(obj)
            with open("answer_pred.json", "w") as f:
                json.dump(objects_json, f, indent=4)
        break



if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)
    args = parser.parse_args()
    eval(args.checkpoint, args.dataroot)
