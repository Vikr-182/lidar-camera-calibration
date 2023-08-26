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
import time
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

def llava_inference(image_processor, tokenizer, conv, user_message, image, device="cuda"):
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

def get_image_projection(cam_left, cam_front, cam_right, cam_rear_left, cam_rear, cam_rear_right, arr):
    cam_imgs = [cam_left, cam_front, cam_right, cam_rear_left, cam_rear, cam_rear_right]
    cam_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    calibration_data = {
        'CAM_FRONT_LEFT':{
            'translation': [-1.57525595, -0.50051938, -1.50696033],
            'rotation': [[ 0.82254604, -0.56868433, -0.00401771], [ 0.00647832,  0.01643407, -0.99984396], [ 0.56866162,  0.82239167,  0.01720189]],
            'camera_intrinsic': [[1257.8625342125129, 0.0, 827.2410631095686], [0.0, 1257.8625342125129, 450.915498205774], [0.0, 0.0, 1.0]]
        },
        'CAM_FRONT':{
            'translation': [-1.72200568, -0.00475453, -1.49491292],
            'rotation': [[ 0.01026021, -0.99987258, -0.01222952], [ 0.00843345,  0.01231626, -0.99988859], [ 0.9999118 ,  0.01015593,  0.00855874]],
            'camera_intrinsic': [[1252.8131021185304, 0.0, 826.588114781398], [0.0, 1252.8131021185304, 469.9846626224581], [0.0, 0.0, 1.0]]
        },
        'CAM_FRONT_RIGHT':{
            'translation': [-1.58082566,  0.49907871, -1.51749368],
            'rotation': [[-0.84397973, -0.53614138, -0.01583178], [ 0.01645551,  0.00362107, -0.99985804], [ 0.5361226 , -0.84412044,  0.00576637]],
            'camera_intrinsic': [[1256.7485116440405, 0.0, 817.7887570959712], [0.0, 1256.7485116440403, 451.9541780095127], [0.0, 0.0, 1.0]]
        },
        'CAM_BACK_LEFT':{
            'translation': [-1.035691  , -0.48479503, -1.59097015],
            'rotation': [[ 0.94776036,  0.31896113,  0.00375564], [ 0.00866572, -0.0139763 , -0.99986478], [-0.31886551,  0.94766474, -0.01601021]],
            'camera_intrinsic': [[1256.7414812095406, 0.0, 792.1125740759628], [0.0, 1256.7414812095406, 492.7757465151356], [0.0, 0.0, 1.0]]
        },
        'CAM_BACK':{
            'translation': [-0.02832603, -0.00345137, -1.57910346],
            'rotation': [[ 0.00242171,  0.99998907, -0.00400023], [-0.01675361, -0.00395911, -0.99985181], [-0.99985672,  0.00248837,  0.01674384]],
            'camera_intrinsic': [[809.2209905677063, 0.0, 829.2196003259838], [0.0, 809.2209905677063, 481.77842384512485], [0.0, 0.0, 1.0]]
        },
        'CAM_BACK_RIGHT':{
            'translation': [-1.0148781 ,  0.48056822, -1.56239545],
            'rotation': [[-0.93477554,  0.35507456, -0.01080503], [ 0.01587584,  0.0113705 , -0.99980932], [-0.35488399, -0.93476883, -0.01626597]],
            'camera_intrinsic': [[1259.5137405846733, 0.0, 807.2529053838625], [0.0, 1259.5137405846733, 501.19579884916527], [0.0, 0.0, 1.0]]
        }
    }

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
    return img_copy

device = torch.device('cuda:0')
device2 = torch.device('cuda:0')
to_visualize = True
compute_losses = False

data_path = "/raid/t1/scratch/vikrant.dewangan/v1.0-trainval"
save_path = "/raid/t1/scratch/vikrant.dewangan/datas_new"

# InstructBLIP-2
model_instructblip, vis_processors = init_instructblip2(device=device2)
print("Initializaed Instruct-BLIP2")

from_scratch = True

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
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )

    for index, batch in enumerate(tqdm(valloader)):
        cur_scene_token = batch['scene_token'][0]
        with torch.no_grad():
            os.makedirs(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index)), exist_ok=True)
            arr = np.zeros((200, 200, 3))
            whe = np.where(batch['hdmap'].squeeze()[2,1] > 0)
            arr[whe[0], whe[1]] = np.array([255,255,255])
            # whe = np.where(batch['hdmap'].squeeze()[2,0] > 0)
            # arr[whe[0], whe[1]] = np.array([255,255,0])
            whe = np.where(batch['segmentation'].squeeze()[2] > 0)
            arr[whe[0], whe[1]] = np.array([0,0,255])
            Image.fromarray(arr.astype(np.uint8)).save(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), "gt_bev.png"))
            barr = np.copy(arr)

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
            try:
                objects_json = json.load(open(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), "answer.json")))
                from_scratch = False
            except:
                objects_json = []
                from_scratch = True

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
                
                user_message = "Describe the central object in brief."
                llm_message_instructblip2 = instructblip2_inference(img_cropped, vis_processors, device2, user_message=user_message)
                obj['llm_message'] = llm_message_instructblip2
                obj['llm_message_instructblip2'] = llm_message_instructblip2
                obj['matched_point_cam'] = matched_point.tolist()
                obj['matched_cam'] = matched_cam
                obj['bev_coords'] = [x.astype(np.int32).tolist(), y.astype(np.int32).tolist()]
                if from_scratch:
                    objects_json.append(obj)
                else:
                    objects_json[idx - 1]['llm_message'] = llm_message_instructblip2
                    objects_json[idx - 1]['llm_message_instructblip2'] = llm_message_instructblip2

            print("saving, ", os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index)))
            for matched_img_ind, matched_img in enumerate(matched_imgs):
                np.save(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), f"{matched_img_ind + 1}_matched_img.npy"), matched_img)

            for cimg_ind, cimg in enumerate(cam_imgs):
                np.save(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), f"{cimg_ind + 1}_cimg.npy"), cimg)
        
            with open(os.path.join(save_path, str(cur_scene_token[0]) + "_" + "{0:0=6d}".format(index), "answer.json"), "w") as f:
                json.dump(objects_json, f, indent=4)

            print("DONE SAVED");print();print();print();print();

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)
    args = parser.parse_args()
    eval(args.checkpoint, args.dataroot)
