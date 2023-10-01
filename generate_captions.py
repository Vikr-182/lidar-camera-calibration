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
import pickle

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

from tqdm import tqdm
import argparse

cam_keys = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

# ========================================
#             MiniGPT4 Initialization
# ========================================
def init_minigp4():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--sam-checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="path to sam weights.")
    parser.add_argument('--model_path', type=str, default="/raid/t1/scratch/vikrant.dewangan/LLaVA/ckpt-old/", help='save path for jsons')
    parser.add_argument('--save_path', type=str, default="/raid/t1/scratch/vikrant.dewangan/datas", help='save path for jsons')
    parser.add_argument('--gpu', type=str, default="cuda:0", help='save path for jsons')
    parser.add_argument('--json_name', type=str, default="answer_pred_both.json", help='save path for jsons')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=100, help='end index')

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

def reset_conv(model_name = "llava"):
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    print('reset conv')

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    return conv

def minigpt4_inference(chat, img_cropped, user_message):
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

def run(index_start, index_end, gpu, model_path, save_path = "/raid/t1/scratch/vikrant.dewangan/datas", json_name="answer_gt.json"):
    json_list = os.listdir(save_path)
    num_to_json = {}
    for ind, objp in enumerate(json_list):
        if os.path.exists(os.path.join(save_path, objp, "answer_gt.json")):
            num_to_json[int(objp.split("_")[1])] = objp

    chat = init_minigp4()

    for ind in tqdm(range(index_start, index_end)):
        valid_jsons = pickle.load(open("refined_jsons.pkl", "rb"))
        if num_to_json[ind] not in valid_jsons:
            continue
        try:
            objs = json.load(open(os.path.join(save_path, num_to_json[ind], json_name)))
        except:
            continue
        imc = {}
        print(os.path.join(save_path, num_to_json[ind]))
        for j in range(6):
            img = np.load(os.path.join(save_path, num_to_json[ind], f"{j + 1}_cimg.npy"))
            user_message = "Describe this image."
            answer = minigpt4_inference(chat, img, user_message)
            imc[cam_keys[j]] = {}
            imc[cam_keys[j]]["description"] = answer

            user_message = "Is there anything unusual about this image?"
            answer = minigpt4_inference(chat, img, user_message)
            imc[cam_keys[j]]["unusual"] = answer

            user_message = "Describe the weather in this image. Is it day or night?"
            answer = minigpt4_inference(chat, img, user_message)
            imc[cam_keys[j]]["weather"] = answer
        for indd, obj in enumerate(objs):
            try:
                img = np.load(os.path.join(save_path, num_to_json[ind], f"{indd + 1}_matched_img_pred.npy"))
                user_message = "Describe the central object in this image."
                answer = minigpt4_inference(chat, img, user_message)
                objs[indd]["minigpt4_crop_brief"] = answer

                user_message = "Is the object's indicator on? If yes, what direction does it want to turn. Please answer carefully."
                answer = minigpt4_inference(chat, img, user_message)
                objs[indd]["minigpt4_crop_lights1"] = answer

                user_message = "In this image, are the object's rear lights closer  to the viewer or forward lights? Please answer in one word - forward/rear."
                answer = minigpt4_inference(chat, img, user_message)
                objs[indd]["minigpt4_crop_lights2"] = answer.lower()

                user_message = "Is there any text written on the object? Please look carefully, and describe it."
                answer = minigpt4_inference(chat, img, user_message)
                objs[indd]["minigpt4_crop_text"] = answer

                objs[indd]["minigpt4_bg_description"] = imc[objs[indd]["matched_cam"]]["description"]
                objs[indd]["minigpt4_bg_unusual"] = imc[objs[indd]["matched_cam"]]["unusual"]
                objs[indd]["minigpt4_bg_weather"] = imc[objs[indd]["matched_cam"]]["weather"]

            except:
                continue

        with open(os.path.join(save_path, num_to_json[ind], json_name), "w") as f:
            json.dump(objs, f, indent=4)
        print("DONE SAVED", os.path.join(save_path, num_to_json[ind]))

def main():
    parser = argparse.ArgumentParser(description='Generate captions.')    
    parser.add_argument('--model_path', type=str, default="/raid/t1/scratch/vikrant.dewangan/LLaVA/ckpt-old/", help='save path for jsons')
    parser.add_argument('--save_path', type=str, default="/raid/t1/scratch/vikrant.dewangan/datas", help='save path for jsons')
    parser.add_argument('--gpu', type=str, default="cuda:0", help='save path for jsons')
    parser.add_argument('--json_name', type=str, default="answer_pred_both.json", help='save path for jsons')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=100, help='end index')

    args = parser.parse_args()
    run(args.start, args.end, args.gpu, args.model_path, args.save_path, json_name=args.json_name)
    # run(0, 100, "cuda:0")

if __name__ == "__main__":
    main()