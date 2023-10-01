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
#             InsructBLIP-2 Model Initialization
# ========================================
def init_instructblip2(model_name = "blip2_vicuna_instruct", device="cuda:0"):
    model, vis_processors, _ = load_model_and_preprocess(
        name=model_name,
        model_type="vicuna13b",
        is_eval=True,
        device=device,
    )
    return model, vis_processors
# ========================================

# ========================================
#             BLIP-2 Model Initialization
# ========================================
def init_blip2(model_name = "blip2_vicuna_instruct", device="cuda:0", model_type="vicuna13b"):
    model, vis_processors, _ = load_model_and_preprocess(
        name=model_name,
        model_type=model_type,
        is_eval=True,
        device=device,
    )
    return model, vis_processors
# ========================================

# ========================================
#             MiniGPT4 Initialization
# ========================================
def init_minigp4(device="cuda:0"):
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
    chat = Chat(model, vis_processor, device=device)
    return chat
# ========================================

# ========================================
#             LLaVa Model Initialization
# ========================================
def init_llava(model_name = "llava", model_path = "/raid/t1/scratch/vikrant.dewangan/LLaVA/ckpt-old/", model_base=None, load_8bit=True, load_4bit=False):
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit)
    return tokenizer, model, image_processor, context_len
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

def instructblip2_inference(model_instructblip, img_cropped, vis_processors, device="cuda:0", user_message="describe the central object in the scene."):
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

# LLaVa
tokenizer, model_llava, image_processor, context_len = init_llava()
print("Initializaed LLaVa")

# InstructBLIP-2
model_instructblip, vis_processors = init_instructblip2(device="cuda:1")
print("Initializaed Instruct-BLIP2")

# MiniGPT-4
chat = init_minigp4(device="cuda:0")
print("Initializaed MiniGPT-4")

model_blip2, vis_processors = init_blip2(model_name = "blip2_t5", device="cuda:3", model_type="pretrain_flant5xxl")
# BLIP-2
print("Initializaed BLIP-2")

import pdb; pdb.set_trace()

user_message = "describe the central object in the scene."
img_cropped = np.load("imgs/qualitative_anushka/3_matched_img.npy")
conv = reset_conv()
llava_message = llava_inference(model_llava, image_processor, tokenizer, conv, user_message, img_cropped, device="cuda:0")
minigpt4_message = minigpt4_inference(chat, img_cropped, user_message)
iblip2_message = instructblip2_inference(model_instructblip, img_cropped, vis_processors, device="cuda:1", user_message="describe the central object in the scene.")
blip2_message = instructblip2_inference(model_blip2, img_cropped, vis_processors, device="cuda:3", user_message="describe the central object in the scene.")