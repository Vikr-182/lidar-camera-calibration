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

def run_minigpt4(index_start, index_end, gpu, model_path, save_path = "/raid/t1/scratch/vikrant.dewangan/datas", json_name="answer_gt.json"):
    json_list = os.listdir(save_path)
    num_to_json = {}
    for ind, objp in enumerate(json_list):
        if os.path.exists(os.path.join(save_path, objp, "answer_gt.json")):
            num_to_json[int(objp.split("_")[1])] = objp

    chat = init_minigp4()

    for ind in tqdm(range(index_start, index_end)):
        ultra_valid_jsons = ['c0d3796765e2452d990309b2568adac5_000263', 'cf10376382b74af1975be86f7190df20_000637', 'e3d714da71d2477695a055716ede7130_000893', 'f29904c2bcdc4d78bbcac75a2b30fe38_000962', '00590cbfa24a430a8c274b51e1c71231_000204', '1d4db80d13f342aba4881b38099bc4b7_000054', '2538a745a83f430eb590d48b4743e179_000276', '268099669c954f429087081530094337_000774', '28d385e6db0e495da3a606b58e2432f0_000682', '312102fd97de4bd4859374946e24ccdc_000618', '4098aaf3c7074e7d87285e2fc95369e0_000642', '4b48dfc43a3f411b88a71031d77e5696_000584', '4d475873416a4860900f5af213e0027c_000033', '7e3a6bdd6c6f4c8fb018cff404974446_000040', '7f8714d8ca814914bf4ff3ed30123db5_000551', '82aef599650d462db73731b7ff40918b_000065', '82aef599650d462db73731b7ff40918b_001029', '9047b53fd41540649dce014a128cbe1b_000071', '9047b53fd41540649dce014a128cbe1b_001035', '9eaabad9385b4fe1b9f397edead326ac_000267', '9eaabad9385b4fe1b9f397edead326ac_000268', 'a8819942e8bb43b6a832c56dfb78cfc6_000144', 'a99120daccb24bcd941b33e6e03bf718_000610', 'dcd5bc29543747e28ef02816dd458290_000134', 'f444b757d7e2444c889da10f02b73491_000067', '363bb6c0fdaf465aa54c39f082bb92c6_000512', '210add02013a4dfa84b7c5e23058781f_000343', '2f093cad7209436aa06e103bfe062857_000623', 'c0d3796765e2452d990309b2568adac5_000261', 'f123185f3bb64faebc08d4453aec2659_000564', '1d4db80d13f342aba4881b38099bc4b7_000056', '2538a745a83f430eb590d48b4743e179_000277', '36fbee38a28543ea9e27a67d64e1dee4_000202', '566311b8201e416f9f99463656dcadce_000601', '566311b8201e416f9f99463656dcadce_000602', '792b78f6cbcf413c821cb591630cddfb_000656', '9a1188aba4bf458c8220818a6c0be55a_000131', 'a1b51c02d8414856a86c0c37e4789c2f_000663', 'bd850592cd2541288b177b8e20baa31e_000522', 'f444b757d7e2444c889da10f02b73491_000068', 'b519ec833e23450a8bd3340b67f2516b_000747', '433a14f8dcf5457fb2c4def5c749122a_000127', '6d4b2bd795ae4c66900ad98ccd2371a6_000114', '6f5133fe62b240e797bac25aeff8b531_000011', '9a442b0f6b7a41568b3e4f7b7ba58402_000519', 'bd210a5bc7004d8c94b68cf6366612a2_000249', '17302a41218442ffbb0b094adb0669ab_000671', '2f093cad7209436aa06e103bfe062857_000622', '6746ba640c0e45c7961efbe6af51757e_000615', '6d4b2bd795ae4c66900ad98ccd2371a6_000115']
        # ultra_valid_jsons = ['c0d3796765e2452d990309b2568adac5_000263', 'cf10376382b74af1975be86f7190df20_000637', 'e3d714da71d2477695a055716ede7130_000893', 'f29904c2bcdc4d78bbcac75a2b30fe38_000962', '00590cbfa24a430a8c274b51e1c71231_000204', '1d4db80d13f342aba4881b38099bc4b7_000054', '2538a745a83f430eb590d48b4743e179_000276', '268099669c954f429087081530094337_000774', '28d385e6db0e495da3a606b58e2432f0_000682', '312102fd97de4bd4859374946e24ccdc_000618', '4098aaf3c7074e7d87285e2fc95369e0_000642', '4b48dfc43a3f411b88a71031d77e5696_000584', '4d475873416a4860900f5af213e0027c_000033', '7e3a6bdd6c6f4c8fb018cff404974446_000040', '7f8714d8ca814914bf4ff3ed30123db5_000551', '82aef599650d462db73731b7ff40918b_000065', '82aef599650d462db73731b7ff40918b_001029', '9047b53fd41540649dce014a128cbe1b_000071', '9047b53fd41540649dce014a128cbe1b_001035', '9eaabad9385b4fe1b9f397edead326ac_000267', '9eaabad9385b4fe1b9f397edead326ac_000268', 'a8819942e8bb43b6a832c56dfb78cfc6_000144', 'a99120daccb24bcd941b33e6e03bf718_000610', 'dcd5bc29543747e28ef02816dd458290_000134', 'f444b757d7e2444c889da10f02b73491_000067', '363bb6c0fdaf465aa54c39f082bb92c6_000512', '210add02013a4dfa84b7c5e23058781f_000343', '2f093cad7209436aa06e103bfe062857_000623', 'c0d3796765e2452d990309b2568adac5_000261', 'f123185f3bb64faebc08d4453aec2659_000564', '1d4db80d13f342aba4881b38099bc4b7_000056', '2538a745a83f430eb590d48b4743e179_000277', '36fbee38a28543ea9e27a67d64e1dee4_000202', '566311b8201e416f9f99463656dcadce_000601', '566311b8201e416f9f99463656dcadce_000602', '792b78f6cbcf413c821cb591630cddfb_000656', '9a1188aba4bf458c8220818a6c0be55a_000131', 'a1b51c02d8414856a86c0c37e4789c2f_000663', 'bd850592cd2541288b177b8e20baa31e_000522', 'f444b757d7e2444c889da10f02b73491_000068', 'b519ec833e23450a8bd3340b67f2516b_000747', '433a14f8dcf5457fb2c4def5c749122a_000127', '6d4b2bd795ae4c66900ad98ccd2371a6_000114', '6f5133fe62b240e797bac25aeff8b531_000011', '9a442b0f6b7a41568b3e4f7b7ba58402_000519', 'bd210a5bc7004d8c94b68cf6366612a2_000249', '17302a41218442ffbb0b094adb0669ab_000671', '2f093cad7209436aa06e103bfe062857_000622', '6746ba640c0e45c7961efbe6af51757e_000615', '6d4b2bd795ae4c66900ad98ccd2371a6_000115']
        valid_jsons = pickle.load(open("refined_jsons.pkl", "rb"))
        if num_to_json[ind] not in ultra_valid_jsons:
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
        
        print("Image Captioning done")
        for indd, obj in enumerate(objs):
            try:
                if "pred" in json_name:
                    img = np.load(os.path.join(save_path, num_to_json[ind], f"{indd + 1}_matched_img_pred.npy"))
                else:
                    img = np.load(os.path.join(save_path, num_to_json[ind], f"{indd + 1}_matched_img.npy"))
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
                print("BAAAAAAAAAAAAAAAAAAAA")
                continue

        import pdb; pdb.set_trace()
        save_path2 = "/raid/t1/scratch/vikrant.dewangan/datas5"
        os.makedirs(os.path.join(save_path2, num_to_json[ind]), exist_ok=True)
        with open(os.path.join(save_path2, num_to_json[ind], json_name), "w") as f:
            json.dump(objs, f, indent=4)
        print("DONE SAVED", os.path.join(save_path2, num_to_json[ind]))

def run_instructblip2(index_start, index_end, gpu, model_path, save_path = "/raid/t1/scratch/vikrant.dewangan/datas", json_name="answer_gt.json"):
    json_list = os.listdir(save_path)
    num_to_json = {}
    for ind, objp in enumerate(json_list):
        if os.path.exists(os.path.join(save_path, objp, "answer_gt.json")):
            num_to_json[int(objp.split("_")[1])] = objp

    model_instructblip, vis_processors = init_instructblip2(model_name = "blip2_vicuna_instruct", device="cuda:0")

    for ind in tqdm(range(index_start, index_end)):
        ultra_valid_jsons = ['c0d3796765e2452d990309b2568adac5_000263', 'cf10376382b74af1975be86f7190df20_000637', 'e3d714da71d2477695a055716ede7130_000893', 'f29904c2bcdc4d78bbcac75a2b30fe38_000962', '00590cbfa24a430a8c274b51e1c71231_000204', '1d4db80d13f342aba4881b38099bc4b7_000054', '2538a745a83f430eb590d48b4743e179_000276', '268099669c954f429087081530094337_000774', '28d385e6db0e495da3a606b58e2432f0_000682', '312102fd97de4bd4859374946e24ccdc_000618', '4098aaf3c7074e7d87285e2fc95369e0_000642', '4b48dfc43a3f411b88a71031d77e5696_000584', '4d475873416a4860900f5af213e0027c_000033', '7e3a6bdd6c6f4c8fb018cff404974446_000040', '7f8714d8ca814914bf4ff3ed30123db5_000551', '82aef599650d462db73731b7ff40918b_000065', '82aef599650d462db73731b7ff40918b_001029', '9047b53fd41540649dce014a128cbe1b_000071', '9047b53fd41540649dce014a128cbe1b_001035', '9eaabad9385b4fe1b9f397edead326ac_000267', '9eaabad9385b4fe1b9f397edead326ac_000268', 'a8819942e8bb43b6a832c56dfb78cfc6_000144', 'a99120daccb24bcd941b33e6e03bf718_000610', 'dcd5bc29543747e28ef02816dd458290_000134', 'f444b757d7e2444c889da10f02b73491_000067', '363bb6c0fdaf465aa54c39f082bb92c6_000512', '210add02013a4dfa84b7c5e23058781f_000343', '2f093cad7209436aa06e103bfe062857_000623', 'c0d3796765e2452d990309b2568adac5_000261', 'f123185f3bb64faebc08d4453aec2659_000564', '1d4db80d13f342aba4881b38099bc4b7_000056', '2538a745a83f430eb590d48b4743e179_000277', '36fbee38a28543ea9e27a67d64e1dee4_000202', '566311b8201e416f9f99463656dcadce_000601', '566311b8201e416f9f99463656dcadce_000602', '792b78f6cbcf413c821cb591630cddfb_000656', '9a1188aba4bf458c8220818a6c0be55a_000131', 'a1b51c02d8414856a86c0c37e4789c2f_000663', 'bd850592cd2541288b177b8e20baa31e_000522', 'f444b757d7e2444c889da10f02b73491_000068', 'b519ec833e23450a8bd3340b67f2516b_000747', '433a14f8dcf5457fb2c4def5c749122a_000127', '6d4b2bd795ae4c66900ad98ccd2371a6_000114', '6f5133fe62b240e797bac25aeff8b531_000011', '9a442b0f6b7a41568b3e4f7b7ba58402_000519', 'bd210a5bc7004d8c94b68cf6366612a2_000249', '17302a41218442ffbb0b094adb0669ab_000671', '2f093cad7209436aa06e103bfe062857_000622', '6746ba640c0e45c7961efbe6af51757e_000615', '6d4b2bd795ae4c66900ad98ccd2371a6_000115']
        if num_to_json[ind] not in ultra_valid_jsons:
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
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]] = {}
            imc[cam_keys[j]]["description"] = answer

            user_message = "Is there anything unusual about this image?"
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]]["unusual"] = answer

            user_message = "Describe the weather in this image. Is it day or night?"
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]]["weather"] = answer
        
        print("Image Captioning done")
        for indd, obj in enumerate(objs):
            try:
                if "pred" in json_name:
                    img = np.load(os.path.join(save_path, num_to_json[ind], f"{indd + 1}_matched_img_pred.npy"))
                else:
                    img = np.load(os.path.join(save_path, num_to_json[ind], f"{indd + 1}_matched_img.npy"))
                user_message = "Describe the central object in this image."
                answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
                objs[indd]["instructblip2_crop_brief"] = answer

                user_message = "Is the object's indicator on? If yes, what direction does it want to turn. Please answer carefully."
                answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
                objs[indd]["instructblip2_crop_lights1"] = answer

                user_message = "In this image, are the object's rear lights closer  to the viewer or forward lights? Please answer in one word - forward/rear."
                answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
                objs[indd]["instructblip2_crop_lights2"] = answer.lower()

                user_message = "Is there any text written on the object? Please look carefully, and describe it."
                answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
                objs[indd]["instructblip2_crop_text"] = answer

                objs[indd]["instructblip2_bg_description"] = imc[objs[indd]["matched_cam"]]["description"]
                objs[indd]["instructblip2_bg_unusual"] = imc[objs[indd]["matched_cam"]]["unusual"]
                objs[indd]["instructblip2_bg_weather"] = imc[objs[indd]["matched_cam"]]["weather"]

            except:
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                continue

        save_path2 = "/raid/t1/scratch/vikrant.dewangan/datas4"
        os.makedirs(os.path.join(save_path2, num_to_json[ind]), exist_ok=True)
        with open(os.path.join(save_path2, num_to_json[ind], json_name), "w") as f:
            json.dump(objs, f, indent=4)
        print("DONE SAVED", os.path.join(save_path2, num_to_json[ind]))

def run_blip2(index_start, index_end, gpu, model_path, save_path = "/raid/t1/scratch/vikrant.dewangan/datas", json_name="answer_gt.json"):
    json_list = os.listdir(save_path)
    num_to_json = {}
    for ind, objp in enumerate(json_list):
        if os.path.exists(os.path.join(save_path, objp, "answer_gt.json")):
            num_to_json[int(objp.split("_")[1])] = objp

    model_instructblip, vis_processors = init_blip2(model_name = "blip2_t5", device="cuda:0", model_type="pretrain_flant5xxl")

    for ind in tqdm(range(index_start, index_end)):
        night_jsons = [
            "4ed628299b1e45a3a73704e8cf8287a9",
            "4ed628299b1e45a3a73704e8cf8287a9",
            "865c607c8ef44f13b39744a0de110740",
            "26c089d84086473e87607ae5c6ae85c6",
            "a2b64d02c5fa4b9bab671a97561b6b33",
            "cfa36eca40364e5bb15f550a077a21c5",
            "04e5f089805540a5a9e73c1b7c54ee8c",
            "e3b6fe9201c64334be311d4249a0c80e",
            "e5514b8f435e475cac4eba74b44773b5",
            "89f20737ec344aa48b543a9e005a38ca",
            "57146b2ebf10432f84b15a3038fe1755",
            "ca6abd081eaf48689f06b5e8fcc9d369",
            "f30849d00388491fa6997a13d56c73fd",
            "6e820d4c44b74793aacd0c7091488865",
            "270fe8382f884386b3445143fa946200",
            "34a9823fac0d4b9db30898e00b5f9f9c",
            "d9161e521b1644cea4cb9e3f21ef9f79",
            "cd9db61edff14e8784678abb347cd674",
            "16c90eedfc7943a5a4c64e84d18876a2",
            "a1e7cd557b9541dbb00822ea0c56204b",
            "f143809fc26c4bb296f4f367b0954c26"
        ]
        ultra_valid_jsons = ['c0d3796765e2452d990309b2568adac5_000263', 'cf10376382b74af1975be86f7190df20_000637', 'e3d714da71d2477695a055716ede7130_000893', 'f29904c2bcdc4d78bbcac75a2b30fe38_000962', '00590cbfa24a430a8c274b51e1c71231_000204', '1d4db80d13f342aba4881b38099bc4b7_000054', '2538a745a83f430eb590d48b4743e179_000276', '268099669c954f429087081530094337_000774', '28d385e6db0e495da3a606b58e2432f0_000682', '312102fd97de4bd4859374946e24ccdc_000618', '4098aaf3c7074e7d87285e2fc95369e0_000642', '4b48dfc43a3f411b88a71031d77e5696_000584', '4d475873416a4860900f5af213e0027c_000033', '7e3a6bdd6c6f4c8fb018cff404974446_000040', '7f8714d8ca814914bf4ff3ed30123db5_000551', '82aef599650d462db73731b7ff40918b_000065', '82aef599650d462db73731b7ff40918b_001029', '9047b53fd41540649dce014a128cbe1b_000071', '9047b53fd41540649dce014a128cbe1b_001035', '9eaabad9385b4fe1b9f397edead326ac_000267', '9eaabad9385b4fe1b9f397edead326ac_000268', 'a8819942e8bb43b6a832c56dfb78cfc6_000144', 'a99120daccb24bcd941b33e6e03bf718_000610', 'dcd5bc29543747e28ef02816dd458290_000134', 'f444b757d7e2444c889da10f02b73491_000067', '363bb6c0fdaf465aa54c39f082bb92c6_000512', '210add02013a4dfa84b7c5e23058781f_000343', '2f093cad7209436aa06e103bfe062857_000623', 'c0d3796765e2452d990309b2568adac5_000261', 'f123185f3bb64faebc08d4453aec2659_000564', '1d4db80d13f342aba4881b38099bc4b7_000056', '2538a745a83f430eb590d48b4743e179_000277', '36fbee38a28543ea9e27a67d64e1dee4_000202', '566311b8201e416f9f99463656dcadce_000601', '566311b8201e416f9f99463656dcadce_000602', '792b78f6cbcf413c821cb591630cddfb_000656', '9a1188aba4bf458c8220818a6c0be55a_000131', 'a1b51c02d8414856a86c0c37e4789c2f_000663', 'bd850592cd2541288b177b8e20baa31e_000522', 'f444b757d7e2444c889da10f02b73491_000068', 'b519ec833e23450a8bd3340b67f2516b_000747', '433a14f8dcf5457fb2c4def5c749122a_000127', '6d4b2bd795ae4c66900ad98ccd2371a6_000114', '6f5133fe62b240e797bac25aeff8b531_000011', '9a442b0f6b7a41568b3e4f7b7ba58402_000519', 'bd210a5bc7004d8c94b68cf6366612a2_000249', '17302a41218442ffbb0b094adb0669ab_000671', '2f093cad7209436aa06e103bfe062857_000622', '6746ba640c0e45c7961efbe6af51757e_000615', '6d4b2bd795ae4c66900ad98ccd2371a6_000115']
        # if num_to_json[ind] not in ultra_valid_jsons:
        if num_to_json[ind].split("_")[0] not in night_jsons:
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
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]] = {}
            imc[cam_keys[j]]["description"] = answer

            user_message = "Is there anything unusual about this image?"
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]]["unusual"] = answer

            user_message = "Describe the weather in this image. Is it day or night?"
            answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
            imc[cam_keys[j]]["weather"] = answer
        
        print("Image Captioning done")
        for indd, obj in enumerate(objs):
            try:
                if "pred" in json_name:
                    img = np.load(os.path.join(save_path, num_to_json[ind], f"{indd + 1}_matched_img_pred.npy"))
                else:
                    img = np.load(os.path.join(save_path, num_to_json[ind], f"{indd + 1}_matched_img.npy"))
                user_message = "Describe the central object in this image."
                answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
                objs[indd]["blip2_crop_brief"] = answer

                user_message = "Is the object's indicator on? If yes, what direction does it want to turn. Please answer carefully."
                answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
                objs[indd]["blip2_crop_lights1"] = answer

                user_message = "In this image, are the object's rear lights closer  to the viewer or forward lights? Please answer in one word - forward/rear."
                answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
                objs[indd]["blip2_crop_lights2"] = answer.lower()

                user_message = "Is there any text written on the object? Please look carefully, and describe it."
                answer = instructblip2_inference(model_instructblip, img, vis_processors, device="cuda:0", user_message=user_message)
                objs[indd]["blip2_crop_text"] = answer

                objs[indd]["blip2_bg_description"] = imc[objs[indd]["matched_cam"]]["description"]
                objs[indd]["blip2_bg_unusual"] = imc[objs[indd]["matched_cam"]]["unusual"]
                objs[indd]["blip2_bg_weather"] = imc[objs[indd]["matched_cam"]]["weather"]

            except:
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
                continue

        import pdb; pdb.set_trace()
        save_path2 = "/raid/t1/scratch/vikrant.dewangan/datas7"
        os.makedirs(os.path.join(save_path2, num_to_json[ind]), exist_ok=True)
        with open(os.path.join(save_path2, num_to_json[ind], json_name), "w") as f:
            json.dump(objs, f, indent=4)
        print("DONE SAVED", os.path.join(save_path2, num_to_json[ind]))

def main():
    parser = argparse.ArgumentParser(description='Generate captions.')    
    parser.add_argument('--model_path', type=str, default="/raid/t1/scratch/vikrant.dewangan/LLaVA/ckpt-old/", help='save path for jsons')
    parser.add_argument('--save_path', type=str, default="/raid/t1/scratch/vikrant.dewangan/datas", help='save path for jsons')
    parser.add_argument('--gpu', type=str, default="cuda:0", help='save path for jsons')
    parser.add_argument('--json_name', type=str, default="answer_pred_both.json", help='save path for jsons')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=100, help='end index')
    parser.add_argument('--model', type=str, required=True, help='model name, use minigpt4 or instructblip2 or blip2')

    args = parser.parse_args()
    if args.model == "minigpt4":
        run_minigpt4(args.start, args.end, args.gpu, args.model_path, args.save_path, json_name=args.json_name)
    elif args.model == "instructblip2":
        run_instructblip2(args.start, args.end, args.gpu, args.model_path, args.save_path, json_name=args.json_name)
    elif args.model == "blip2":
        run_blip2(args.start, args.end, args.gpu, args.model_path, args.save_path, json_name=args.json_name)

    # run(0, 100, "cuda:0")

if __name__ == "__main__":
    main()