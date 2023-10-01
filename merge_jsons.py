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

# extract the jsons first
data_path = "/scratch/talk2bev/datas"
save_path = "/scratch/talk2bev/jsons"
json_list = os.listdir(data_path)
num_to_json = {}
for ind, objp in enumerate(json_list):
    if os.path.exists(os.path.join(data_path, objp, "answer_gt.json")):
        num_to_json[int(objp.split("_")[1])] = objp

for ind in range(0, 800):
    try:
        objs = json.load(open(os.path.join(save_path, num_to_json[ind], "answer_gt.json")))
    except:
        print("not able to do ", ind)
        continue
    with open(os.path.join(save_path, num_to_json[ind], "answer_captioned_gt.json"), "w") as f:
        json.dump(objs, f, indent=4)
