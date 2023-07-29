import plotly.graph_objects as go
import numpy as np
import torch

cpts = np.load("cam_rear_pts.npy")
camind = 0
#depths = np.load("depths.npy")[0,0,0].reshape(-1, 3)
#segs = np.load("segs.npy")[0].reshape(-1,1)

if True:
    classes = {
        "0": [0, 0, 0],         # None
        "1": [70, 70, 70],      # Buildings
        "2": [190, 153, 153],   # Fences
        "3": [72, 0, 90],       # Other
        "4": [220, 20, 60],     # Pedestrians
        "5": [153, 153, 153],   # Poles
        "6": [157, 234, 50],    # RoadLines
        "7": [128, 64, 128],    # Roads
        "8": [244, 35, 232],    # Sidewalks
        "9": [107, 142, 35],    # Vegetation
        "10": [0, 0, 255],      # Vehicles
        "11": [102, 102, 156],  # Walls
        "12": [220, 220, 0],     # TrafficSigns
        "13": [0, 0, 0],         # None
        "14": [0, 0, 0],         # None
        "15": [0, 0, 0],         # None
        "16": [0, 0, 0],         # None
        "17": [0, 0, 0],         # None
        "18": [0, 0, 0],         # None
        "19": [0, 0, 0],         # None
        "20": [0, 0, 0],         # None
        "21": [0, 0, 0],         # None
        "22": [0, 0, 0],         # None
        "23": [0, 0, 0],         # None
        "24": [0, 0, 0],         # None
        "25": [0, 0, 0],         # None
        "27": [0, 0, 0],         # None
        "27": [0, 0, 0],         # None
        "28": [0, 0, 0],         # None
    }


def rgb_to_hex(rgb: np.ndarray) -> str:
    rgb = np.array(rgb)
    rgb = rgb.reshape(3)
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

# Helix equation
cx = 128.0
cy = 128.0
fx = 167.8199
fy = 167.8199
pcd = []
depths = np.load("depths.npy")[0,0,0, ::8,::8]
segs = np.load("segs.npy")[0,0,::8,::8]
pcd = []
colors = []
h, w = depths.shape
seg_image=np.ones((32, 32, 3))
print(depths.shape)
for i in range(h):
    for j in range(w):
        # if segs[i, j] not in [6,7,10]:continue
        z = depths[i,j]
        x = (i - cx) * z/fx
        y = (j - cy) * z/fy
        pcd.append([x, y, z])
        seg_image[i, j] = np.array(classes[str(segs[i,j])])/256
        colors.append(rgb_to_hex(classes[str(segs[i,j])]))

import matplotlib.pyplot as plt
plt.imshow(seg_image)
plt.savefig("seg_image.png");plt.clf()

print(np.array(pcd).shape)
pcd = np.array(pcd)
import pdb; pdb.set_trace()
fig = go.Figure(data=[go.Scatter3d(x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], mode='markers',
    marker = dict(
            size=1,
            color=colors
        )
    )])
fig.show()
