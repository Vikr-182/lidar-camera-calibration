import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

from segment_anything import sam_model_registry, SamPredictor

model_type = "vit_h"
checkpoint="/home/t1/vikrant.dewangan/llm-bev/MiniGPT-4/sam_vit_h_4b8939.pth"
device="cuda:0"

sam = sam_model_registry[model_type](checkpoint="/home/t1/vikrant.dewangan/llm-bev/MiniGPT-4/sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    mix = max(mix, 0)
    mix = min(mix, 1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

projection_matrix = np.array(
    [[ 307.74978638, -336.39245605,    0.,         -171.69383118],
     [ 165.84719849,    0.,         -336.39245605, -233.83039856],
     [   1.,            0.,            0.,           -0.7       ]])

print(projection_matrix.shape)

fil_name = "cloud.pcd"
seq_num = "000000"
pc = np.array(o3d.io.read_point_cloud(f"data/cie_to_vc/pcd/{seq_num}/{fil_name}").points)
# pc = pc[np.linalg.norm(pc, axis=1) <= 8]
pc = pc[pc[:, 0] > 0]
pc[:, 0] = pc[:, 0] + 3.0
pc[:, 1] = pc[:, 1]# - 0.3
# pc[:, 2] -= 2.5
# import pdb; pdb.set_trace()
# pc = np.zeros((100, 3))
# pc[:, 0] = np.linspace(3.12, 4.10, pc.shape[0]) + 1.5
# pc[:, 1] = np.linspace(-2, -1.3, pc.shape[0]) - 0.4
# # import pdb; pdb.set_trace()
# pcc = np.zeros((100 * 100, 3))
# pcc[:, :2] = np.array(np.meshgrid(pc[:, 0], pc[:, 1])).T.reshape(pc.shape[0] * pc.shape[0], 2)
# pcc[:, 2] = np.linspace(-2, -2, pcc.shape[0])
# pc = np.concatenate((pcc, np.ones((pcc.shape[0], 1))), axis=1)

pc = np.concatenate((pc, np.ones((pc.shape[0], 1))), axis=1)
if fil_name != "cloud.pcd":
    aa = np.ones_like(pc)
    aa[:, 0] = pc[:, 2]
    aa[:, 1] = pc[:, 0]
    aa[:, 2] = pc[:, 1]
    pc = aa
print(pc.shape)
print((projection_matrix @ pc.T).T.shape)

c1='blue' #blue
c2='yellow' #green

# coloring = np.array([colorFader(c1,c2, (pc[ind, 2] + 5)/((np.max(pc[:, 2]) + 5))) for ind in range(len(pc))])

cam_points = (projection_matrix @ pc.T).T
cam_points = cam_points/cam_points[:, 2:]

cam_points_ = np.copy(cam_points)

img = np.array(Image.open(f"data/cie_to_vc/images/{seq_num}.png"))

# cam_points = cam_points.T
mask = np.ones(cam_points.shape[0], dtype=bool)
# mask = np.logical_and(mask, img > 1.0)
mask = np.logical_and(mask, cam_points[:, 0] > 1)
mask = np.logical_and(mask, cam_points[:, 0] < img.shape[1] - 1)
mask = np.logical_and(mask, cam_points[:, 1] > 1)
mask = np.logical_and(mask, cam_points[:, 1] < img.shape[0] - 1)

cam_points = cam_points[mask]
pc = pc[mask]
# coloring = coloring[mask]

# color = np.array([colorFader(c1,c2, abs(pc[ind, 2])/((np.max(pc[:, 2])))) for ind in range(len(pc))])
plt.scatter((pc[:, 0]).astype(np.int32), pc[:, 1].astype(np.int32), s=0.1)
plt.xlim([-40, 40])
plt.ylim([-40, 40])
# plt.axis("equal")
plt.savefig("test_down.png")
plt.clf()
plt.imshow(img)
plt.scatter(cam_points[:, 0], cam_points[:, 1], s=1)
plt.savefig("test.png")
plt.clf()

# driver
input_point = np.array([[450, 200]])
input_label = np.array([1])
predictor.set_image(img)
masks_1, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label,multimask_output=True)
# scooter
input_point = np.array([[450, 250]])
input_label = np.array([1])
predictor.set_image(img)
masks_2, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label,multimask_output=True)

masks = np.maximum(0, masks_1 + masks_2)
scooter_mask = masks[-1]

plt.imshow(scooter_mask)
plt.savefig("mask.png")
plt.clf()

ys = np.where(scooter_mask > 0)[0]
xs = np.where(scooter_mask > 0)[1]
pts = np.dstack((xs, ys))[0]

ccp = cam_points.astype(np.int32)[:, :2]
# common_points = np.array([point for pointind, point in enumerate(ccp) if point in pts])
common_points = []
common_points_3d = []
ccpind = 0
for point in ccp:
    flag = False
    # check whether this point in pts
    for ind in pts:
        if ind[0] == point[0] and ind[1] == point[1]:
            flag = True
    if flag:
        common_points.append(point)
        common_points_3d.append(pc[ccpind])
    ccpind = ccpind + 1

common_points = np.array(common_points)
common_points_3d = np.array(common_points_3d)
plt.imshow(img)
plt.scatter(common_points[:, 0], common_points[:, 1], s=1)
plt.savefig("test_cp.png")
plt.clf()

plt.clf()
plt.scatter(pc[:, 0], pc[:, 1])
plt.scatter(common_points_3d[:, 0], common_points_3d[:, 1])
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.savefig("test_cm.png")
import pdb; pdb.set_trace()

# exit()
################## TESTING BY BACKPROJECTING
pc = common_points_3d
cam_points = (projection_matrix @ pc.T).T
cam_points = cam_points/cam_points[:, 2:]

cam_points_ = np.copy(cam_points)

img = np.array(Image.open(f"data/cie_to_vc/images/{seq_num}.png"))

# cam_points = cam_points.T
mask = np.ones(cam_points.shape[0], dtype=bool)
# mask = np.logical_and(mask, img > 1.0)
mask = np.logical_and(mask, cam_points[:, 0] > 1)
mask = np.logical_and(mask, cam_points[:, 0] < img.shape[1] - 1)
mask = np.logical_and(mask, cam_points[:, 1] > 1)
mask = np.logical_and(mask, cam_points[:, 1] < img.shape[0] - 1)

cam_points = cam_points[mask]
pc = pc[mask]
# coloring = coloring[mask]
plt.clf()
# color = np.array([colorFader(c1,c2, abs(pc[ind, 2])/((np.max(pc[:, 2])))) for ind in range(len(pc))])
plt.scatter((pc[:, 0]).astype(np.int32), pc[:, 1].astype(np.int32), s=0.1)
plt.xlim([-40, 40])
plt.ylim([-40, 40])
# plt.axis("equal")
plt.savefig("test_down_test.png")
plt.clf()
plt.imshow(img)
plt.scatter(cam_points[:, 0], cam_points[:, 1], s=1)
plt.savefig("test_test.png")
plt.clf()
