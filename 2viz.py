import numpy as np
from matplotlib import pyplot as plt
from nuscenes.utils.data_classes import PointCloud, LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

cam_ind = 1

arr = np.load("arr.npy")
unn = np.load("unn.npy")[-1][cam_ind]
rotations = np.load("rotations.npy")[-1][cam_ind]
translations = np.load("translations.npy")[-1][cam_ind]
camera_intrinsics = np.load("camera_intrinsics.npy")[-1][cam_ind]
point_clouds = np.load("point_clouds.npy")[cam_ind].T

import pdb; pdb.set_trace()

def distance(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 ) ** ( 1 / 2)
 
# Function to calculate K closest points
def kClosest(points, target, K):
    pts = []
    n = len(points)
    d = []

    for i in range(n):
        d.append({
            "first": distance(points[i][0], points[i][1], points[i][2], target[0], target[1], target[2]),
            "second": i
        })
     
    d = sorted(d, key=lambda l:l["first"])
 
    for i in range(K):
        pt = []
        pt.append(points[d[i]["second"]][0])
        pt.append(points[d[i]["second"]][1])
        pt.append(points[d[i]["second"]][2])
        pt.append(points[d[i]["second"]][3])
        pts.append(pt)
 
    return pts

def mouse_event(event):
    print('x: {} and y: {}'.format(event.xdata, event.ydata))
    x = event.xdata - 100
    y = event.ydata - 100
    x = x/5
    y = y/5
    target = [x, y, 0.0]
    arr = kClosest(point_clouds, target, 5)

    pc = LidarPointCloud(np.array(arr).T)
    pc.translate(translations)
    pc.rotate(rotations)
    depths_img = pc.points[2, :]

    min_dist = 1.0

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    ok = pc.points
    points = view_points(pc.points[:3, :], np.array(camera_intrinsics), normalize=True)
    mask = np.ones(depths_img.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths_img > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < 900 - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < 1600 - 1)

    points = points[:, mask]
    depths_img = depths_img[mask]

    fig2.clf()
    ax = fig2.add_subplot(111)
    depth_gt = np.zeros((900, 1600))
    pts_int = np.array(points, dtype=int)
    depth_gt[pts_int[0,:], pts_int[1,:]] = depths_img

    # import pdb; pdb.set_trace()

    ax.imshow(unn)
    ax.imshow(depth_gt)
    plt.show()

    print(arr)
    print(mask.sum()) 


fig = plt.figure()
ax = fig.add_subplot(111)
im1 = ax.imshow(arr/256)
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

# x = np.linspace(-10, 10, 100)
# y = np.sin(x)

# plt.plot(x, y)

# plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
im2 = ax2.imshow(unn)
plt.show()