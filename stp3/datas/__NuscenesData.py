import os
from PIL import Image

import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision

from shapely.geometry import LineString

from pyquaternion import Quaternion
from shapely import affinity
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.arcline_path_utils import discretize_lane, ArcLinePath
from stp3.utils.tools import ( gen_dx_bx, get_nusc_maps)
import matplotlib.pyplot as plt 
from math import cos, sin

from stp3.utils.geometry import (
    resize_and_crop_image,
    update_intrinsics,
    calculate_birds_eye_view_parameters,
    convert_egopose_to_matrix_numpy,
    pose_vec2mat,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
    get_global_pose,
    get_patch_coord,
    dfs,
    dfs_incoming,
    remove_overlapping_lane_seq,
    extend_both_sides,
    project_to_frenet_frame,
    project_to_cartesian_frame
)
from stp3.utils.instance import convert_instance_mask_to_center_and_offset_label
import stp3.utils.sampler as trajectory_sampler
from stp3.utils.spline import Spline2D
from stp3.utils.Optnode import *

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

resolution_meters = 0.1
t_fin = 3.5
num = 61
nvar = 11
bbx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
ddx = np.array([0.5, 0.5])
debug = True
brdebug = False
problem = OPTNode(t_fin=t_fin, num=num)

class FuturePredictionDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5 #SECOND
    def __init__(self, nusc, is_train, cfg):
        self.nusc = nusc
        self.dataroot = self.nusc.dataroot
        self.nusc_exp = NuScenesExplorer(nusc)
        self.nusc_can = NuScenesCanBus(dataroot=self.dataroot)
        self.is_train = is_train
        self.cfg = cfg
        self.problem = OPTNode(t_fin=3.5, num=self.cfg.N_FUTURE_FRAMES + 1)

        if self.is_train == 0:
            self.mode = 'train'
        elif self.is_train == 1:
            self.mode = 'val'
        elif self.is_train == 2:
            self.mode = 'test'
        else:
            raise NotImplementedError

        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        self.indices = self.get_indices()

        # Image resizing and cropping
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

        # Normalising input images
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # The number of sampled trajectories
        self.n_samples = self.cfg.PLANNING.SAMPLE_NUM

        # HD-map feature extractor
        self.nusc_maps = get_nusc_maps(self.cfg.DATASET.MAP_FOLDER)
        self.scene2map = {}
        for sce in self.nusc.scene:
            log = self.nusc.get('log', sce['log_token'])
            self.scene2map[sce['name']] = log['location']
        self.save_dir = cfg.DATASET.SAVE_DIR

    def get_scenes(self):
        # filter by scene split
        split = {'v1.0-trainval': {0: 'train', 1: 'val', 2: 'test'},
                 'v1.0-mini': {0: 'mini_train', 1: 'mini_val'},}[
            self.nusc.version
        ][self.is_train]

        blacklist = [419] + self.nusc_can.can_blacklist  # # scene-0419 does not have vehicle monitor data
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

        scenes = create_splits_scenes()[split][:]
        for scene_no in blacklist:
            if scene_no in scenes:
                scenes.remove(scene_no)

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        resize_scale = self.cfg.IMAGE.RESIZE_SCALE
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = self.cfg.IMAGE.TOP_CROP
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def get_input_data(self, rec):
        """
        Parameters
        ----------
            rec: nuscenes identifier for a given timestamp

        Returns
        -------
            images: torch.Tensor<float> (N, 3, H, W)
            intrinsics: torch.Tensor<float> (3, 3)
            extrinsics: torch.Tensor(N, 4, 4)
        """
        images = []
        intrinsics = []
        extrinsics = []
        depths = []
        cameras = self.cfg.IMAGE.NAMES

        # The extrinsics we want are from the camera sensor to "flat egopose" as defined
        # https://github.com/nutonomy/nuscenes-devkit/blob/9b492f76df22943daf1dc991358d3d606314af27/python-sdk/nuscenes/nuscenes.py#L279
        # which corresponds to the position of the lidar.
        # This is because the labels are generated by projecting the 3D bounding box in this lidar's reference frame.

        # From lidar egopose to world.
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])

        for cam in cameras:
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])

            # Transformation from world to egopose
            car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
            egopose_rotation = Quaternion(car_egopose['rotation']).inverse
            egopose_translation = -np.array(car_egopose['translation'])[:, None]
            world_to_car_egopose = np.vstack([
                np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                np.array([0, 0, 0, 1])
            ])

            # From egopose to sensor
            sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
            intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(sensor_sample['translation'])[:, None]
            car_egopose_to_sensor = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1])
            ])
            car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

            # Combine all the transformation.
            # From sensor to lidar.
            lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
            sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()

            # Load image
            image_filename = os.path.join(self.dataroot, camera_sample['filename'])
            img = Image.open(image_filename)
            # Resize and crop
            img = resize_and_crop_image(
                img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
            )
            # Normalise image
            normalised_img = self.normalise_image(img)

            # Combine resize/cropping in the intrinsics
            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]
            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )

            # Get Depth
            # Depth data should under the dataroot path 
            if self.cfg.LIFT.GT_DEPTH:
                base_root = os.path.join(self.dataroot, 'depths') 
                filename = os.path.basename(camera_sample['filename']).split('.')[0] + '.npy'
                depth_file_name = os.path.join(base_root, cam, 'npy', filename)
                depth = torch.from_numpy(np.load(depth_file_name)).unsqueeze(0).unsqueeze(0)
                depth = F.interpolate(depth, scale_factor=self.cfg.IMAGE.RESIZE_SCALE, mode='bilinear')
                depth = depth.squeeze()
                crop = self.augmentation_parameters['crop']
                depth = depth[crop[1]:crop[3], crop[0]:crop[2]]
                depth = torch.round(depth)
                depths.append(depth.unsqueeze(0).unsqueeze(0))

            images.append(normalised_img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))

        images, intrinsics, extrinsics = (torch.cat(images, dim=1),
                                          torch.cat(intrinsics, dim=1),
                                          torch.cat(extrinsics, dim=1)
                                          )
        if len(depths) > 0:
            depths = torch.cat(depths, dim=1)

        return images, intrinsics, extrinsics, depths

    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def get_depth_from_lidar(self, lidar_sample, cam_sample):
        points, coloring, im = self.nusc_exp.map_pointcloud_to_image(lidar_sample, cam_sample)
        tmp_cam = np.zeros((self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH))
        points = points.astype(np.int)
        tmp_cam[points[1, :], points[0,:]] = coloring
        tmp_cam = torch.from_numpy(tmp_cam).unsqueeze(0).unsqueeze(0)
        tmp_cam = F.interpolate(tmp_cam, scale_factor=self.cfg.IMAGE.RESIZE_SCALE, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        tmp_cam = tmp_cam.squeeze()
        crop = self.augmentation_parameters['crop']
        tmp_cam = tmp_cam[crop[1]:crop[3], crop[0]:crop[2]]
        tmp_cam = torch.round(tmp_cam)
        return tmp_cam


    def get_birds_eye_view_label(self, rec, instance_map, in_pred):
        translation, rotation = self._get_top_lidar_pose(rec)
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

        for annotation_token in rec['anns']:
            # Filter out all non vehicle instances
            annotation = self.nusc.get('sample_annotation', annotation_token)

            if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and in_pred is False:
                continue
            if in_pred is True and annotation['instance_token'] not in instance_map:
                continue

            # NuScenes filter
            if 'vehicle' in annotation['category_name']:
                if annotation['instance_token'] not in instance_map:
                    instance_map[annotation['instance_token']] = len(instance_map) + 1
                instance_id = instance_map[annotation['instance_token']]
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(instance, [poly_region], instance_id)
                cv2.fillPoly(segmentation, [poly_region], 1.0)
            elif 'human' in annotation['category_name']:
                if annotation['instance_token'] not in instance_map:
                    instance_map[annotation['instance_token']] = len(instance_map) + 1
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(pedestrian, [poly_region], 1.0)


        return segmentation, instance, pedestrian, instance_map

    def _get_poly_region_in_image(self, instance_annotation, ego_translation, ego_rotation):
        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(ego_translation)
        box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]
        return pts, z

    def get_label(self, rec, instance_map, in_pred):
        segmentation_np, instance_np, pedestrian_np, instance_map = \
            self.get_birds_eye_view_label(rec, instance_map, in_pred)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
        instance = torch.from_numpy(instance_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0).unsqueeze(0)

        return segmentation, instance, pedestrian, instance_map

    def get_future_egomotion(self, rec, index):
        rec_t0 = rec

        # Identity
        future_egomotion = np.eye(4, dtype=np.float32)

        if index < len(self.ixes) - 1:
            rec_t1 = self.ixes[index + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                egopose_t0 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
                )
                egopose_t1 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
                )

                egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
                egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

                future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
                future_egomotion[3, :3] = 0.0
                future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        return future_egomotion.unsqueeze(0)

    def get_trajectory_sampling(self, rec=None, sample_indice=None):
        if rec is None and sample_indice is None:
            raise ValueError("No valid input rec or token")
        if rec is None and sample_indice is not None:
            rec = self.ixes[sample_indice]

        ref_scene = self.nusc.get("scene", rec['scene_token'])

        # vm_msgs = self.nusc_can.get_messages(ref_scene['name'], 'vehicle_monitor')
        # vm_uts = [msg['utime'] for msg in vm_msgs]
        pose_msgs = self.nusc_can.get_messages(ref_scene['name'],'pose')
        pose_uts = [msg['utime'] for msg in pose_msgs]
        steer_msgs = self.nusc_can.get_messages(ref_scene['name'], 'steeranglefeedback')
        steer_uts = [msg['utime'] for msg in steer_msgs]

        ref_utime = rec['timestamp']
        # vm_index = locate_message(vm_uts, ref_utime)
        # vm_data = vm_msgs[vm_index]
        pose_index = locate_message(pose_uts, ref_utime)
        pose_data = pose_msgs[pose_index]
        steer_index = locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]

        # initial speed
        # v0 = vm_data["vehicle_speed"] / 3.6  # km/h to m/s
        v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s

        # curvature (positive: turn left)
        # steering = np.deg2rad(vm_data["steering"])
        steering = steer_data["value"]

        location = self.scene2map[ref_scene['name']]
        # flip x axis if in left-hand traffic (singapore)
        flip_flag = True if location.startswith('singapore') else False
        if flip_flag:
            steering *= -1
        Kappa = 2 * steering / 2.588

        # initial state
        T0 = np.array([0.0, 1.0])  # define front
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])  # define side

        t_start = 0  # second
        t_end = self.cfg.N_FUTURE_FRAMES * self.SAMPLE_INTERVAL  # second
        t_interval = self.SAMPLE_INTERVAL / 10
        tt = np.arange(t_start, t_end + t_interval, t_interval)
        sampled_trajectories_fine = trajectory_sampler.sample(v0, Kappa, T0, N0, tt, self.n_samples)
        sampled_trajectories = sampled_trajectories_fine[:, ::10]
        return sampled_trajectories

    def generate_centerlines(self, rec, map_name):
        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND)
        stretch = [self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1]]
        dx, bx = dx[:2].numpy(), bx[:2].numpy()

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1,0], rot[0,0]) # in radian
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        box_coords = (
            center[0],
            center[1],
            stretch[0]*2,
            stretch[1]*2
        ) # (x_center, y_center, width, height)
        canvas_size = (
                int(self.cfg.LIFT.X_BOUND[1] * 2 / self.cfg.LIFT.X_BOUND[2]),
                int(self.cfg.LIFT.Y_BOUND[1] * 2 / self.cfg.LIFT.Y_BOUND[2])
        )

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        x, y = egopose["translation"][0], egopose["translation"][1]
        nusc_map = self.nusc_maps[map_name]
        closest_lane = nusc_map.get_closest_lane(x, y, radius=2)
        candidates_future = dfs(nusc_map, closest_lane, dist=0, threshold=self.cfg.DATASET.THRESHOLD, resolution_meters=resolution_meters)
        candidates_past = dfs_incoming(nusc_map, closest_lane, dist=0, threshold=self.cfg.DATASET.THRESHOLD, resolution_meters=resolution_meters)
        plt.savefig("output_vis/debug_dfs.png")
        # lane_ids = dfs(nusc_map, closest_lane, dist=0, threshold=self.cfg.DATASET.THRESHOLD)
        lane_ids = []
        for past_lane_seq in candidates_past:
            for future_lane_seq in candidates_future:
                lane_ids.append(past_lane_seq + future_lane_seq[1:])
        lane_ids = remove_overlapping_lane_seq(lane_ids)
        centerlines = []
        for lane_id in lane_ids:
            centerline = []
            for lane in lane_id:
                lane_record = nusc_map.get_arcline_path(lane)
                cx = np.array(discretize_lane(lane_record, resolution_meters=resolution_meters))
                centerline.extend(cx)
            centerlines.append(np.array(centerline))
        lanes = self.nusc_maps[map_name].get_records_in_radius(x, y, 2000, ['lane', 'lane_connector'])
        lanes = lanes['lane'] + lanes['lane_connector']
        discrete_points = self.nusc_maps[map_name].discretize_lanes(lanes, resolution_meters=resolution_meters)
        lanes = []
        for lane_id, points in discrete_points.items():
            lanes.append(np.array(points)[:, :2])
        
        patch_box = box_coords
        patch_angle = rot * 180 / np.pi

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = get_patch_coord(patch_box, patch_angle)

        line_list = []
        for lane in centerlines:
            line = LineString([(arr[0], arr[1]) for arr in lane[:, :2]])
            c = lane[:, :2]
            if not line.intersection(patch).is_empty:
                new_line = line
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                        [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)
        
        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        patch_x, patch_y, patch_h, patch_w = local_box

        patch = get_patch_coord(local_box)

        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]
        scale_height = canvas_h/patch_h
        scale_width = canvas_w/patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        overall_lines = []
        for line in line_list:
            if not line.intersection(patch).is_empty:
                new_line = line
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
                overall_lines.append([list(new_line.coords)])

        return overall_lines

    def voxelize_hd_map(self, rec):
        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND)
        stretch = [self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1]]
        dx, bx = dx[:2].numpy(), bx[:2].numpy()

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1,0], rot[0,0]) # in radian
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        box_coords = (
            center[0],
            center[1],
            stretch[0]*2,
            stretch[1]*2
        ) # (x_center, y_center, width, height)
        canvas_size = (
                int(self.cfg.LIFT.X_BOUND[1] * 2 / self.cfg.LIFT.X_BOUND[2]),
                int(self.cfg.LIFT.Y_BOUND[1] * 2 / self.cfg.LIFT.Y_BOUND[2])
        )

        elements = self.cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
        hd_features = self.nusc_maps[map_name].get_map_mask(box_coords, rot * 180 / np.pi , elements, canvas_size=canvas_size)
        #traffic = self.hd_traffic_light(map_name, center, stretch, dx, bx, canvas_size)
        #return torch.from_numpy(np.concatenate((hd_features, traffic), axis=0)[None]).float()
        hd_features = torch.from_numpy(hd_features[None]).float()
        hd_features = torch.transpose(hd_features,-2,-1) # (y,x) replace horizontal and vertical coordinates
        return hd_features

    def hd_traffic_light(self, map_name, center, stretch, dx, bx, canvas_size):

        roads = np.zeros(canvas_size)
        my_patch = (
            center[0] - stretch[0],
            center[1] - stretch[1],
            center[0] + stretch[0],
            center[1] + stretch[1],
        )
        tl_token = self.nusc_maps[map_name].get_records_in_patch(my_patch, ['traffic_light'], mode='intersect')['traffic_light']
        polys = []
        for token in tl_token:
            road_token =self.nusc_maps[map_name].get('traffic_light', token)['from_road_block_token']
            pt = self.nusc_maps[map_name].get('road_block', road_token)['polygon_token']
            polygon = self.nusc_maps[map_name].extract_polygon(pt)
            polys.append(np.array(polygon.exterior.xy).T)

        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])
        # convert to local coordinates in place
        rot = get_rot(np.arctan2(center[3], center[2])).T
        for rowi in range(len(polys)):
            polys[rowi] -= center[:2]
            polys[rowi] = np.dot(polys[rowi], rot)

        for la in polys:
            pts = (la - bx) / dx
            pts = np.int32(np.around(pts))
            cv2.fillPoly(roads, [pts], 1)

        return roads[None]

    def get_gt_trajectory(self, rec, ref_index):
        n_output = self.cfg.N_FUTURE_FRAMES
        gt_trajectory = np.zeros((n_output+1+2, 3), np.float64)

        egopose_cur = get_global_pose(rec, self.nusc, inverse=True)

        for i in range(2):
            index = ref_index - 2 + i
            if index < len(self.ixes):
                rec_future = self.ixes[index]

                egopose_future = get_global_pose(rec_future, self.nusc, inverse=False)

                egopose_future = egopose_cur.dot(egopose_future)
                theta = quaternion_yaw(Quaternion(matrix=egopose_future))

                origin = np.array(egopose_future[:3, 3])

                gt_trajectory[i, :] = [origin[0], origin[1], theta]

        for i in range(n_output+1):
            index = ref_index + i
            if index < len(self.ixes):
                rec_future = self.ixes[index]

                egopose_future = get_global_pose(rec_future, self.nusc, inverse=False)

                egopose_future = egopose_cur.dot(egopose_future)
                theta = quaternion_yaw(Quaternion(matrix=egopose_future))

                origin = np.array(egopose_future[:3, 3])

                gt_trajectory[i + 2, :] = [origin[0], origin[1], theta]

        if gt_trajectory[-1][0] >= 2:
            command = 'RIGHT'
        elif gt_trajectory[-1][0] <= -2:
            command = 'LEFT'
        else:
            command = 'FORWARD'

        return gt_trajectory, command

    def get_routed_map(self, gt_points):
        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND)
        dx, bx = dx[:2].numpy(), bx[:2].numpy()

        canvas_size = (
            int(self.cfg.LIFT.X_BOUND[1] * 2 / self.cfg.LIFT.X_BOUND[2]),
            int(self.cfg.LIFT.Y_BOUND[1] * 2 / self.cfg.LIFT.Y_BOUND[2])
        )

        roads = np.zeros(canvas_size)
        W = 1.85
        pts = np.array([
            [-4.084 / 2. + 0.5, W / 2.],
            [4.084 / 2. + 0.5, W / 2.],
            [4.084 / 2. + 0.5, -W / 2.],
            [-4.084 / 2. + 0.5, -W / 2.],
        ])
        pts = (pts - bx) / dx
        pts[:, [0, 1]] = pts[:, [1, 0]]

        pts = np.int32(np.around(pts))
        cv2.fillPoly(roads, [pts], 1)

        gt_points = gt_points[:-1].numpy()
        # 坐标原点在左上角
        target = pts.copy()
        target[:,0] = pts[:,0] + gt_points[0] / dx[0]
        target[:,1] = pts[:,1] - gt_points[1] / dx[1]
        target = np.int32(np.around(target))
        cv2.fillPoly(roads, [target], 1)
        return roads

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns
        -------
            data: dict with the following keys:
                image: torch.Tensor<float> (T, N, 3, H, W)
                    normalised cameras images with T the sequence length, and N the number of cameras.
                intrinsics: torch.Tensor<float> (T, N, 3, 3)
                    intrinsics containing resizing and cropping parameters.
                extrinsics: torch.Tensor<float> (T, N, 4, 4)
                    6 DoF pose from world coordinates to camera coordinates.
                segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                    (H_bev, W_bev) are the pixel dimensions in bird's-eye view.
                instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
                offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
                flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
                future_egomotion: torch.Tensor<float> (T, 6)
                    6 DoF egomotion t -> t+1

        """
        data = {}
        keys = ['image', 'intrinsics', 'extrinsics', 'depths',
                'segmentation', 'instance', 'centerness', 'offset', 'flow', 'pedestrian',
                'future_egomotion', 'hdmap', 'gt_trajectory', 'indices', 'poses'
                ]
        for key in keys:
            data[key] = []

        instance_map = {}
        # Loop over all the frames in the sequence.
        total_centerlines = []
        for i, index_t in enumerate(self.indices[index]):
            if i >= self.receptive_field:
                in_pred = True
            else:
                in_pred = False
            rec = self.ixes[index_t]
            map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

            if i < self.receptive_field:
                images, intrinsics, extrinsics, depths = self.get_input_data(rec)
                data['image'].append(images)
                data['intrinsics'].append(intrinsics)
                data['extrinsics'].append(extrinsics)
                data['depths'].append(depths)
            segmentation, instance, pedestrian, instance_map = self.get_label(rec, instance_map, in_pred)

            future_egomotion = self.get_future_egomotion(rec, index_t)
            hd_map_feature = self.voxelize_hd_map(rec)

            data['segmentation'].append(segmentation)
            data['instance'].append(instance)
            data['pedestrian'].append(pedestrian)
            data['future_egomotion'].append(future_egomotion)
            data['hdmap'].append(hd_map_feature)
            data['indices'].append(index_t)

            """
                get lanes in 100m radius
            """
            centerlines = self.generate_centerlines(rec, map_name)
            total_centerlines.append(centerlines)

            if i == self.cfg.TIME_RECEPTIVE_FIELD-1:
                gt_trajectory, command = self.get_gt_trajectory(rec, index_t)
                gt_trajectory_prev = gt_trajectory
                traj=torch.tensor(gt_trajectory_prev)[:, :2].reshape(1,9,2)
                gt_trajectory = gt_trajectory[2:]
                data['gt_trajectory'] = torch.from_numpy(gt_trajectory).float()
                data['command'] = command
                trajs = self.get_trajectory_sampling(rec)
                data['sample_trajectory'] = torch.from_numpy(trajs).float()


                """
                    DISPLAY DEBUG
                """
                display = np.ones((200, 200, 3))
                whe = np.where(hd_map_feature[0,1] >= 1)
                display[whe[0], whe[1]] = [0, 0, 0]
                whe = np.where(segmentation[0,0] >= 1)
                display[whe[0], whe[1]] = [0, 0, 1]

                """
                    CREATE GRID
                    grid: (M, 2) -> sample grid
                """
                sample_trajectories = []
                if debug and brdebug:breakpoint()
                if debug:
                    plt.figure(figsize=(10, 10))

                """
                    ITERATE OVER CENTERLINES
                """
                total_trajs = []
                total_eps = []
                data_centerlines = []
                for centerline in total_centerlines[-1]:
                    """
                        INITIALIZE GRID
                    """
                    x = np.linspace(0, 19, 20)
                    y = np.linspace(-3, 3, 7)
                    xv, yv = np.meshgrid(x, y)
                    grid = np.dstack((xv, yv))
                    grid = np.concatenate((grid), axis=0)

                    """
                        CONVERT CENTERLINE TO BOX FRAME
                    """
                    cx = np.array(centerline)[0]
                    cx = extend_both_sides(cx, resolution=resolution_meters, extension=30)
                    cx_o = np.zeros_like(cx) + cx
                    cx = cx * ddx
                    cx += bbx
                    cx[:, [1, 0]] = cx[:, [0, 1]]
                    cx[:, :1] = cx[:, :1] * -1
                    if debug:
                        gtt = gt_trajectory_prev[:, :2]
                        plt.plot(gtt[:, 0], gtt[:, 1], color="orange")
                        plt.plot(cx[:, 0], cx[:, 1], color="green")
                        plt.savefig(f"output_vis/{index}debug_cartesian.png")
                        plt.clf()
                    #################################################################################


                    """
                        CONVERT CENTERLINE TO SPLINE
                            spline: represents the spline
                            gt_trajectory_prev: (1, 9, 2) representing the GT
                            ssd_list: frenet frame of gt_trajectory_prev
                            s_offset: s coordinate at T=2
                            d_offset: d coordinate at T=2
                    """
                    trj = torch.tensor(gt_trajectory_prev)[:, :2].reshape(1, 9, 2)
                    oracle_centerline = torch.tensor(cx).reshape(1, cx.shape[0], 2)
                    spline = Spline2D(x=oracle_centerline[0,:, 0].numpy(), y=oracle_centerline[0,:, 1].numpy(), resolution=resolution_meters, extension=15)
                    ssd_list = []
                    for xy in trj[0]:
                        s,d = spline.calc_frenet_position(xy[0].numpy(), xy[1].numpy())
                        ssd_list.append([s, d])
                    ssd = torch.tensor(ssd_list).reshape(1, trj.shape[1], 2)
                    s_offset = ssd[0,2,0].detach()
                    d_offset = ssd[0,2,1].detach()
                    vel = ssd[0,2] - ssd[0,1]
                    theta = np.arctan2(vel[1], vel[0]) # to make angle 0 wrt ego-agent
                    vel_prev = ssd[0,1] - ssd[0, 0]
                    theta_prev = np.arctan2(vel_prev[1], vel_prev[0])
                    psidot_init = (theta - theta_prev)/0.5
                    psi_init = theta
                    v_init = np.linalg.norm(vel)/0.5
                    y_init = 0
                    x_init = 0
                    rot_inv = np.array([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]])
                    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
                    if debug and brdebug:breakpoint()                    
                    #################################################################################

                    """
                        CONVERT CENTERLINE ITSELF TO CENTERLINE FRAME
                    """
                    sdd_list = []
                    trj = oracle_centerline[:, 10:-10]
                    for xy in trj[0]:
                        s,d = spline.calc_frenet_position(xy[0].numpy(), xy[1].numpy())
                        sdd_list.append([s, d])
                    sdd = torch.tensor(sdd_list).reshape(1, trj.shape[1], 2)
                    if debug:
                        plt.plot(sdd[0, :, 0].cpu().numpy(), sdd[0, :, 1].cpu().numpy(), color="green")
                        plt.scatter(grid[:, 0], grid[ :, 1], color="red", s=1)
                        plt.plot(ssd[0, :, 0], ssd[0, :, 1], color="blue")
                        plt.axis('equal')
                        plt.savefig(f"output_vis/{index}debug_sdd_vefore.png")
                        plt.clf()

                    if debug and brdebug:breakpoint()
                    ssd_frame = ssd -  torch.tensor([s_offset, d_offset])
                    sdd_frame = sdd -  torch.tensor([s_offset, d_offset])
                    eps_frame = grid - np.array([0, d_offset])
                    if debug:
                        plt.plot(sdd_frame[0, :, 0].cpu().numpy(), sdd_frame[0, :, 1].cpu().numpy(), color="green")
                        plt.scatter(eps_frame[:, 0], eps_frame[ :, 1], color="red", s=1)
                        plt.plot(ssd_frame[0, :, 0], ssd_frame[0, :, 1], color="blue")
                        plt.axis('equal')
                        plt.savefig(f"output_vis/{index}debug_sdd.png")
                        plt.clf()
                        plt.plot(ssd_frame[:, 0], ssd_frame[:, 1], color="blue");plt.savefig("output_vis/debug_aa.png");plt.clf()
                    #################################################################################

                    """
                        CONVERT GRID TO AGENT FRAME
                    """
                    grid = grid - np.array([0, d_offset])
                    if debug and brdebug:breakpoint()
                    #################################################################################

                    """
                        ROTATE EVERYTHING TO AGENT FRAME
                    """
                    sdd_frame[0] = torch.tensor(np.dot(rot_inv, sdd_frame[0].numpy().T).T)
                    ssd_frame = np.dot(rot_inv, ssd_frame[0].T).T
                    eps_frame = np.dot(rot_inv, grid.T).T

                    if debug:
                        plt.plot(ssd_frame[:, 0], ssd_frame[:, 1], color="blue")
                        plt.plot(sdd_frame[0, :, 0].cpu().numpy(), sdd_frame[0, :, 1].cpu().numpy(), color="green")
                        plt.scatter(eps_frame[:, 0], eps_frame[ :, 1], color="red", s=1)
                        plt.axis('equal')
                        plt.savefig(f"output_vis/{index}debug_sdd_rotated.png")
                        plt.clf()
                    #################################################################################

                    """
                        SOLVE THE PROBLEM
                    """                        
                    fixed_params = torch.zeros((1, 5)).expand(eps_frame.shape[0], 5)
                    fixed_params[0] = torch.tensor([x_init, y_init, v_init, psi_init, psidot_init])
                    variable_params = torch.zeros((eps_frame.shape[0], 3))
                    variable_params[:, 0] = torch.tensor(eps_frame[:, 0])
                    variable_params[:, 1] = torch.tensor(eps_frame[:, 1])
                    variable_params[:, 2] = torch.tensor(-theta)
                    P=problem.P
                    cc, _ = problem.solve(fixed_params.double(), variable_params.double())
                    x_sol = np.dot(P, cc.T[:nvar]).T
                    y_sol = np.dot(P, cc.T[nvar:2*nvar]).T
                    min_val = 1e11
                    min_ind = 0
                    #################################################################################

                    """
                        COMPUTE MIN_TRAJ
                    """
                    for i in range(len(x_sol)):
                        traj = np.dstack((x_sol[i][::10], y_sol[i][::10]))[0]
                        dist = np.linalg.norm(traj - ssd_frame[2:])
                        if dist < min_val:
                            min_val = dist
                            min_ind = i
                    if debug:
                        for i in range(len(x_sol)):
                            plt.plot(x_sol[i], y_sol[i],linewidth=0.5, color="blue", alpha=0.5)
                            plt.scatter(x_sol[i][::10], y_sol[i][::10],s=1, color="blue", alpha=0.5)
                        plt.plot(ssd_frame[:, 0], ssd_frame[:, 1], color="red", zorder=100, label="GT")
                        plt.scatter(ssd_frame[:, 0], ssd_frame[:, 1], color="red", zorder=100, s=1)
                        plt.scatter(eps_frame[:, 0], eps_frame[:, 1], color="yellow", zorder=100, s=1)
                        plt.plot(sdd_frame[0, 180:-50, 0].cpu().numpy(), sdd_frame[0, 180:-50, 1].cpu().numpy(), color="green")
                        plt.plot(x_sol[min_ind][::10], y_sol[min_ind][::10], color="orange", zorder=100, label="predicted")
                        plt.scatter(x_sol[min_ind][::10], y_sol[min_ind][::10], color="orange", zorder=100, s=3)
                        plt.legend()
                        plt.axis('equal')
                        plt.savefig(f"output_vis/{index}debug.png")
                        plt.clf()
                    #################################################################################

                    """
                       PROJECT TO CARTESIAN FRAME 
                    """
                    trajs = np.dstack((x_sol[:, ::10], y_sol[:,::10])) # N, 7, 2
                    for i in range(len(trajs)):
                        trajs[i] = np.dot(rot, trajs[i].T).T
                    trajs = trajs + np.array([s_offset, d_offset])
                    sdd_frame[0] = torch.tensor(np.dot(rot, sdd_frame[0].numpy().T).T)
                    ssd_back = np.dot(rot, ssd_frame.T).T
                    eps_back = np.dot(rot, eps_frame.T).T
                    ssd_back = ssd_back + np.array([s_offset, d_offset])
                    eps_back = eps_back + np.array([s_offset, d_offset])
                    sdd_back = sdd_frame[0].numpy() + np.array([s_offset, d_offset])
                    if debug:
                        done = False
                        plt.plot(sdd_back[:, 0], sdd_back[:, 1], color="green")
                        plt.plot(ssd_back[:, 0], ssd_back[:, 1], color="red")
                        plt.scatter(ssd_back[:, 0], ssd_back[:, 1], color="red", s=3)
                        plt.scatter(eps_back[:, 0], eps_back[:, 1], color="yellow", s=2)
                        for j in range(len(trajs)):
                            if not done:
                                done = True
                                plt.plot(trajs[min_ind, :7, 0], trajs[min_ind, :7, 1], color="orange", zorder=2000)
                                plt.scatter(trajs[min_ind, :7, 0], trajs[min_ind, :7, 1], color="orange", s=2,zorder=2000)
                            plt.plot(trajs[j,:7, 0], trajs[j,:7, 1], color="blue", alpha=0.5)
                    #################################################################################

                    """
                        ADD TO OVERALL TRAJECTORY SET
                    """
                    for j in range(len(trajs)):
                        xys = []
                        for sds in trajs[j]:
                            x, y = spline.calc_global_position_offline(sds[0], sds[1])
                            xys.append([x, y])
                        trajs[j] = torch.tensor(xys)
                        total_trajs.append(trajs[j, :7])
                    cxx = oracle_centerline[0, :, 0].detach()
                    cyy = oracle_centerline[0, :, 1].detach()
                    data_centerlines.append(np.dstack((cxx, cyy)))
                    if debug:
                        plt.axis('equal')
                        plt.savefig(f"output_vis/{index}__debug.png")
                        plt.clf()
                        orc = []
                        for sds in sdd_back:
                            x, y = spline.calc_global_position_offline(sds[0], sds[1])
                            orc.append([x, y])
                        gd = []
                        for sds in eps_back:
                            x, y = spline.calc_global_position_offline(sds[0], sds[1])
                            gd.append([x, y])
                            total_eps.append([x, y])
                        gd = np.array(gd)
                        orc = np.array(orc)
                        for i in range(len(trajs)):
                            pass
                            plt.plot(trajs[i, :7, 0], trajs[i, :7, 1], color="blue", alpha=0.5)
                        plt.plot(trajs[min_ind, :7, 0], trajs[min_ind, :7, 1], color="orange", zorder=2000)
                        plt.scatter(trajs[min_ind, :7, 0], trajs[min_ind, :7, 1], color="orange", s=3,zorder=2000)
                        plt.plot(gt_trajectory_prev[ :, 0], gt_trajectory_prev[ : ,1], zorder=2000)
                        plt.plot(oracle_centerline[0, :, 0], oracle_centerline[0, : ,1])
                        for traj in data['sample_trajectory']:
                            pass
                            # plt.plot(traj[:, 0], traj[:, 1], "green", alpha=0.05)
                        plt.scatter(gd[:, 0], gd[:, 1], color="yellow", s=3, zorder=2000)
                        plt.scatter(orc[:, 0], orc[:, 1], color="red", s=3, zorder=2000)
                        plt.axis('equal');
                        plt.savefig(f"output_vis/{index}_debug.png")
                        plt.clf()

                trajs = torch.tensor(total_trajs).float()
                zero = torch.zeros((trajs.shape[0], trajs.shape[1], 1))
                trajs = torch.cat((trajs, zero), dim=2)
                if self.cfg.DENSE:
                    data['sample_trajectory'] = trajs
                gt_trajectory[:, :1] = gt_trajectory[:, :1] * -1
                gt_trajectory = (gt_trajectory[:, :2] - bbx) / ddx
                data["centerlines"] = data_centerlines


        for key, value in data.items():
            if key in ['image', 'intrinsics', 'extrinsics', 'depths', 'segmentation', 'instance', 'future_egomotion', 'hdmap', 'pedestrian']:
                if key == 'depths' and self.cfg.LIFT.GT_DEPTH is False:
                    continue
                data[key] = torch.cat(value, dim=0)

        data['target_point'] = torch.tensor([0., 0.])
        instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label(
            data['instance'], data['future_egomotion'],
            num_instances=len(instance_map), ignore_index=self.cfg.DATASET.IGNORE_INDEX, subtract_egomotion=True,
            spatial_extent=self.spatial_extent,
        )
        data['centerness'] = instance_centerness
        data['offset'] = instance_offset
        data['flow'] = instance_flow

        return data