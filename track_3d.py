import numpy as np
import cv2
import open3d as o3d
from conversions import from_tracking_image


def get_depth_scale(depth):
    if depth.dtype == float:
        depth_scale = 1
    elif depth.dtype == np.uint16:
        depth_scale = 0.001
    else:
        raise RuntimeError(f"Unknown depth type {depth.dtype}")
    return depth_scale


class TrackedObject:
    def __init__(self, tracking_id, pose, class_id):
        self.tracking_id = tracking_id
        self.pose = pose  # [x, y, z, 1.]
        self.class_id = class_id
        self.tracklet_len = 1

    def update(self, pose):
        k = self.tracklet_len / (self.tracklet_len + 1)
        max_k = 0.8
        k = min(k, max_k)
        self.pose = k * self.pose + (1 - k) * pose
        self.pose[3] = 1
        self.tracklet_len += 1

    def is_visible(self, camera_pose_inv, depth, K, D):
        pose_in_camera = np.matmul(camera_pose_inv, self.pose)
        pose_z = pose_in_camera[2]
        if pose_z <= 0:
            return False

        pose_in_camera = np.expand_dims(pose_in_camera[:3], axis=(0, 1))
        point, _ = cv2.projectPoints(pose_in_camera, np.zeros((3,)), np.zeros((3,)), K, D)
        point = point[0, 0].astype(int)
        height, width = depth.shape
        if not np.all((np.array([0, 0]) <= point) & (point < np.array([width, height]))):
            return False

        square_half_size = 20
        min_x = max(point[0] - square_half_size, 0)
        min_y = max(point[1] - square_half_size, 0)
        max_x = min(point[0] + square_half_size, width)
        max_y = min(point[1] + square_half_size, height)
        depth_scale = get_depth_scale(depth)
        depth_z = depth[min_y:max_y, min_x:max_x].flatten() * depth_scale
        shift = 0.05
        rate = np.count_nonzero(depth_z > pose_z - shift) / depth_z.size

        thresh = 0.5
        if rate > thresh:
            return True
        else:
            return False


class Tracker3D:
    def __init__(self, erosion_size, K, D):
        self.erosion_size = erosion_size
        if self.erosion_size > 0:
            self.erosion_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                (2 * self.erosion_size + 1, 2 * self.erosion_size + 1),
                (self.erosion_size, self.erosion_size))
        self.K = K
        self.D = D

        self.tracked_objects = list()

        assert np.all(self.D == 0), "Distorted images are not supported yet"

    def update(self, camera_pose, depth, classes_ids, tracking_ids, masks):
        objects_poses_in_camera = self.get_objects_poses(depth, masks)
        objects_poses_in_camera = np.hstack((objects_poses_in_camera, np.ones((len(objects_poses_in_camera), 1))))
        objects_poses_in_camera = np.expand_dims(objects_poses_in_camera, axis=-1)
        objects_poses = np.matmul(camera_pose, objects_poses_in_camera)
        objects_poses = objects_poses[:, :, 0]

        for tracked_object in self.tracked_objects:
            ### set status Lost
            pass

        unused_indices = list()
        for i, (object_pose, class_id, tracking_id) in \
                enumerate(zip(objects_poses, classes_ids, tracking_ids)):
            for tracked_object in self.tracked_objects:
                if tracked_object.tracking_id == tracking_id:
                    assert tracked_object.class_id == class_id
                    tracked_object.update(object_pose)
                    ### set status Tracked
                    break
            else:
                unused_indices.append(i)

        new_objects_poses = objects_poses[unused_indices]
        new_classes_ids = classes_ids[unused_indices]
        new_tracking_ids = tracking_ids[unused_indices]
        for new_object_pose, new_class_id, new_tracking_id in \
                zip(new_objects_poses, new_classes_ids, new_tracking_ids):
            new_tracked_object = TrackedObject(new_tracking_id, new_object_pose, new_class_id)
            self.tracked_objects.append(new_tracked_object)

    def get_objects_poses(self, depth, masks):
        depth_scale = get_depth_scale(depth)
        object_poses = list()
        for mask in masks:
            if self.erosion_size > 0:
                mask = cv2.erode(mask, self.erosion_element)
            masked_depth = depth.copy()
            masked_depth[mask == 0] = 0
            if masked_depth.dtype.byteorder != '=':
                masked_depth = masked_depth.astype(masked_depth.dtype.newbyteorder('='))
            masked_depth = o3d.geometry.Image(masked_depth)
            height, width = mask.shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, self.K)
            pc = o3d.geometry.PointCloud.create_from_depth_image(
                masked_depth, intrinsic, depth_scale=(1 / depth_scale))
            object_pose = pc.get_center()
            object_poses.append(object_pose)
        if len(object_poses) > 0:
            object_poses = np.array(object_poses)
        else:
            object_poses = np.empty((0, 3))

        return object_poses

    @staticmethod
    def from_tracking_image(tracking_image):
        classes_ids, tracking_ids, boxes, masks = from_tracking_image(tracking_image)
        return classes_ids, tracking_ids, masks
