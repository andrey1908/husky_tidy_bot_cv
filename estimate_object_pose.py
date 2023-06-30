import numpy as np
import cv2
import open3d as o3d
from segment_by_color import refine_mask_by_polygons, get_sv


class ObjectPoseEstimation:
    def __init__(self, voxel_size, depth_scale, K, D,
            global_max_correspondence_distance, max_correspondence_distances):
        self.voxel_size = voxel_size
        self.depth_scale = depth_scale
        self.K = K
        self.D = D
        self.global_max_correspondence_distance = global_max_correspondence_distance
        self.max_correspondence_distances = max_correspondence_distances

        assert np.all(self.D == 0), "Distorted images are not supported yet"

    def estimate_pose(self, mask, depth):
        self.extracted_pc = self.extract_pc(mask, depth)
        if len(self.extracted_pc.points) < 1000:
            self.reason = \
                f"Too few points in extracted point cloud " \
                f"({len(self.extracted_pc.points)} points)"
            return None
        self.extracted_pc_down, extracted_fpfh = self._prepare_pc(self.extracted_pc)
        if self.extracted_pc_down is None:
            return None

        try:
            self.global_reg = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                self.gt_pc_down, self.extracted_pc_down, self.gt_fpfh, extracted_fpfh,
                option=o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=self.global_max_correspondence_distance))
        except:
            self.reason = "Global registration failed"
            return None

        pose = self.global_reg.transformation
        for max_correspondence_distance in self.max_correspondence_distances:
            self.reg = o3d.pipelines.registration.registration_icp(
                self.gt_pc_down, self.extracted_pc_down,
                max_correspondence_distance, init=pose,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
            pose = self.reg.transformation

        return pose

    def extract_pc(self, mask, depth):
        masked_depth = depth.copy()
        masked_depth[mask == 0] = 0
        if masked_depth.dtype.byteorder != '=':
            masked_depth = masked_depth.astype(masked_depth.dtype.newbyteorder('='))
        masked_depth = o3d.geometry.Image(masked_depth)
        height, width = mask.shape[:2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, self.K)
        extracted_pc = o3d.geometry.PointCloud.create_from_depth_image(
            masked_depth, intrinsic, depth_scale=(1 / self.depth_scale))
        return extracted_pc

    def _init_gt_pc(self):
        self.gt_pc = self._get_gt_pc()
        self.gt_pc_down, self.gt_fpfh = self._prepare_pc(self.gt_pc, no_noise_allowed=True)

    def _get_gt_pc(self):
        raise NotImplementedError()

    def _prepare_pc(self, pc, no_noise_allowed=False):
        pc_down = pc.voxel_down_sample(self.voxel_size)

        pc_clusters_labels = pc_down.cluster_dbscan(self.voxel_size * 2, 0)
        pc_clusters_labels = np.array(pc_clusters_labels)
        unique_labels, counts = np.unique(pc_clusters_labels, return_counts=True)
        if no_noise_allowed:
            assert -1 not in unique_labels, \
                "There are noise points after clustering " \
                "and no_noise_allowed is True"
        label = unique_labels[np.argmax(counts)]
        if label == -1:
            self.reason = "Too many noise points after clustering"
            return None, None
        indices = np.where(pc_clusters_labels == label)[0]
        pc_down = pc_down.select_by_index(indices)

        pc_down.estimate_normals(search_param=
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100))
        return pc_down, fpfh


class BoxSegmentation:
    def __init__(self, erosion_size):
        self.erosion_size = erosion_size
        if self.erosion_size > 0:
            self.erosion_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                (2 * self.erosion_size + 1, 2 * self.erosion_size + 1),
                (self.erosion_size, self.erosion_size))

    def segment_box(self, image, box):
        assert box in ("white", "green", "red")
        if box == "white":
            return self.segment_white_box(image)
        if box == "green":
            return self.segment_green_box(image)
        if box == "red":
            return self.segment_red_box(image)

    def segment_white_box(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        min_color = np.array([175, 40, 30], dtype=np.uint8)
        max_color = np.array([240, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, min_color, max_color)
        refined_mask, _ = refine_mask_by_polygons(
            mask, min_polygon_length=500, max_polygon_length=10000,
            select_top_n_polygons_by_length=1)
        box_mask = (mask == 0) & (refined_mask != 0)
        box_mask = box_mask.astype(np.uint8) * 255
        box_mask, _ = refine_mask_by_polygons(
            box_mask, min_polygon_length=200, max_polygon_length=2000,
            select_top_n_polygons_by_length=1)

        if self.erosion_size > 0:
            box_mask = cv2.erode(box_mask, self.erosion_element)

        return box_mask

    def segment_green_box(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        min_color = np.array([80, 40, 30], dtype=np.uint8)
        max_color = np.array([140, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, min_color, max_color)
        box_mask, _ = refine_mask_by_polygons(
            mask, min_polygon_length=300, max_polygon_length=4000,
            min_polygon_area_length_ratio=20, select_top_n_polygons_by_length=1)

        if self.erosion_size > 0:
            box_mask = cv2.erode(box_mask, self.erosion_element)

        return box_mask

    def segment_red_box(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        hsv[:, :, 0] += 100
        min_color = np.array([80, 50], dtype=np.uint8)
        max_color = np.array([110, 255], dtype=np.uint8)
        h = hsv[:, :, 0]
        sv = get_sv(hsv)
        h_sv = np.dstack((h, sv))
        mask = cv2.inRange(h_sv, min_color, max_color)
        box_mask, _ = refine_mask_by_polygons(
            mask, min_polygon_length=300, max_polygon_length=4000,
            min_polygon_area_length_ratio=20, select_top_n_polygons_by_length=1)

        if self.erosion_size > 0:
            box_mask = cv2.erode(box_mask, self.erosion_element)

        return box_mask


class BoxPoseEstimation(ObjectPoseEstimation):
    def __init__(self, edges_sizes, edge_points_per_cm, voxel_size, depth_scale, K, D,
            global_max_correspondence_distance, max_correspondence_distances):
        super().__init__(voxel_size, depth_scale, K, D,
            global_max_correspondence_distance, max_correspondence_distances)

        self.edges_sizes = edges_sizes
        self.edge_points_per_cm = edge_points_per_cm
        super()._init_gt_pc()

    def _get_gt_pc(self):
        return self._get_box_pc()

    def _get_box_pc(self):
        box_points = self._get_box_points()
        box_pc = o3d.geometry.PointCloud()
        box_pc.points = o3d.utility.Vector3dVector(box_points)
        return box_pc

    def _get_box_points(self):
        faces = list()
        for axis_index in (0, 1, 2):
            axis_size = self.edges_sizes[axis_index]
            for displacement in (-axis_size / 2, axis_size / 2):
                face = self._get_box_face(axis_index, displacement)
                faces.append(face)
        points = np.vstack(faces)
        return points

    def _get_box_face(self, axis_index, displacement):
        face_axes_indices = np.delete(np.array([0, 1, 2]), axis_index)
        face_edges_sizes = self.edges_sizes[face_axes_indices]
        face = np.mgrid[
            -face_edges_sizes[0] / 2 : face_edges_sizes[0] / 2 : int(face_edges_sizes[0] * 100 * self.edge_points_per_cm) * 1j,
            -face_edges_sizes[1] / 2 : face_edges_sizes[1] / 2 : int(face_edges_sizes[1] * 100 * self.edge_points_per_cm) * 1j]
        face = face.reshape(2, -1).swapaxes(0, 1)
        face = np.hstack((face, np.full((len(face), 1), displacement)))
        axes_order = np.hstack((face_axes_indices, axis_index))
        face[:, axes_order] = face[:, [0, 1, 2]]
        return face

    @staticmethod
    def align_box_poses(ref_box_pose, box_pose):
        box_pose = box_pose.copy()
        ref_z = ref_box_pose[:3, 2]
        z = box_pose[:3, 2]
        if np.dot(ref_z, z) < 0:
            correction, _ = cv2.Rodrigues(np.array([np.pi, 0, 0]))
            box_pose_rotation = box_pose[:3, :3]
            np.matmul(box_pose_rotation, correction, out=box_pose_rotation)

        ref_x = ref_box_pose[:3, 0]
        x = box_pose[:3, 0]
        if np.dot(ref_x, x) < 0:
            correction, _ = cv2.Rodrigues(np.array([0, 0, np.pi]))
            box_pose_rotation = box_pose[:3, :3]
            np.matmul(box_pose_rotation, correction, out=box_pose_rotation)

        return box_pose
