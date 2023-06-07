import numpy as np
import cv2
import open3d as o3d
from segment_by_color import refine_mask_by_polygons


class BoxSegmentator:
    def __init__(self, erosion_size):
        self.erosion_size = erosion_size

    def segment_box(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        min_color = np.array([165, 50, 60], dtype=np.uint8)
        max_color = np.array([220, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, min_color, max_color)
        refined_mask, _ = refine_mask_by_polygons(
            mask, min_polygon_length=500, max_polygon_length=10000)
        box_mask = (mask == 0) & (refined_mask != 0)
        box_mask = box_mask.astype(np.uint8) * 255
        box_mask, _ = refine_mask_by_polygons(
            box_mask, min_polygon_length=200, max_polygon_length=2000)

        if self.erosion_size > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                (2 * self.erosion_size + 1, 2 * self.erosion_size + 1),
                (self.erosion_size, self.erosion_size))
            box_mask = cv2.erode(box_mask, element)

        return box_mask


class BoxPoseEstimator:
    def __init__(self, edges_sizes, edge_points_per_cm, voxel_size, depth_scale, K, D,
            mutual_filter, global_max_dist, global_checker_max_dist, max_dist):
        self.edges_sizes = edges_sizes
        self.edge_points_per_cm = edge_points_per_cm
        self.voxel_size = voxel_size
        self.depth_scale = depth_scale
        self.K = K
        self.D = D
        self.mutual_filter = mutual_filter
        self.global_max_dist = global_max_dist
        self.global_checker_max_dist = global_checker_max_dist
        self.max_dist = max_dist

        self.box_pc = self._get_box_pc()
        self.box_pc_down, self.box_fpfh = self._prepare_pc(self.box_pc)

    def estimate_box_pose(self, mask, depth):
        extracted_box_pc = self._extract_box_pc(mask, depth)
        if len(extracted_box_pc.points) < 1000:
            return None, None, None, None
        extracted_box_pc_down, extracted_box_fpfh = self._prepare_pc(extracted_box_pc)

        global_reg = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.box_pc_down, extracted_box_pc_down, self.box_fpfh, extracted_box_fpfh,
            self.mutual_filter, self.global_max_dist,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=3, checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    self.global_checker_max_dist)],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

        transform_init = global_reg.transformation
        reg = o3d.pipelines.registration.registration_icp(
            self.box_pc, extracted_box_pc, self.max_dist, init=transform_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

        return global_reg, reg, extracted_box_pc, extracted_box_pc_down

    def _extract_box_pc(self, mask, depth):
        depth[mask == 0] = 0
        depth = o3d.geometry.Image(depth)
        height, width = mask.shape[:2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, self.K)
        extracted_box_pc = o3d.geometry.PointCloud.create_from_depth_image(
            depth, intrinsic, depth_scale=(1 / self.depth_scale))
        return extracted_box_pc

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
            -face_edges_sizes[0] / 2 : face_edges_sizes[0] / 2 : face_edges_sizes[0] * 100 * self.edge_points_per_cm * 1j,
            -face_edges_sizes[1] / 2 : face_edges_sizes[1] / 2 : face_edges_sizes[1] * 100 * self.edge_points_per_cm * 1j]
        face = face.reshape(2, -1).swapaxes(0, 1)
        face = np.hstack((face, np.full((len(face), 1), displacement)))
        axes_order = np.hstack((face_axes_indices, axis_index))
        face[:, axes_order] = face[:, [0, 1, 2]]
        return face

    def _prepare_pc(self, pc):
        pc_down = pc.voxel_down_sample(self.voxel_size)
        pc_down.estimate_normals(search_param=
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100))
        return pc_down, fpfh 
