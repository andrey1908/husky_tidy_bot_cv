import numpy as np
import cv2
import open3d as o3d
from segment_by_color import refine_mask_by_polygons


def get_box_face(edges_sizes, num_edge_points, axis_index, displacement):
    face = np.mgrid[
        -edges_sizes[0] / 2 : edges_sizes[0] / 2 : num_edge_points * 1j,
        -edges_sizes[1] / 2 : edges_sizes[1] / 2 : num_edge_points * 1j]
    face = face.reshape(2, -1).swapaxes(0, 1)
    face = np.hstack((face, np.full((len(face), 1), displacement)))
    face[:, [axis_index, 2]] = face[:, [2, axis_index]]
    return face


def get_box_points(edges_sizes, num_edge_points=30):
    faces = list()
    for axis_index in (0, 1, 2):
        axis_size = edges_sizes[axis_index]
        face_edges_sizes_indices = [0, 1, 2]
        face_edges_sizes_indices[axis_index] = 2
        face_edges_sizes_indices = face_edges_sizes_indices[:2]
        face_edges_sizes = [edges_sizes[i] for i in face_edges_sizes_indices]
        for displacement in (-axis_size / 2, axis_size / 2):
            face = get_box_face(
                face_edges_sizes, num_edge_points, axis_index, displacement)
            faces.append(face)
    points = np.vstack(faces)
    return points


def get_box_pc(edge_sizes, num_edge_points=30):
    box_points = get_box_points(edge_sizes, num_edge_points=num_edge_points)
    box_pc = o3d.geometry.PointCloud()
    box_pc.points = o3d.utility.Vector3dVector(box_points)
    return box_pc


def get_box_mask(image):
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

    erosion_size = 5
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size))
    eroded_box_mask = cv2.erode(box_mask, element)
    return eroded_box_mask


def extract_box_pc(image, depth, depth_scale, K):
    mask = get_box_mask(image)
    depth[mask == 0] = 0
    depth = o3d.geometry.Image(depth)
    height, width = image.shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K)
    box_pc = o3d.geometry.PointCloud.create_from_depth_image(
        depth, intrinsic, depth_scale=1/depth_scale)
    return box_pc


def prepare_pc(pc, voxel_size=0.003):
    pc_down = pc.voxel_down_sample(voxel_size)
    pc_down.estimate_normals(search_param=
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pc_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pc_down, pc_fpfh


def estimate_box_pose(
        extracted_box_pc, box_pc,
        extracted_box_pc_down, box_pc_down,
        extracted_box_fpfh, box_fpfh,
        voxel_size=0.003):
    global_reg = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        box_pc_down, extracted_box_pc_down, box_fpfh, extracted_box_fpfh,
        True, voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3, checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    transform_init = global_reg.transformation
    reg = o3d.pipelines.registration.registration_icp(
        box_pc, extracted_box_pc, voxel_size * 1, init=transform_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return global_reg, reg
