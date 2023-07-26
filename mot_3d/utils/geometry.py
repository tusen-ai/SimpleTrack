import numpy as np
from copy import deepcopy
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numba
from ..data_protos import BBox


__all__ = ['pc_in_box', 'downsample', 'pc_in_box_2D',
           'apply_motion_to_points', 'make_transformation_matrix',
           'iou2d', 'iou3d', 'pc2world', 'giou2d', 'giou3d', 
           'back_step_det', 'm_distance', 'velo2world', 'score_rectification']


def velo2world(ego_matrix, velo):
    """ transform local velocity [x, y] to global
    """
    new_velo = velo[:, np.newaxis]
    new_velo = ego_matrix[:2, :2] @ new_velo
    return new_velo[:, 0]


def apply_motion_to_points(points, motion, pre_move=0):
    transformation_matrix = make_transformation_matrix(motion)
    points = deepcopy(points)
    points = points + pre_move
    new_points = np.concatenate((points,
                                 np.ones(points.shape[0])[:, np.newaxis]),
                                 axis=1)

    new_points = transformation_matrix @ new_points.T
    new_points = new_points.T[:, :3]
    new_points -= pre_move
    return new_points


@numba.njit
def downsample(points, voxel_size=0.05):
    sample_dict = dict()
    for i in range(points.shape[0]):
        point_coord = np.floor(points[i] / voxel_size)
        sample_dict[(int(point_coord[0]), int(point_coord[1]), int(point_coord[2]))] = True
    res = np.zeros((len(sample_dict), 3), dtype=np.float32)
    idx = 0
    for k, v in sample_dict.items():
        res[idx, 0] = k[0] * voxel_size + voxel_size / 2
        res[idx, 1] = k[1] * voxel_size + voxel_size / 2
        res[idx, 2] = k[2] * voxel_size + voxel_size / 2
        idx += 1
    return res


# def pc_in_box(box, pc, box_scaling=1.5):
#     center_x, center_y, length, width = \
#         box['center_x'], box['center_y'], box['length'], box['width']
#     center_z, height = box['center_z'], box['height']
#     yaw = box['heading']

#     rx = np.abs((pc[:, 0] - center_x) * np.cos(yaw) + (pc[:, 1] - center_y) * np.sin(yaw))
#     ry = np.abs((pc[:, 0] - center_x) * -(np.sin(yaw)) + (pc[:, 1] - center_y) * np.cos(yaw))
#     rz = np.abs(pc[:, 2] - center_z)

#     mask_x = (rx < (length * box_scaling / 2))
#     mask_y = (ry < (width * box_scaling / 2))
#     mask_z = (rz < (height / 2))

#     mask = mask_x * mask_y * mask_z
#     indices = np.argwhere(mask == 1).reshape(-1)
#     return pc[indices, :]


# def pc_in_box_2D(box, pc, box_scaling=1.0):
#     center_x, center_y, length, width = \
#         box['center_x'], box['center_y'], box['length'], box['width']
#     yaw = box['heading']
    
#     cos = np.cos(yaw) 
#     sin = np.sin(yaw)
#     rx = np.abs((pc[:, 0] - center_x) * cos + (pc[:, 1] - center_y) * sin)
#     ry = np.abs((pc[:, 0] - center_x) * -(sin) + (pc[:, 1] - center_y) * cos)

#     mask_x = (rx < (length * box_scaling / 2))
#     mask_y = (ry < (width * box_scaling / 2))

#     mask = mask_x * mask_y
#     indices = np.argwhere(mask == 1).reshape(-1)
#     return pc[indices, :]


def pc_in_box(box, pc, box_scaling=1.5):
    center_x, center_y, length, width = \
        box.x, box.y, box.l, box.w
    center_z, height = box.z, box.h
    yaw = box.o
    return pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.5):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)
        rz = np.abs(pc[i, 2] - center_z)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2) and rz < (height * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def pc_in_box_2D(box, pc, box_scaling=1.0):
    center_x, center_y, length, width = \
        box.x, box.y, box.l, box.w
    center_z, height = box.z, box.h
    yaw = box.o
    return pc_in_box_2D_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_2D_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.0):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def make_transformation_matrix(motion):
    x, y, z, theta = motion
    transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                                      [np.sin(theta),  np.cos(theta), 0, y],
                                      [0            ,  0            , 1, z],
                                      [0            ,  0            , 0, 1]])
    return transformation_matrix


def iou2d(box_a, box_b):
    boxa_corners = np.array(BBox.box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap = reca.intersection(recb).area
    area_a = reca.area
    area_b = recb.area
    iou = overlap / (area_a + area_b - overlap + 1e-10)
    return iou


def iou3d(box_a, box_b):
    boxa_corners = np.array(BBox.box2corners2d(box_a))
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap_area = reca.intersection(recb).area
    iou_2d = overlap_area / (reca.area + recb.area - overlap_area)

    ha, hb = box_a.h, box_b.h
    za, zb = box_a.z, box_b.z
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    overlap_volume = overlap_area * overlap_height
    union_volume = box_a.w * box_a.l * ha + box_b.w * box_b.l * hb - overlap_volume
    iou_3d = overlap_volume / (union_volume + 1e-5)

    return iou_2d, iou_3d


def pc2world(ego_matrix, pcs):
    new_pcs = np.concatenate((pcs,
                              np.ones(pcs.shape[0])[:, np.newaxis]),
                              axis=1)
    new_pcs = ego_matrix @ new_pcs.T
    new_pcs = new_pcs.T[:, :3]
    return new_pcs


def giou2d(box_a: BBox, box_b: BBox):
    boxa_corners = np.array(BBox.box2corners2d(box_a))
    boxb_corners = np.array(BBox.box2corners2d(box_b))
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    
    # compute intersection and union
    I = reca.intersection(recb).area
    U = box_a.w * box_a.l + box_b.w * box_b.l - I

    # compute the convex area
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area

    # compute giou
    return I / U - (C - U) / C


def giou3d(box_a: BBox, box_b: BBox):
    boxa_corners = np.array(BBox.box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    ha, hb = box_a.h, box_b.h
    za, zb = box_a.z, box_b.z
    overlap_height = max(0, min([(za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2), ha, hb]))
    union_height = max([(za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2), ha, hb])
    
    # compute intersection and union
    I = reca.intersection(recb).area * overlap_height
    U = box_a.w * box_a.l * ha + box_b.w * box_b.l * hb - I

    # compute the convex area
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area * union_height

    # compute giou
    giou = I / U - (C - U) / C
    return giou


def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area


def back_step_det(det: BBox, velo, time_lag):
    result = BBox()
    BBox.copy_bbox(result, det)
    result.x -= (time_lag * velo[0])
    result.y -= (time_lag * velo[1])
    return result


def diff_orientation_correction(diff):
    """
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    """
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    return diff


def m_distance(det, trk, trk_inv_innovation_matrix=None):
    det_array = BBox.bbox2array(det)[:7]
    trk_array = BBox.bbox2array(trk)[:7]
    
    diff = np.expand_dims(det_array - trk_array, axis=1)
    corrected_yaw_diff = diff_orientation_correction(diff[3])
    diff[3] = corrected_yaw_diff

    if trk_inv_innovation_matrix is not None:
        result = \
            np.sqrt(np.matmul(np.matmul(diff.T, trk_inv_innovation_matrix), diff)[0][0])
    else:
        result = np.sqrt(np.dot(diff.T, diff))
    return result


def score_rectification(dets, gts):
    """ rectify the scores of detections according to their 3d iou with gts
    """
    result = deepcopy(dets)
    
    if len(gts) == 0:
        for i, _ in enumerate(dets):
            result[i].s = 0.0
        return result

    if len(dets) == 0:
        return result

    iou_matrix = np.zeros((len(dets), len(gts)))
    for i, d in enumerate(dets):
        for j, g in enumerate(gts):
            iou_matrix[i, j] = iou3d(d, g)[1]
    max_index = np.argmax(iou_matrix, axis=1)
    max_iou = np.max(iou_matrix, axis=1)
    index = list(reversed(sorted(range(len(dets)), key=lambda k:max_iou[k])))

    matched_gt = []
    for i in index:
        if max_iou[i] >= 0.1 and max_index[i] not in matched_gt:
            result[i].s = max_iou[i]
            matched_gt.append(max_index[i])
        elif max_iou[i] >= 0.1 and max_index[i] in matched_gt:
            result[i].s = 0.2
        else:
            result[i].s = 0.05

    return result
