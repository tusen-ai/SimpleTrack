import numpy as np, numba


def pc_in_box(box, pc, box_scaling=1.0):
    center_x, center_y, length, width = \
        box[0], box[1], box[4], box[5]
    center_z, height = box[2], box[6]
    yaw = box[3]
    return pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.0):
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