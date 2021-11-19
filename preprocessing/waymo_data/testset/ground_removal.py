""" Remove the ground of point clouds
    The output are two types of .npz files
    clean_pc: {str(frame_number): point cloud after ground removal}
    ground_pc: {str(frame_num): ground point cloud}
"""
import os, numpy as np, argparse, multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/mot_test',
    help='the location of data')
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()
args.raw_pc_folder = os.path.join(args.data_folder, 'pc', 'raw_pc')
args.clean_pc_folder = os.path.join(args.data_folder, 'pc', 'clean_pc')
args.ground_pc_folder = os.path.join(args.data_folder, 'pc', 'ground_pc')
if not os.path.exists(args.clean_pc_folder):
    os.makedirs(args.clean_pc_folder)
if not os.path.exists(args.ground_pc_folder):
    os.makedirs(args.ground_pc_folder)  


def str2int(strs):
    result = [int(s) for s in strs]
    return result


def extract_init_seed(pts_sort, n_lpr, th_seed):
    lpr = np.mean(pts_sort[:n_lpr, 2])
    seed = pts_sort[pts_sort[:, 2] < lpr + th_seed, :]
    return seed


def get_ground(pts):
    th_seeds_ = 1.2
    num_lpr_ = 20
    n_iter = 10
    th_dist_ = 0.3
    pts_sort = pts[pts[:, 2].argsort(), :]
    pts_g = extract_init_seed(pts_sort, num_lpr_, th_seeds_)
    normal_ = np.zeros(3)
    for i in range(n_iter):
        mean = np.mean(pts_g, axis=0)[:3]
        xx = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 0] - mean[0]))
        xy = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 1] - mean[1]))
        xz = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 2] - mean[2]))
        yy = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 1] - mean[1]))
        yz = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 2] - mean[2]))
        zz = np.mean((pts_g[:, 2] - mean[2]) * (pts_g[:, 2] - mean[2]))
        cov = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        U, S, V = np.linalg.svd(cov)
        normal_ = U[:, 2]
        d_ = -normal_.dot(mean)
        th_dist_d_ = th_dist_ - d_
        result = pts[:, :3].dot(normal_)
        pts_n_g = pts[result>th_dist_d_]
        pts_g = pts[result<th_dist_d_]
    return pts_g, pts_n_g


def main(token, raw_pc_folder, clean_pc_folder, ground_pc_folder):
    file_names = sorted(os.listdir(raw_pc_folder))
    for file_index, file_name in enumerate(file_names):
        # for multiprocessing
        if file_index % token[1] != token[0]:
            continue
        raw_pc_data = np.load(os.path.join(raw_pc_folder, file_name), allow_pickle=True)
        keys = str2int(list(raw_pc_data.keys()))

        clean_pcs = dict()
        ground_pcs = dict()
        for frame_index, frame_key in enumerate(keys):
            frame_pc = raw_pc_data[str(frame_key)]
            ground_frame_pc, clean_frame_pc = get_ground(frame_pc)
            
            clean_pcs[str(frame_key)] = clean_frame_pc
            ground_pcs[str(frame_key)] = ground_frame_pc

            if (frame_index + 1) % 10 == 0:
                print('Ground Removal SEQ {} / {}, Frame {} / {}'.format(file_index + 1, len(file_names), frame_index + 1, len(keys)))
        
        np.savez_compressed(os.path.join(clean_pc_folder, file_name), **clean_pcs)
        np.savez_compressed(os.path.join(ground_pc_folder, file_name), **ground_pcs)


if __name__ == '__main__':
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=((token, args.process), args.raw_pc_folder, args.clean_pc_folder, args.ground_pc_folder))
        pool.close()
        pool.join()
    else:
        main((0, 1), args.raw_pc_folder, args.clean_pc_folder, args.ground_pc_folder)