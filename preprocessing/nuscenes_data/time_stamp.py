import os, numpy as np, nuscenes, argparse, json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='../../../raw/nuscenes/data/sets/nuscenes/')
parser.add_argument('--data_folder', type=str, default='../../../datasets/nuscenes/')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
args = parser.parse_args()


def main(nusc, scene_names, root_path, ts_folder, mode):
    pbar = tqdm(total=len(scene_names))
    for scene_index, scene_info in enumerate(nusc.scene):
        scene_name = scene_info['name']
        if scene_name not in scene_names:
            continue

        first_sample_token = scene_info['first_sample_token']
        last_sample_token = scene_info['last_sample_token']
        frame_data = nusc.get('sample', first_sample_token)
        if mode == '20hz':
            cur_sample_token = frame_data['data']['LIDAR_TOP']
        elif mode == '2hz':
            cur_sample_token = deepcopy(first_sample_token)
        time_stamps = list()

        while True:
            if mode == '2hz':
                frame_data = nusc.get('sample', cur_sample_token)
                time_stamps.append(frame_data['timestamp'])
            elif mode == '20hz':
                frame_data = nusc.get('sample_data', cur_sample_token)
                # time stamp and if key frame
                time_stamps.append((frame_data['timestamp'], frame_data['is_key_frame']))

            # clean up and prepare for the next
            cur_sample_token = frame_data['next']
            if cur_sample_token == '':
                break
        f = open(os.path.join(ts_folder, '{:}.json'.format(scene_name)), 'w')
        json.dump(time_stamps, f)
        f.close()
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    ts_folder = os.path.join(args.data_folder, 'ts_info')
    os.makedirs(ts_folder, exist_ok=True)

    val_scene_names = splits.create_splits_scenes()['val']
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)
    main(nusc, val_scene_names, args.raw_data_folder, ts_folder, args.mode)
