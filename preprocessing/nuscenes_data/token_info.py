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


def set_selected_or_not(frame_tokens):
    """ under the 20hz setting, 
        we have to set whether to use a certain frame
        1. select at the interval of 1 frames
        2. if meet key frame, reset the counter
    """
    counter = -1
    selected = list()
    frame_num = len(frame_tokens)
    for _, tokens in enumerate(frame_tokens):
        is_key_frame = tokens[1]
        counter += 1
        if is_key_frame:
            selected.append(True)
            counter = 0
            continue
        else:
            if counter % 2 == 0:
                selected.append(True)
            else:
                selected.append(False)
    result_tokens = [(list(frame_tokens[i]) + [selected[i]]) for i in range(frame_num)]
    return result_tokens


def main(nusc, scene_names, root_path, token_folder, mode):
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
        frame_tokens = list()

        while True:
            # find the path to lidar data
            if mode == '2hz':
                frame_data = nusc.get('sample', cur_sample_token)
                frame_tokens.append(cur_sample_token)
            elif mode == '20hz':
                frame_data = nusc.get('sample_data', cur_sample_token)
                frame_tokens.append((cur_sample_token, frame_data['is_key_frame'], frame_data['sample_token']))

            # clean up and prepare for the next
            cur_sample_token = frame_data['next']
            if cur_sample_token == '':
                break
        
        if mode == '20hz':
            frame_tokens = set_selected_or_not(frame_tokens)
        f = open(os.path.join(token_folder, '{:}.json'.format(scene_name)), 'w')
        json.dump(frame_tokens, f)
        f.close()

        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    print('token info')
    os.makedirs(args.data_folder, exist_ok=True)

    token_folder = os.path.join(args.data_folder, 'token_info')
    os.makedirs(token_folder, exist_ok=True)

    val_scene_names = splits.create_splits_scenes()['val']
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)
    main(nusc, val_scene_names, args.raw_data_folder, token_folder, args.mode)
