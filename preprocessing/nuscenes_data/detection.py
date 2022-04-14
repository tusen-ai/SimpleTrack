import os, argparse, numpy as np, json
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/nuscenes/')
parser.add_argument('--detection_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/nuscenes/')
parser.add_argument('--det_name', type=str, default='cp')
parser.add_argument('--raw_file_path', type=str, 
                    default='/juno/u/tsadja/SimpleTrack/dataset/nuscenes/validation_2hz/detection/megvii/raw/val.json')
parser.add_argument('--velo', action='store_true', default=False)
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()


def get_sample_tokens(data_folder, mode):
    token_folder = os.path.join(data_folder, 'token_info')
    file_names = sorted(os.listdir(token_folder))
    result = dict()
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(token_folder, file_name)
        scene_name = file_name.split('.')[0]
        tokens = json.load(open(file_path, 'r'))

        if mode == '2hz':
            result[scene_name] = tokens
        elif mode == '20hz':
            result[scene_name] = [t[0] for t in tokens]
    return result


def sample_result2bbox_array(sample):
    trans, size, rot, score = \
        sample['translation'], sample['size'],sample['rotation'], sample['detection_score']
    return trans + size + rot + [score]


def main(det_name, raw_file_path, detection_folder, data_folder, mode):
    # dealing with the paths
    output_folder = os.path.join(detection_folder, det_name, 'dets')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # load the detection file
    print('LOADING RAW FILE')
    f = open(raw_file_path, 'r')
    det_data = json.load(f)['results']
    f.close()

    # prepare the scene names and all the related tokens
    tokens = get_sample_tokens(data_folder, mode)
    scene_names = sorted(list(tokens.keys()))
    bboxes, inst_types, velos = dict(), dict(), dict()
    for scene_name in scene_names:
        frame_num = len(tokens[scene_name])
        bboxes[scene_name], inst_types[scene_name] = \
            [[] for i in range(frame_num)], [[] for i in range(frame_num)]
        if args.velo:
            velos[scene_name] = [[] for i in range(frame_num)]

    # enumerate through all the frames
    sample_keys = list(det_data.keys())
    print('PROCESSING...')
    pbar = tqdm(total=len(sample_keys))
    for sample_index, sample_key in enumerate(sample_keys):
        # find the corresponding scene and frame index
        scene_name, frame_index = None, None
        for scene_name in scene_names:
            if sample_key in tokens[scene_name]:
                frame_index = tokens[scene_name].index(sample_key)
                break
        
        # extract the bboxes and types
        sample_results = det_data[sample_key]
        for sample in sample_results:
            bbox, inst_type = sample_result2bbox_array(sample), sample['detection_name']
            inst_velo = sample['velocity']
            bboxes[scene_name][frame_index] += [bbox]
            inst_types[scene_name][frame_index] += [inst_type]

            if args.velo:
                velos[scene_name][frame_index] += [inst_velo]
        pbar.update(1)
    pbar.close()

    # save the results
    print('SAVING...')
    pbar = tqdm(total=len(scene_names))
    for scene_name in scene_names:
        if args.velo:
            np.savez_compressed(os.path.join(output_folder, '{:}.npz'.format(scene_name)),
                bboxes=bboxes[scene_name], types=inst_types[scene_name], velos=velos[scene_name])
        else:
            np.savez_compressed(os.path.join(output_folder, '{:}.npz'.format(scene_name)),
                bboxes=bboxes[scene_name], types=inst_types[scene_name])
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    if args.test:
        data_folder = os.path.join(args.data_folder, 'test_{:}'.format(args.mode))
    else:
        data_folder = os.path.join(args.data_folder, 'validation_{:}'.format(args.mode))

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    detection_folder = os.path.join(data_folder, 'detection')
    if not os.path.exists(detection_folder):
        os.makedirs(detection_folder)

    main(args.det_name, args.raw_file_path, detection_folder, data_folder, args.mode)