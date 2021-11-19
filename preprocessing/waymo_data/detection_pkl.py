""" convert pkl from center point to bin format in waymo
"""
import os, numpy as np, argparse, pickle
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='velo_pkl')
parser.add_argument('--det_folder', type=str, default='../../../datasets/waymo/sot/detection/')
parser.add_argument('--pred_file', type=str, default='prediction.pkl')
parser.add_argument('--info_file', type=str, default='infos.pkl')
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/sot/')
parser.add_argument('--metadata', action='store_true', default=False)
parser.add_argument('--id', action='store_true', default=False)
args = parser.parse_args()


args.output_folder = os.path.join(args.det_folder, args.name, 'dets')
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


LABEL_TO_TYPE = {0: 1, 1:2, 2:4}


def frame_bboxes_to_objects(bbox_array, scores, labels, context_name, time_stamp):
    objects = list()
    if args.metadata:
        velos = bbox_array[:, [6, 7]]
    else:
        velos = None

    bbox_array[:, -1] = -bbox_array[:, -1] - np.pi / 2
    bbox_array = bbox_array[:, [0, 1, 2, 4, 3, 5, -1]]

    object_num = bbox_array.shape[0]
    for i in range(object_num):
        det = bbox_array[i]
        score = scores[i]
        label = labels[i]
        o = metrics_pb2.Object()
        o.context_name = context_name
        o.frame_timestamp_micros = time_stamp

        # Populating box and score.
        box = label_pb2.Label.Box()
        box.center_x = det[0]
        box.center_y = det[1]
        box.center_z = det[2]
        box.length = det[3]
        box.width = det[4]
        box.height = det[5]
        box.heading = det[-1]
        o.object.box.CopyFrom(box)
        o.score = score

        # Use correct type.
        o.object.type = LABEL_TO_TYPE[label]

        # metadata
        if args.metadata:
            o.object.metadata.speed_x = velos[i][0]
            o.object.metadata.speed_y = velos[i][1]

        # append to the results
        objects.append(o)
    return objects


def main(det_folder, pred_file, info_file):
    raw_folder = os.path.join(det_folder, 'raw')
    predictions = pickle.load(open(os.path.join(raw_folder, pred_file), 'rb'))
    infos = pickle.load(open(os.path.join(raw_folder, info_file), 'rb'))

    frame_num = len(infos)
    print('CONVERTING...')
    pbar = tqdm(total=frame_num)
    raw_objects = []
    for frame_index, frame_info in enumerate(infos):
        token = frame_info['token']
        context_name = frame_info['context_name']
        time_stamp = frame_info['frame_timestamp_micros']
        frame_dets = predictions[token]
        bbox_array, scores, labels = frame_dets['box3d_lidar'], frame_dets['scores'], frame_dets['label_preds']
        bbox_array, scores, labels = bbox_array.detach().cpu().numpy(), scores.detach().cpu().numpy(), labels.detach().cpu().numpy()

        raw_objects += frame_bboxes_to_objects(bbox_array, scores, labels, context_name, time_stamp)
        pbar.update(1)
    pbar.close()

    print('SAVING...')
    result_objects = metrics_pb2.Objects()
    for obj in raw_objects:
        result_objects.objects.append(obj)
    return result_objects


if __name__ == '__main__':
    det_folder = os.path.join(args.det_folder, args.name)
    result_objs = main(det_folder, args.pred_file, args.info_file)
    result_path = os.path.join(det_folder, 'validation.bin')

    f = open(result_path, 'wb')
    f.write(result_objs.SerializeToString())
    f.close()
