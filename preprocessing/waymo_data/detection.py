""" Extract the detections from .bin files
    Each sequence is a .npz file containing three fields: bboxes, types, ids.
    bboxes, types, and ids follow the same format:
    [[bboxes in frame 0],
     [bboxes in frame 1],
     ...
     [bboxes in the last frame]]
"""
import os, numpy as np, argparse, json
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from google.protobuf.descriptor import FieldDescriptor as FD
tf.enable_eager_execution()
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='public')
parser.add_argument('--file_path', type=str, default='validation.bin')
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/mot/')
parser.add_argument('--metadata', action='store_true', default=False)
args = parser.parse_args()


def bbox_dict2array(box_dict):
    """transform box dict in waymo_open_format to array
    Args:
        box_dict ([dict]): waymo_open_dataset formatted bbox
    """
    result = np.array([
        box_dict['center_x'],
        box_dict['center_y'],
        box_dict['center_z'],
        box_dict['heading'],
        box_dict['length'],
        box_dict['width'],
        box_dict['height'],
        box_dict['score']
    ])
    return result


def str_list_to_int(lst):
    result = []
    for t in lst:
        try:
            t = int(t)
            result.append(t)
        except:
            continue
    return result


def main(name, data_folder, det_folder, file_path, out_folder):
    # load timestamp and segment names
    ts_info_folder = os.path.join(data_folder, 'ts_info')
    ts_files = os.listdir(ts_info_folder)
    ts_info = dict()
    segment_name_list = list()
    for ts_file_name in ts_files:
        ts = json.load(open(os.path.join(ts_info_folder, ts_file_name), 'r'))
        segment_name = ts_file_name.split('.')[0]
        ts_info[segment_name] = ts
        segment_name_list.append(segment_name)
    
    # load detection file
    det_folder = os.path.join(det_folder, name)
    f = open(file_path, 'rb')
    objects = metrics_pb2.Objects()
    objects.ParseFromString(f.read())
    f.close()
    
    # parse and aggregate detections
    objects = objects.objects
    object_num = len(objects)

    result_bbox, result_type, result_velo, result_accel, result_ids = dict(), dict(), dict(), dict(), dict()
    for seg_name in ts_info.keys():
        result_bbox[seg_name] = dict()
        result_type[seg_name] = dict()
        result_velo[seg_name] = dict()
        result_accel[seg_name] = dict()
        result_ids[seg_name] = dict()

    print('Converting')
    pbar = tqdm(total=object_num)    
    for _i in range(object_num):
        instance = objects[_i]
        segment = instance.context_name
        time_stamp = instance.frame_timestamp_micros

        box = instance.object.box
        bbox_dict = {
            'center_x': box.center_x,
            'center_y': box.center_y,
            'center_z': box.center_z,
            'width': box.width,
            'length': box.length,
            'height': box.height,
            'heading': box.heading,
            'score': instance.score
        }
        bbox_array = bbox_dict2array(bbox_dict)
        obj_type = instance.object.type
        
        if args.metadata:
            meta_data = instance.object.metadata
            velo = (meta_data.speed_x, meta_data.speed_y)
            accel = (meta_data.accel_x, meta_data.accel_y)

        if args.id:
            id = instance.object.id

        val_index = None
        for _j in range(len(segment_name_list)):
            if segment in segment_name_list[_j]:
                val_index = _j
                break
        segment_name = segment_name_list[val_index]

        frame_number = None
        for _j in range(len(ts_info[segment_name])):
            if ts_info[segment_name][_j] == time_stamp:
                frame_number = _j
                break
        
        if str(frame_number) not in result_bbox[segment_name].keys():
            result_bbox[segment_name][str(frame_number)] = list()
            result_type[segment_name][str(frame_number)] = list()
            if args.metadata:
                result_velo[segment_name][str(frame_number)] = list()
                result_accel[segment_name][str(frame_number)] = list()
            if args.id:
                result_ids[segment_name][str(frame_number)] = list()

        result_bbox[segment_name][str(frame_number)].append(bbox_array)
        result_type[segment_name][str(frame_number)].append(obj_type)
        if args.metadata:
            result_velo[segment_name][str(frame_number)].append(velo)
            result_accel[segment_name][str(frame_number)].append(accel)
        if args.id:
            result_ids[segment_name][str(frame_number)].append(id)

        pbar.update(1)
    pbar.close()

    print('Saving')
    pbar = tqdm(total=len(segment_name_list))    
    # store in files
    for _i, segment_name in enumerate(segment_name_list):
        dets = result_bbox[segment_name]
        types = result_type[segment_name]
        if args.metadata:
            velos = result_velo[segment_name]
            accels = result_accel[segment_name]
        if args.id:
            ids = result_ids[segment_name]

        frame_keys = sorted(str_list_to_int(dets.keys()))
        max_frame = max(frame_keys)
        bboxes = list()
        obj_types = list()
        velocities, accelerations, id_names = list(), list(), list()
        for key in range(max_frame + 1):
            if str(key) in dets.keys():
                bboxes.append(dets[str(key)])
                obj_types.append(types[str(key)])
                if args.metadata:
                    velocities.append(velos[str(key)])
                    accelerations.append(accels[str(key)])
                if args.id:
                    id_names.append(ids[str(key)])
            else:
                bboxes.append([])
                obj_types.append([])
                if args.metadata:
                    velocities.append([])
                    accelerations.append([])
                if args.id:
                    id_names.append([])
        result = {'bboxes': bboxes, 'types': obj_types}
        if args.metadata:
            result['velos'] = velocities
            result['accels'] = accelerations
        if args.id:
            result['ids'] = id_names

        np.savez_compressed(os.path.join(out_folder, "{:}.npz".format(segment_name)), **result)
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    det_folder = os.path.join(args.data_folder, 'detection')
    os.makedirs(det_folder, exist_ok=True)
    output_folder = os.path.join(det_folder, args.name, 'dets')
    os.makedirs(args.output_folder, exist_ok=True)

    main(args.name, args.data_folder, det_folder, args.file_path, output_folder)
