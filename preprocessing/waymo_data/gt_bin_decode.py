""" Process the .bin file of ground truth, and save it in our detection format.
    Each sequence is a .npz file containing three fields: bboxes, types, ids.
    bboxes, types, and ids follow the same format:
    [[bboxes in frame 0],
     [bboxes in frame 1],
     ...
     [bboxes in the last frame]]
"""
import os, numpy as np, argparse, json
from mot_3d.data_protos import BBox
import mot_3d.utils as utils
import tensorflow.compat.v1 as tf
from google.protobuf.descriptor import FieldDescriptor as FD
tf.enable_eager_execution()
from waymo_open_dataset.protos import metrics_pb2


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='/../../../datasets/waymo/mot/')
parser.add_argument('--file_path', type=str, default='../datasets/waymo/mot/info/validation_gt.bin')
args = parser.parse_args()


def main(file_path, out_folder, data_folder):
    # load time stamp
    ts_info_folder = os.path.join(data_folder, 'ts_info')
    time_stamp_info = dict()
    ts_files = os.listdir(ts_info_folder)
    for ts_file_name in ts_files:
        segment_name = ts_file_name.split('.')[0]
        ts_path = os.path.join(ts_info_folder, '{}.json'.format(segment_name))
        f = open(ts_path, 'r')
        ts = json.load(f)
        f.close()
        time_stamp_info[segment_name] = ts
    
    # segment name list
    segment_name_list = list()
    for ts_file_name in ts_files:
        segment_name = ts_file_name.split('.')[0]
        segment_name_list.append(segment_name)
    
    # load gt.bin file
    f = open(file_path, 'rb')
    objects = metrics_pb2.Objects()
    objects.ParseFromString(f.read())
    f.close()

    # parse and aggregate detections
    objects = objects.objects
    object_num = len(objects)

    result_bbox, result_type, result_id = dict(), dict(), dict()
    for seg_name in time_stamp_info.keys():
        result_bbox[seg_name] = dict()
        result_type[seg_name] = dict()
        result_id[seg_name] = dict()
    
    for i in range(object_num):
        instance = objects[i]
        segment = instance.context_name
        time_stamp = instance.frame_timestamp_micros

        box = instance.object.box
        box_dict = {
            'center_x': box.center_x,
            'center_y': box.center_y,
            'center_z': box.center_z,
            'width': box.width,
            'length': box.length,
            'height': box.height,
            'heading': box.heading,
            'score': instance.score
        }

        val_index = None
        for _j in range(len(segment_name_list)):
            if segment in segment_name_list[_j]:
                val_index = _j
                break
        segment_name = segment_name_list[val_index]

        frame_number = None
        for _j in range(len(time_stamp_info[segment_name])):
            if time_stamp_info[segment_name][_j] == time_stamp:
                frame_number = _j
                break
        
        if str(frame_number) not in result_bbox[segment_name].keys():
            result_bbox[segment_name][str(frame_number)] = list()
            result_type[segment_name][str(frame_number)] = list()
            result_id[segment_name][str(frame_number)] = list()
        
        result_bbox[segment_name][str(frame_number)].append(BBox.bbox2array(BBox.dict2bbox(box_dict)))
        result_type[segment_name][str(frame_number)].append(instance.object.type)
        result_id[segment_name][str(frame_number)].append(instance.object.id)

        if (i + 1) % 10000 == 0:
            print(i + 1, ' / ', object_num)
    
    # store in files
    for _i, segment_name in enumerate(time_stamp_info.keys()):
        dets = result_bbox[segment_name]
        types = result_type[segment_name]
        ids = result_id[segment_name]
        print('{} / {}'.format(_i + 1, len(time_stamp_info.keys())))

        frame_keys = sorted(utils.str2int(dets.keys()))
        max_frame = max(frame_keys)
        obj_ids, bboxes, obj_types = list(), list(), list()

        for key in range(max_frame + 1):
            if str(key) in dets.keys():
                bboxes.append(dets[str(key)])
                obj_types.append(types[str(key)])
                obj_ids.append(ids[str(key)])
            else:
                bboxes.append([])
                obj_types.append([])
                obj_ids.append([])

        np.savez_compressed(os.path.join(out_folder, "{}.npz".format(segment_name)), 
            bboxes=bboxes, types=obj_types, ids=obj_ids)


if __name__ == '__main__':
    out_folder = os.path.join(args.data_folder, 'detection', 'gt', 'dets')
    os.makedirs(out_folder)
    main(args.file_path, out_folder, args.data_folder)
