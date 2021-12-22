from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
import os, time, numpy as np, sys, pickle as pkl
import argparse, json
from copy import deepcopy
from mot_3d.data_protos import BBox, Validity
import mot_3d.utils as utils
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='baseline')
parser.add_argument('--obj_types', type=str, default='vehicle,pedestrian,cyclist')
parser.add_argument('--result_folder', type=str, default='../mot_results/')
parser.add_argument('--data_folder', type=str, default='../datasets/waymo/mot/')
args = parser.parse_args()


def get_context_name(file_name: str):
    context = file_name.split('.')[0] # file name
    context = context.split('-')[1]   # after segment
    context = context.split('w')[0]   # before with
    context = context[:-1]
    return context

def pred_content_filter(pred_contents, pred_states):
    result_contents = list()
    for contents, states in zip(pred_contents, pred_states):
        indices = [i for i in range(len(states)) if Validity.valid(states[i])]
        frame_contents = [contents[i] for i in indices]
        result_contents.append(frame_contents)
    return result_contents

def main(name, obj_type, result_folder, raw_data_folder, output_folder, output_file_name):
    summary_folder = os.path.join(result_folder, args.src, obj_type)
    file_names = sorted(os.listdir(summary_folder))[:]

    if obj_type == 'vehicle':
        type_token = 1
    elif obj_type == 'pedestrian':
        type_token = 2
    elif obj_type == 'cyclist':
        type_token = 4
    
    ts_info_folder = os.path.join(raw_data_folder, 'ts_info')
    ego_info_folder = os.path.join(raw_data_folder, 'ego_info')
    obj_list = list()

    print('Converting TYPE {:} into WAYMO Format'.format(obj_type))
    pbar = tqdm(total=len(file_names))
    for file_index, file_name in enumerate(file_names[:]):
        file_name_prefix = file_name.split('.')[0]
        context_name = get_context_name(file_name)
        
        ts_path = os.path.join(ts_info_folder, '{}.json'.format(file_name_prefix))
        ts_data = json.load(open(ts_path, 'r')) # list of time stamps by order of frame

        # load ego motions
        ego_motions = np.load(os.path.join(ego_info_folder, '{:}.npz'.format(file_name_prefix)), allow_pickle=True)

        pred_result = np.load(os.path.join(summary_folder, file_name), allow_pickle=True)
        pred_ids, pred_bboxes, pred_states = pred_result['ids'], pred_result['bboxes'], pred_result['states'] 
        pred_bboxes = pred_content_filter(pred_bboxes, pred_states)
        pred_ids = pred_content_filter(pred_ids, pred_states)
        pred_velos, pred_accels = None, None

        if args.velo:
            pred_velos = pred_result['velos']
            pred_velos = pred_content_filter(pred_velos, pred_states)
        if args.accel:
            pred_accels = pred_result['accels']
            pred_accels = pred_content_filter(pred_accels, pred_states)
        obj_list += create_sequence(pred_ids, pred_bboxes, type_token, context_name, 
            ts_data, ego_motions, pred_velos, pred_accels)
        pbar.update(1)
    pbar.close()
    objects = metrics_pb2.Objects()
    for obj in obj_list:
        objects.objects.append(obj)

    output_folder = os.path.join(output_folder, obj_type)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, '{:}.bin'.format(output_file_name))
    f = open(output_path, 'wb')
    f.write(objects.SerializeToString())
    f.close()
    return


def create_single_pred_bbox(id, bbox, type_token, time_stamp, context_name, inv_ego_motion, velo, accel):
    o = metrics_pb2.Object()
    o.context_name = context_name
    o.frame_timestamp_micros = time_stamp
    box = label_pb2.Label.Box()
    
    proto_box = BBox.array2bbox(bbox)
    proto_box = BBox.bbox2world(inv_ego_motion, proto_box)
    bbox = BBox.bbox2array(proto_box)

    box.center_x, box.center_y, box.center_z, box.heading = bbox[:4]
    box.length, box.width, box.height = bbox[4:7]
    o.object.box.CopyFrom(box)
    o.score = bbox[-1]

    meta_data = label_pb2.Label.Metadata()
    if args.velo:
        meta_data.speed_x, meta_data.speed_y = velo[0], velo[1]
    if args.accel:
        meta_data.accel_x, meta_data.accel_y = accel[0], accel[1]
    o.object.metadata.CopyFrom(meta_data)

    o.object.id = '{:}_{:}'.format(type_token, id)
    o.object.type = type_token
    return o


def create_sequence(pred_ids, pred_bboxes, type_token, context_name, time_stamps, ego_motions, pred_velos, pred_accels):
    frame_num = len(pred_ids)
    sequence_objects = list()
    for frame_index in range(frame_num):
        time_stamp = time_stamps[frame_index]
        frame_obj_num = len(pred_ids[frame_index])
        ego_motion = ego_motions[str(frame_index)]
        inv_ego_motion = np.linalg.inv(ego_motion)
        for obj_index in range(frame_obj_num):
            pred_id = pred_ids[frame_index][obj_index]
            pred_bbox = pred_bboxes[frame_index][obj_index]
            pred_velo, pred_accel = None, None
            if args.velo:
                pred_velo = pred_velos[frame_index][obj_index]
            if args.accel:
                pred_accel = pred_accels[frame_index][obj_index]
            sequence_objects.append(create_single_pred_bbox(
                pred_id, pred_bbox, type_token, time_stamp, context_name, inv_ego_motion, pred_velo, pred_accel))
    return sequence_objects


def merge_results(output_folder, obj_types, output_file_name):
    print('Merging different object types')
    result_objs = list()
    for obj_type in obj_types:
        bin_path = os.path.join(output_folder, obj_type, '{:}.bin'.format(output_file_name))
        f = open(bin_path, 'rb')
        objects = metrics_pb2.Objects()
        objects.ParseFromString(f.read())
        f.close()
        objects = objects.objects
        result_objs += objects
    
    output_objs = metrics_pb2.Objects()
    for obj in result_objs:
        output_objs.objects.append(obj)
    
    output_path = os.path.join(output_folder, '{:}.bin'.format(output_file_name))
    f = open(output_path, 'wb')
    f.write(output_objs.SerializeToString())
    f.close()
        

if __name__ == '__main__':
    result_folder = os.path.join(args.result_folder, args.name)
    output_folder = os.path.join(result_folder, 'bin')
    os.makedirs(output_folder, exist_ok=True)

    obj_types = args.obj_types.split(',')
    for obj_type in obj_types:
        main(args.name, obj_type, result_folder, args.data_folder, output_folder, 'pred')
    
    merge_results(output_folder, obj_types, 'pred')
