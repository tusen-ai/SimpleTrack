""" Extract the point cloud sequences from the tfrecords
    output format: a compressed dict stored in an npz file
    {
        str(frame_number): pc (N * 3 numpy array)
    }
"""
import argparse
import math
import numpy as np
import json
import os
import sys
import multiprocessing
from google.protobuf.descriptor import FieldDescriptor as FD
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/testing/',
    help='the location of tfrecords')
parser.add_argument('--output_folder', type=str, default='../../../datasets/waymo/mot_test/',
    help='the location of raw pcs')
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()
args.output_folder = os.path.join(args.output_folder, 'pc', 'raw_pc')
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


def pb2dict(obj):
    """
    Takes a ProtoBuf Message obj and convertes it to a dict.
    """
    adict = {}
    # if not obj.IsInitialized():
    #     return None
    for field in obj.DESCRIPTOR.fields:
        if not getattr(obj, field.name):
            continue
        if not field.label == FD.LABEL_REPEATED:
            if not field.type == FD.TYPE_MESSAGE:
                adict[field.name] = getattr(obj, field.name)
            else:
                value = pb2dict(getattr(obj, field.name))
                if value:
                    adict[field.name] = value
        else:
            if field.type == FD.TYPE_MESSAGE:
                adict[field.name] = [pb2dict(v) for v in getattr(obj, field.name)]
            else:
                adict[field.name] = [v for v in getattr(obj, field.name)]
    return adict


def main(data_folder, output_folder, multi_process_token=(0, 1)):
    tf_records = os.listdir(data_folder)
    tf_records = [x for x in tf_records if 'tfrecord' in x]
    tf_records = sorted(tf_records) 
    for record_index, tf_record_name in enumerate(tf_records):
        if record_index % multi_process_token[1] != multi_process_token[0]:
            continue
        print('starting for raw pc', record_index + 1, ' / ', len(tf_records), ' ', tf_record_name)
        FILE_NAME = os.path.join(data_folder, tf_record_name)
        dataset = tf.data.TFRecordDataset(FILE_NAME, compression_type='')
        segment_name = tf_record_name.split('.')[0]

        # if os.path.exists(os.path.join(output_folder, '{}.npz'.format(segment_name))):
        #     continue

        frame_num = 0
        pcs = dict()

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
     
            # extract the points
            (range_images, camera_projections, range_image_top_pose) = \
                frame_utils.parse_range_image_and_camera_projection(frame)
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose, ri_index=0)
            points_all = np.concatenate(points, axis=0)
            pcs[str(frame_num)] = points_all

            frame_num += 1
            if frame_num % 10 == 0:
                print('Record {} / {} FNumber {:}'.format(record_index + 1, len(tf_records), frame_num))
        print('{:} frames in total'.format(frame_num))

        np.savez_compressed(os.path.join(output_folder, "{}.npz".format(segment_name)), **pcs)


if __name__ == '__main__':
    # multiprocessing accelerate the speed
    pool = multiprocessing.Pool(args.process)
    for token in range(args.process):
        result = pool.apply_async(main, args=(args.data_folder, args.output_folder, (token, args.process)))
    pool.close()
    pool.join()
