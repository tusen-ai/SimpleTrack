""" Extract the time stamp information about each frame
"""
import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import argparse
import json
from google.protobuf.descriptor import FieldDescriptor as FD
tf.enable_eager_execution()
from multiprocessing import Process, Pool

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, default='../../../datasets/waymo/mot_test/',
    help='the location of output information')
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/testing/',
    help='the location that stores the tfrecords')
args = parser.parse_args()


def main(data_folder, out_folder):
    tf_records = os.listdir(data_folder)
    tf_records = [x for x in tf_records if 'tfrecord' in x]
    tf_records = sorted(tf_records)

    for record_index, tf_record_name in enumerate(tf_records):
        print('starting for time stamp: ', record_index + 1, ' / ', len(tf_records), ' ', tf_record_name)
        FILE_NAME = os.path.join(data_folder, tf_record_name)
        dataset = tf.data.TFRecordDataset(FILE_NAME, compression_type='')

        frame_num = 0
        time_stamps = list()
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            time_stamps.append(frame.timestamp_micros)
        
        file_name = tf_record_name.split('.')[0]
        print(file_name)
        f = open(os.path.join(out_folder, '{}.json'.format(file_name)), 'w')
        json.dump(time_stamps, f)
        f.close()


if __name__ == '__main__':
    args.output_folder = os.path.join(args.output_folder, 'ts_info')
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    main(args.data_folder, args.output_folder)