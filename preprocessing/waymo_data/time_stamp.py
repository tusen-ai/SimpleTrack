""" Extract the time stamp information about each frame.
    Each sequence has a json file containing a list of timestamps.
"""
import os
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import json
from google.protobuf.descriptor import FieldDescriptor as FD
tf.enable_eager_execution()
import multiprocessing
from waymo_open_dataset import dataset_pb2 as open_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='../../../datasets/waymo/validation/',
    help='location of tfrecords')
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/mot/',
    help='output folder')
parser.add_argument('--process', type=int, default=1, help='use multiprocessing for acceleration')
args = parser.parse_args()


def main(raw_data_folder, data_folder, process, token):
    tf_records = os.listdir(raw_data_folder)
    tf_records = [x for x in tf_records if 'tfrecord' in x]
    tf_records = sorted(tf_records)

    for record_index, tf_record_name in enumerate(tf_records):
        if record_index % process != token:
            continue
        print('starting for time stamp: ', record_index + 1, ' / ', len(tf_records), ' ', tf_record_name)
        FILE_NAME = os.path.join(raw_data_folder, tf_record_name)
        dataset = tf.data.TFRecordDataset(FILE_NAME, compression_type='')

        time_stamps = list()
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            time_stamps.append(frame.timestamp_micros)
        
        file_name = tf_record_name.split('.')[0]
        print(file_name)
        f = open(os.path.join(data_folder, '{:}.json'.format(file_name)), 'w')
        json.dump(time_stamps, f)
        f.close()


if __name__ == '__main__':
    args.data_folder = os.path.join(args.data_folder, 'ts_info')
    os.makedirs(args.data_folder, exist_ok=True)
    
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.raw_data_folder, args.data_folder, args.process, token))
        pool.close()
        pool.join()
    else:
        main(args.raw_data_folder, args.data_folder)
