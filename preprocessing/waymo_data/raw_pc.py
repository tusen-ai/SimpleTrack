""" Extract the point cloud sequences from the tfrecords
    output format: a compressed dict stored in an npz file
    {
        str(frame_number): pc (N * 3 numpy array)
    }
"""
import argparse
import numpy as np
import os
import multiprocessing
from google.protobuf.descriptor import FieldDescriptor as FD
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='../../../datasets/waymo/validation/',
    help='the location of tfrecords')
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/mot/',
    help='the location of raw pcs')
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()


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


def main(raw_data_folder, data_folder, process=1, token=0):
    tf_records = os.listdir(raw_data_folder)
    tf_records = [x for x in tf_records if 'tfrecord' in x]
    tf_records = sorted(tf_records) 
    for record_index, tf_record_name in enumerate(tf_records):
        if record_index % process != token:
            continue
        print('starting for raw pc', record_index + 1, ' / ', len(tf_records), ' ', tf_record_name)
        FILE_NAME = os.path.join(raw_data_folder, tf_record_name)
        dataset = tf.data.TFRecordDataset(FILE_NAME, compression_type='')
        segment_name = tf_record_name.split('.')[0]

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
                print('Point Cloud Record {} / {} FNumber {:}'.format(record_index + 1, len(tf_records), frame_num))
        print('{:} frames in total'.format(frame_num))

        np.savez_compressed(os.path.join(data_folder, "{}.npz".format(segment_name)), **pcs)


if __name__ == '__main__':
    args.data_folder = os.path.join(args.data_folder, 'pc', 'raw_pc')
    os.makedirs(args.data_folder, exist_ok=True)
    
    if args.process > 1:
    # multiprocessing accelerate the speed
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.raw_data_folder, args.data_folder, args.process, token))
        pool.close()
        pool.join()
    else:
        main(args.raw_data_folder, args.data_folder)
