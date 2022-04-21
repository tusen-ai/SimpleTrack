import os, argparse, json, numpy as np
from pyquaternion import Quaternion
from mot_3d.data_protos import BBox, Validity


def main(name, obj_types, result_folder):
    raw_results = list()
    for type_name in obj_types:
        path = os.path.join(result_folder, type_name, 'results.json')
        f = open(path, 'r')
        raw_results.append(json.load(f)['results'])
        f.close()
    
    results = raw_results[0]
    sample_tokens = list(results.keys())
    for token in sample_tokens:
        for i in range(1, len(obj_types)):
            results[token] += raw_results[i][token]
    
    submission_file = {
        'meta': {
            'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False
        },
        'results': results
    }

    f = open(os.path.join(result_folder, 'results.json'), 'w')
    json.dump(submission_file, f)
    f.close()
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--obj_types', default='car,bus,trailer,truck,pedestrian,bicycle,motorcycle')
    parser.add_argument('--result_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/nu_mot_results/')
    args = parser.parse_args()

    result_folder = os.path.join(args.result_folder, args.name, 'results')
    obj_types = args.obj_types.split(',')
    main(args.name, obj_types, result_folder)