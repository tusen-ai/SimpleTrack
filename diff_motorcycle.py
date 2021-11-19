import numpy as np, os
from mot_3d.data_protos import BBox
import pdb
from tqdm import tqdm

name0 = 'debug'
name1 = 'bip_two_stage_0.5_0.1'

folder = '/mnt/truenas/scratch/ziqi.pang/10hz_exps/'
folder0 = os.path.join(folder, name0, 'summary', 'motorcycle')
folder1 = os.path.join(folder, name1, 'summary', 'motorcycle')

file_names = sorted(os.listdir(folder0))
pbar = tqdm(total=len(file_names))
for file_index, file_name in enumerate(file_names[:]):
    file0 = os.path.join(folder0, file_name)
    file1 = os.path.join(folder1, file_name)

    data0 = np.load(file0, allow_pickle=True)
    data1 = np.load(file1, allow_pickle=True)

    bboxes0, ids0, states0 = data0['bboxes'], data0['ids'], data0['states']
    bboxes1, ids1, states1 = data1['bboxes'], data1['ids'], data1['states']

    frame_num = len(ids0)
    for frame_index in range(frame_num):
        if ids0[frame_index] != ids1[frame_index] or states0[frame_index] != states1[frame_index]:
            print(frame_index)
            pdb.set_trace()
            k = 1
    pbar.update(1)
pbar.close()
