import numpy as np, os, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.visualization import Visualizer2D
from contrast.contrast import frame_contrast


def corres_location(instance_masks, pred_ids, multi_seq=True):
    """ for a list of labels,  compute the corresponding sequence, frame, and id
    """
    result_seq_ids = list()
    result_frame_ids = list()
    result_ids = list()
    for seq_id, instance_mask in enumerate(instance_masks):
        for frame_id, frame_mask in enumerate(instance_mask):
            result_seq_ids += [seq_id for i in range(len(frame_mask))]
            result_frame_ids += [frame_id for i in range(len(frame_mask))]
            for instance_idx, instance_location in enumerate(frame_mask):
                result_ids += pred_ids[seq_id][frame_id][instance_location]
    return result_seq_ids, result_frame_ids, result_ids


def gt_bbox2world(bboxes, egos):
    frame_num = len(egos)
    for i in range(frame_num):
        ego = egos[i]
        bbox_num = len(bboxes[i])
        for j in range(bbox_num):
            bboxes[i][j] = BBox.bbox2world(ego, bboxes[i][j])
    return bboxes


def pc2world(pcs, egos):
    result = list()
    for i, (pc, ego) in enumerate(zip(pcs, egos)):
        result.append(utils.pc2world(ego, pc))
    return result


def load_dets(det_data_folder, data_folder, segment_name):
    dets_info = np.load(os.path.join(det_data_folder, 'dets', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    dets_bboxes = dets_info['bboxes']
    inst_types = dets_info['types']

    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_keys = sorted(utils.str2int(ego_info.keys()))
    egos = [ego_info[str(key)] for key in ego_keys]
    
    det_bboxes = [dets_bboxes[key] for key in ego_keys]
    inst_types = [inst_types[key] for key in ego_keys]
    dets = [[] for i in range(len(dets_bboxes))]
    for i in range(len(det_bboxes)):
        for j in range(len(det_bboxes[i])):
            dets[i].append(BBox.array2bbox(dets_bboxes[i][j]))
    return dets, inst_types


class Displayer:
    def __init__(self, namea, nameb, obj_type, seq_id, data_folder, result_folder, det_data_folder, gt_folder):
        self.namea = namea
        self.nameb = nameb
        self.seq_id = seq_id
        self.data_folder = data_folder
        self.result_folder = result_folder
        self.gt_folder = gt_folder
        self.obj_type = obj_type
        if obj_type == 'vehicle':
            self.type_token = 1
        elif obj_type == 'pedestrian':
            self.type_token = 2
        elif obj_type == 'cyclist':
            self.type_token = 4

        # raw data
        self.ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(seq_id)), 
            allow_pickle=True)
        self.pcs = np.load(os.path.join(data_folder, 'pc', 'clean_pc', '{:}.npz'.format(seq_id)),
            allow_pickle=True)
        
        gt_info = np.load(os.path.join(gt_folder, '{:}.npz'.format(seq_id)), 
            allow_pickle=True)
        bboxes, ids, inst_types = gt_info['bboxes'], gt_info['ids'], gt_info['types']
        self.gt_ids, self.gt_bboxes = utils.inst_filter(ids, bboxes, inst_types, type_field=[self.type_token], id_trans=True)
        ego_keys = sorted(utils.str2int(self.ego_info.keys()))
        egos = [self.ego_info[str(key)] for key in ego_keys]
        self.pcs = [self.pcs[str(key)] for key in ego_keys]
        self.pcs = pc2world(self.pcs, egos)
        self.gt_bboxes = gt_bbox2world(self.gt_bboxes, egos)

        self.dets, dets_types = load_dets(det_data_folder, data_folder, seq_id)
        for i in range(len(self.dets)):
            det_num = len(self.dets[i])
            self.dets[i] = [self.dets[i][j] for j in range(det_num) if dets_types[i][j] == self.type_token]

        # tracking results
        pred_resulta = np.load(os.path.join(result_folder, self.namea, 'summary', self.obj_type, '{:}.npz'.format(self.seq_id)), allow_pickle=True)
        pred_resultb = np.load(os.path.join(result_folder, self.nameb, 'summary', self.obj_type, '{:}.npz'.format(self.seq_id)), allow_pickle=True)
        self.pred_idsa, self.pred_idsb = pred_resulta['ids'], pred_resultb['ids']
        self.pred_bboxesa, self.pred_bboxesb = pred_resulta['bboxes'], pred_resultb['bboxes']
        self.pred_statesa, self.pred_statesb = pred_resulta['states'], pred_resultb['states']
        for i in range(len(self.pred_bboxesa)):
            for j in range(len(self.pred_bboxesa[i])):
                self.pred_bboxesa[i][j] = BBox.array2bbox(self.pred_bboxesa[i][j])
        for i in range(len(self.pred_bboxesb)):
            for j in range(len(self.pred_bboxesb[i])):
                self.pred_bboxesb[i][j] = BBox.array2bbox(self.pred_bboxesb[i][j])
        return
    
    def display(self, frame_id, instance_id, interval):
        return
    
    def single_frame_display(self, frame_id):
        print('|||||||||| FRAME {:} ||||||||||'.format(frame_id))
        vis = Visualizer2D(name='{:}'.format(frame_id), figsize=(12, 12))
        vis.handler_pc(self.pcs[frame_id])

        frame_gts = self.gt_bboxes[frame_id]
        frame_pd_bboxesa, frame_pd_bboxesb = self.pred_bboxesa[frame_id], self.pred_bboxesb[frame_id]
        frame_pd_idsa, frame_pd_idsb = self.pred_idsa[frame_id], self.pred_idsb[frame_id]
        frame_pd_statesa, frame_pd_statesb = self.pred_statesa[frame_id], self.pred_statesb[frame_id]
        
        # visualize the gt
        for _, bbox in enumerate(self.gt_bboxes[frame_id]):
            vis.handler_box(bbox, message='', color='black')

        # visualize the bboxes
        normala, normalb, diffa, diffb, iousa, iousb = frame_contrast(frame_pd_bboxesa, frame_pd_bboxesb, frame_gts)
        diff_info_a, diff_info_b = list(), list()
        
        for idx in normala:
            bbox, state, id, iou = frame_pd_bboxesa[idx], frame_pd_statesa[idx], frame_pd_idsa[idx], iousa[idx]
            if Validity.valid(state):
                vis.handler_box(bbox, message='%s %.2f' % (id, iou), color='green', linestyle='solid')
            else:
                vis.handler_box(bbox, message='%s %.2f' % (id, iou), color='green', linestyle='dashed')
        
        for idx in normalb:
            bbox, state, id, iou = frame_pd_bboxesb[idx], frame_pd_statesb[idx], frame_pd_idsb[idx], iousb[idx]
            if Validity.valid(state):
                vis.handler_box(bbox, message='%s %.2f' % (id, iou), color='purple', linestyle='solid')
            else:
                vis.handler_box(bbox, message='%s %.2f' % (id, iou), color='purple', linestyle='dashed')
        
        for idx in diffa:
            bbox, state, id, iou = frame_pd_bboxesa[idx], frame_pd_statesa[idx], frame_pd_idsa[idx], iousa[idx]
            if Validity.valid(state):
                vis.handler_box(bbox, message='A: %s %.2f' % (id, iou), color='light_blue', linestyle='solid')
                diff_info_a.append((id, '%.2f' % iou))
            else:
                vis.handler_box(bbox, message='A: %s %.2f' % (id, iou), color='light_blue', linestyle='dashed')
        
        for idx in diffb:
            bbox, state, id, iou = frame_pd_bboxesb[idx], frame_pd_statesb[idx], frame_pd_idsb[idx], iousb[idx]
            if Validity.valid(state):
                vis.handler_box(bbox, message='B: %s %.2f' % (id, iou), color='red', linestyle='solid')
                diff_info_b.append((id, '%.2f' % iou))
            else:
                vis.handler_box(bbox, message='B: %s %.2f' % (id, iou), color='red', linestyle='dashed')
        print(diff_info_a)
        print(diff_info_b)
        print('__________')
        print('\n')

        vis.show()
        vis.close()
        return