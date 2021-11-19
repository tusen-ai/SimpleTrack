import numpy as np, os, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.visualization import Visualizer2D
import motmetrics as mm


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


def compute_iou_matrix(gt_bboxes, pd_bboxes, thres=0.3):
    iou_matrix = np.empty((len(gt_bboxes), len(pd_bboxes)))
    for i, gt in enumerate(gt_bboxes):
        for j, pd in enumerate(pd_bboxes):
            iou_matrix[i, j] = utils.iou3d(gt, pd)[1]
    iou_matrix = 1 - iou_matrix
    iou_matrix = np.where(iou_matrix > thres, np.nan, iou_matrix)
    return iou_matrix


class Displayer:
    def __init__(self, name, obj_type, seq_id, data_folder, result_folder, det_data_folder, gt_folder,
        thres=0.3, start_frame=0):
        self.name = name
        self.seq_id = seq_id
        self.seq_id = seq_id
        self.data_folder = data_folder
        self.result_folder = result_folder
        self.gt_folder = gt_folder
        self.obj_type = obj_type
        self.start_frame = start_frame
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
        pred_result= np.load(os.path.join(result_folder, self.name, 'summary', self.obj_type, '{:}.npz'.format(self.seq_id)), allow_pickle=True)
        self.pred_ids = pred_result['ids']
        self.pred_bboxes, self.pred_states = pred_result['bboxes'], pred_result['states']
        for i in range(len(self.pred_bboxes)):
            for j in range(len(self.pred_bboxes[i])):
                self.pred_bboxes[i][j] = BBox.array2bbox(self.pred_bboxes[i][j])
        
        # tracking buffer
        self.id_switch_frame = start_frame - 1
        self.matching_thres = thres
        self.mot_acc = mm.MOTAccumulator(auto_id=True)
        return
    
    def frame_motmetrics(self, gt_ids, gt_bboxes, pd_ids, pd_bboxes, thres):
        dist_matrix = compute_iou_matrix(gt_bboxes, pd_bboxes, thres=thres)
        self.mot_acc.update(
            gt_ids,
            pd_ids,
            dist_matrix
        )
        return
    
    def single_frame_display(self, frame_id):
        print('|||||||||| FRAME {:} ||||||||||'.format(frame_id))
        vis = Visualizer2D(name='{:}'.format(frame_id), figsize=(12, 12))
        vis.handler_pc(self.pcs[frame_id])

        frame_gt_bboxes, frame_gt_ids = self.gt_bboxes[frame_id], self.gt_ids[frame_id]
        frame_pd_bboxes, frame_pd_ids, frame_pd_states = \
            self.pred_bboxes[frame_id], self.pred_ids[frame_id], self.pred_states[frame_id]
        valid_indexes = [i for i, state in enumerate(frame_pd_states) if Validity.valid(state)]
        valid_bboxes, valid_ids = [frame_pd_bboxes[i] for i in valid_indexes], [frame_pd_ids[i] for i in valid_indexes]
    
        # visualize the gt
        for _, (bbox, id) in enumerate(zip(self.gt_bboxes[frame_id], self.gt_ids[frame_id])):
            vis.handler_box(bbox, message=str(id), color='black')
        
        for _, (bbox, id, state) in enumerate(zip(frame_pd_bboxes, frame_pd_ids, frame_pd_states)):
            if Validity.valid(state):
                vis.handler_box(bbox, message=str(id), color='red')
            else:
                vis.handler_box(bbox, message=str(id), color='light_blue')
        
        # update the mot accumulator
        if frame_id > self.id_switch_frame:
            self.id_switch_frame = frame_id
            self.frame_motmetrics(frame_gt_ids, frame_gt_bboxes, valid_ids, valid_bboxes, 
                self.matching_thres)

        try:
            frame_mot_events = self.mot_acc.mot_events.loc[frame_id - self.start_frame]
            id_switch_rows = frame_mot_events.loc[frame_mot_events['Type'] == 'SWITCH']
            # print(id_switch_rows)
            ids = id_switch_rows['HId']
            ids = [int(id) for id in ids]
            result_ids = []
            for id in ids:
                if id in self.pred_ids[frame_id - 1]:
                    result_ids += [id]
            if len(result_ids) > 0:
                print(frame_id, result_ids)
        except:
            pass

        # vis.show()
        vis.save('./imgs/euler/{:}.png'.format(frame_id))
        vis.close()
        return
        