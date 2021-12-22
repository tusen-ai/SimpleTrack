# SimpleTrack: Simple yet Effective 3D Multi-object Tracking

This is the repository for our paper [SimpleTrack: Understanding and Rethinking 3D Multi-object Tracking](https://arxiv.org/abs/2111.09621).

If you find our paper or code useful for you, please consider cite us by:
```
@article{pang2021simpletrack,
  title={SimpleTrack: Understanding and Rethinking 3D Multi-object Tracking},
  author={Pang, Ziqi and Li, Zhichao and Wang, Naiyan},
  journal={arXiv preprint arXiv:2111.09621},
  year={2021}
}
```

## Experiments on Waymo Open Dataset and nuScenes

### Data Preprocessing

We provide instructions for Waymo Open Dataset and nuScenes. Please follow the command at [Data Preprocessing Documentations](./docs/data_preproces.md). 

After data preprocessing, the directories to the preprocessed data are `waymo_data_dir` for Waymo Open Dataset, `nuscenes2hz_data_dir` and `nuscenes20hz_data_dir` for the key-frame and all-frame data of nuScenes.

### Waymo Open Dataset

To run for a standard SimpleTrack setting experiment, run the following commands, the per-sequence results are then saved in the `${waymo_result_dir}/SimpleTrack/summary`, with subfolder taking the name `vehicle/`, `pedestrian/`, and `cyclist/`.

In the experiments, 
* `${det_name}` denotes the name of the detection preprocessed; 
* `${waymo_result_dir}` is for saving tracking results;
* `${waymo_data_dir}` is the folder to the preprocessed waymo data;
* `${proc_num}` uses multiprocessing to inference different sequences (for your information, I generally use 202 processes, so that each one of the process runs a sequence of the validation set).

```bash
# for vehicle
python tools/main_waymo.py --name SimpleTrack --det_name ${det_name} --obj_type vehicle --config_path configs/waymo_configs/vc_kf_giou.yaml --data_folder ${waymo_data_dir} --result_folder ${waymo_result_dir} --process ${proc_num}

# for pedestrian
python tools/main_waymo.py --name SimpleTrack --det_name ${det_name} --obj_type vehicle --config_path configs/waymo_configs/vc_kf_giou.yaml --data_folder ${waymo_data_dir} --result_folder ${waymo_result_dir} --process ${proc_num}

# for cyclist
python tools/main_waymo.py --name SimpleTrack --det_name ${det_name} --obj_type vehicle --config_path configs/waymo_configs/vc_kf_giou.yaml --data_folder ${waymo_data_dir} --result_folder ${waymo_result_dir} --process ${proc_num}
```

To work with the official formats of Waymo Open Dataset, the following commands convert the results in SimpleTrack format to the `.bin` format.

After running the command, we will have four `pred.bin` files in `${waymo_result_dir}/SimpleTrack/bin/` as `prd.bin` (all objects), `vehicle/pred.bin` (vehicles only), `pedestrain/pred.bin` (pedestrian only), and `cyclist/pred.bin` (cyclist only).

```bash
python tools/waymo_pred_bin.py --name SimpleTrack --result_folder ${waymo_result_dir} --data_folder ${waymo_data_dir}
```

Eventually, use the evaluation provided by Waymo officials by following [Quick Guide to Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).