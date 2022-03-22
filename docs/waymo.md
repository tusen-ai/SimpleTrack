# Waymo Open Dataset Inference

This documentation describes the steps to infer on the validation set of Waymo Open Dataset.

## Data Preprocessing

We provide instructions for Waymo Open Dataset and nuScenes. Please follow the command at [Data Preprocessing Documentations](data_preprocess.md). After data preprocessing, the directories to the preprocessed data are `waymo_data_dir`, the name for the detection is `det_name`.

## Inference with SimpleTrack

To run for a standard SimpleTrack setting experiment, run the following commands, the per-sequence results are then saved in the `${waymo_result_dir}/SimpleTrack/summary`, with subfolder taking the name `vehicle/`, `pedestrian/`, and `cyclist/`.

In the experiments, 
* `${det_name}` denotes the name of the detection preprocessed; 
* `${waymo_result_dir}` is for saving tracking results;
* `${waymo_data_dir}` is the folder to the preprocessed waymo data;
* `${proc_num}` uses multiprocessing to inference different sequences (for your information, I generally use 202 processes, so that each one of the process runs a sequence of the validation set).
* **Please look out for the different config files we use for each type of object.** 

```bash
# for vehicle
python tools/main_waymo.py \
    --name SimpleTrack \
    --det_name ${det_name} \
    --obj_type vehicle \
    --config_path configs/waymo_configs/vc_kf_giou.yaml \
    --data_folder ${waymo_data_dir} \
    --result_folder ${waymo_result_dir} \
    --gt_folder ${gt_dets_dir} \
    --process ${proc_num}

# for pedestrian
python tools/main_waymo.py \
    --name SimpleTrack \
    --det_name ${det_name} \
    --obj_type vehicle \
    --config_path configs/waymo_configs/pd_kf_giou.yaml \
    --data_folder ${waymo_data_dir} \
    --result_folder ${waymo_result_dir} \
    --process ${proc_num}

# for cyclist
python tools/main_waymo.py \
    --name SimpleTrack \
    --det_name ${det_name} \
    --obj_type vehicle \
    --config_path configs/waymo_configs/vc_kf_giou.yaml \
    --data_folder ${waymo_data_dir} \
    --result_folder ${waymo_result_dir} 
    --process ${proc_num}
```

## Output Format

In the folder of `${waymo_result_dir}/SimpleTrack/summary/`, there are three sub-folders `vehicle`, `pedestrian`, and `cyclist`. Each of them contains 202 `.npz` files, corresponding to the sequences in the validation set of Waymo Open Dataset. For the format in each `.npz` file, please refer to [Output Format](output_format.md).

## Converting to Waymo Open Dataset Format

To work with the official formats of Waymo Open Dataset, the following commands convert the results in SimpleTrack format to the `.bin` format.

After running the command, we will have four `pred.bin` files in `${waymo_result_dir}/SimpleTrack/bin/` as `prd.bin` (all objects), `vehicle/pred.bin` (vehicles only), `pedestrain/pred.bin` (pedestrian only), and `cyclist/pred.bin` (cyclist only).

```bash
python tools/waymo_pred_bin.py \
    --name SimpleTrack \
    --result_folder ${waymo_result_dir} \
    --data_folder ${waymo_data_dir}
```

Eventually, use the evaluation provided by Waymo officials by following [Quick Guide to Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).

## Files and Detailed Metrics

Please see [Dropbox link](https://www.dropbox.com/sh/u6o8dcwmzya04uk/AAAUsNvTt7ubXul9dx5Xnp4xa?dl=0).
