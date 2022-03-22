# nuScenes Inference

## Data Preprocessing

Please follow the command at [Data Preprocessing Documentations](./data_preprocess.md). After data preprocessing, the directories to the preprocessed data are `nuscenes2hz_data_dir` and `nuscenes20hz_data_dir` for the key-frame and all-frame data of nuScenes.

## Inference with SimpleTrack

**Important: Please strictly follow the config file we use.**

For the setting of inference only on key frames (2Hz), run the following command. The per-sequence results are then saved in the `${nuscenes_result_dir}/SimpleTrack2Hz/summary`, with subfolders of the types of the objects containing the results for each type.

```bash
python tools/main_nuscenes.py \
    --name SimpleTrack2Hz \
    --det_name ${det_name} \
    --config_path configs/nu_configs/giou.yaml \
    --result_folder ${nuscenes_result_dir} \
    --data_folder ${nuscenes2hz_data_dir} \
    --process ${proc_num}
```

For the 10Hz settings proposed in the paper, run the following commands. The per-sequence results are then saved in the `${nuscenes_result_dir}/SimpleTrack20Hz/summary`.

```bash
python tools/main_nuscenes_10hz.py \
    --name SimpleTrack10Hz \
    --det_name ${det_name} \
    --config_path configs/nu_configs/giou.yaml \
    --result_folder ${nuscenes_result_dir} \
    --data_folder ${nuscenes20hz_data_dir} \
    --process ${proc_num}
```

I use the process number of 150 in my experiments, which is the same as the number of sequences in nuScenes validation set.

## Output Format

In the folder of `${nuscenes_result_dir}/SimpleTrack/summary/`, there are sub-folders corresponding to each object type in nuScenes. Inside each sub-folder, there are 150 `.npz` files, matching the 150 sequences in the nuScenes validation set. For the format in each `.npz` file, please refer to [Output Format](output_format.md).

## Converting to nuScenes Format

Use the following command to convert the output results in the SimpleTrack format into the `.json` format specified by the nuScenes officials. After running the following commands, there will the tracking results in `.json` formats in `${nuscenes_result_dir}/SimpleTrack2Hz/results` and `${nuscenes_result_dir}/SimpleTrack10Hz/results`.

For the setting of 2Hz, which only inferences on the key frames, run the following commands.

```bash
python tools/nuscenes_result_creation.py \
    --name SimpleTrack2Hz \
    --result_folder ${nuscenes_result_dir} \
    --data_folder ${nuscenes2hz_data_dir}

python tools/nuscenes_type_merge.py \
    --name SimpleTrack2Hz \
    --result_folder ${nuscenes_result_dir}
```

For the setting of 10Hz, run the following commands.

```bash
python tools/nuscenes_result_creation_10hz.py \
    --name SimpleTrack10Hz \
    --result_folder ${nuscenes_result_dir} \
    --data_folder ${nuscenes20hz_data_dir}

python tools/nuscenes_type_merge.py \
    --name SimpleTrack10Hz \
    --result_folder ${nuscenes_result_dir}
```

## Files and Detailed Metrics

Please see [Dropbox Link](https://www.dropbox.com/sh/8906exnes0u5e89/AAD0xLwW1nq_QiuUBaYDrQVna?dl=0).