# Data Preprocessing

## Waymo Open Dataset

We decompose the preprocessing of Waymo Open Dataset into the following steps.

### 1. Raw Data

This step converts the information needed from `tf_records` into more handy forms. Suppose the folder storing `tf_records` is `raw_data_folder`, the target location if `data_folder`, run the following command: (you can specify `process_num` to be an integer greater than 1 for faster preprocessing.)

```bash
cd preprocess/waymo_data
bash waymo_preprocess.sh raw_data_folder data_folder process_num
```

### 2. Ground Truth Information

The gorund truth for the 3D MOT and 3D Detection are the same. **You have to download a .bin file from Waymo Open Dataset for the ground truth, which we have no right to share according to the license.** 

To decode the ground truth information, suppose `bin_path` is the path to the ground truth file, `data_folder` is the target location of data preprocess. Eventually, we store the ground truth information in `${data_folder}/detection/gt/dets/`.

```bash
cd preprocess/waymo_data
python gt_bin_decode.py --data_folder data_dir --file_path bin_path
```

### 3. Detection

To infer 3D MOT on your detection file, we still need the `bin_path` indicating the path to the detection results, then name your detection as `name` for future convenience. The preprocessing of the detection follows the below scripts. (Only use `metadata` if you want to save the velocity / acceleration contained in the detection file.)

```bash
cd preprocess/waymo_data
python detection --name name --data_folder data_dir --file_path bin_path --metadata
```

## nuScenes

### 1. Preprocessing

To preprocessing the raw data from nuScenes, suppose you have put the raw data of nuScenes at `raw_data_dir`. We provide two modes of proprocessing:
* Only the data on the key frames (2Hz) is extracted, the target location is `data_dir_2hz`.
* All the data (20Hz) is extracted to the location of `data_dir_20hz`.

Run the following commands. (`proc_num` is used for accelerating the speed with multiple processes.)

```bash
cd preprocess/nuscenes_data
bash nuscenes_preprocess.sh raw_data_dir data_dir_2hz data_dir_20hz proc_num
```

### 2. Detection

To infer 3D MOT on your detection file, we convert the json format detection files at `file_path` into the .npz files similar to our approach on Waymo Open Dataset. Please name your detection as `name` for future convenience. The preprocessing of the detection follows the below scripts. (Only use `velo` if you want to save the velocity contained in the detection file.)

```bash
cd preprocess/nuscenes_data

# for 2Hz detection file
python detection.py --raw_data_folder raw_data_dir --data_folder data_dir_2hz --det_name name --file_path file_path --mode 2hz --velo

# for 20Hz detection file
python detection.py --raw_data_folder raw_data_dir --data_folder data_dir_20hz --det_name name --file_path file_path --mode 20hz --velo
```