raw_data_dir=$1
data_dir_2hz=$2
data_dir_20hz=$3
proc_num=$4
proc_num=$(($proc_num))

# token information
python token_info.py --raw_data_folder raw_data_dir --data_folder data_dir_2hz --mode 2hz
python token_info.py --raw_data_folder raw_data_dir --data_folder data_dir_20hz --mode 20hz

# time stamp information
python time_stamp.py --raw_data_folder raw_data_dir --data_folder data_dir_2hz --mode 2hz
python time_stamp.py --raw_data_folder raw_data_dir --data_folder data_dir_20hz --mode 20hz

# sensor calibration information
python sensor_calibration.py --raw_data_folder raw_data_dir --data_folder data_dir_2hz --mode 2hz
python sensor_calibration.py --raw_data_folder raw_data_dir --data_folder data_dir_20hz --mode 20hz

# ego pose
python ego_pose.py --raw_data_folder raw_data_dir --data_folder data_dir_2hz --mode 2hz
python ego_pose.py --raw_data_folder raw_data_dir --data_folder data_dir_20hz --mode 20hz

# ego pose
python gt_info.py --raw_data_folder raw_data_dir --data_folder data_dir_2hz --mode 2hz
python gt_info.py --raw_data_folder raw_data_dir --data_folder data_dir_20hz --mode 20hz

# point cloud, useful for visualization
python ego_pose.py --raw_data_folder raw_data_dir --data_folder data_dir_2hz --mode 2hz --process proc_num
python ego_pose.py --raw_data_folder raw_data_dir --data_folder data_dir_20hz --mode 20hz --process proc_num

