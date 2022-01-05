raw_data_dir=$1
data_dir=$2
proc_num=$3

# ego pose
python ego_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir --process $proc_num

# # time stamp
python time_stamp.py --raw_data_folder $raw_data_dir --data_folder $data_dir --process $proc_num

# point cloud, useful for visualization
python raw_pc.py --raw_data_folder $raw_data_dir --data_folder $data_dir --process $proc_num