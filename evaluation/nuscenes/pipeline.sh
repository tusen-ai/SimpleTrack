name=$1
python ./evaluation/nuscenes/result_creation.py --name "${name}"
python ./evaluation/nuscenes/type_merge.py --name "${name}"
bash ./evaluation/nuscenes/eval.sh "${name}"