name=$1
python ./evaluation/nuscenes/result_creation_20hz.py --name "${name}"
python ./evaluation/nuscenes/type_merge_10hz.py --name "${name}"
bash ./evaluation/nuscenes/eval_10hz.sh "${name}"