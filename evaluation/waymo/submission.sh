name=$1
echo "Processing EXP ${name}"
cd /mnt/truenas/scratch/ziqi.pang/waymo-od/bazel-bin/waymo_open_dataset/
submission_path="/mnt/truenas/scratch/ziqi.pang/mot_results/${name}/submission/"
if [ ! -e "${submission_path}" ]; then
    mkdir "${submission_path}"
fi

metrics/tools/create_submission \
    --input_filenames="/mnt/truenas/scratch/ziqi.pang/mot_results/${name}/bin/pred.bin" \
    --output_filename="${submission_path}" \
    --submission_filename='/mnt/truenas/scratch/ziqi.pang/mot_3d/evaluation/waymo/submission.txtpb'

cd "/mnt/truenas/scratch/ziqi.pang/mot_results/${name}/"

tar cvf "${name}.tar" "./submission/"
gzip "${name}.tar"