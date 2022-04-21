
import json
import numpy as np
import os

"""
This script Renders provided AMOTA curves, normalizing for one of the AMOTAS.
"""

SAMPLED_RECALLS = {
    'car': [0.90, 0.85, 0.8],
    'pedestrian': [0.92, 0.87, 0.8],
    'bus': [0.89, 0.85, 0.8],
    'motorcycle': [0.87, 0.8, 0.75],
    'trailer': [0.71, 0.65, 0.6],
    'truck': [0.85, 0.80, 0.72],
    'bicycle': [0.85, 0.7, 0.55]
}
NUSC_CLASSES = SAMPLED_RECALLS.keys()
METRIC_KEYS = ['ids', 'fp', 'fn', 'mota']

def load_info(filepath):
    mota_metrics = {}
    with open(filepath, "r") as f:
        data = json.load(f)
        for cls in NUSC_CLASSES:
            recalls = data[cls]['recall']
            cls_eval_recalls = np.array(SAMPLED_RECALLS[cls])
            mota_metrics[cls] = {'recall': cls_eval_recalls.tolist()}
            for k in METRIC_KEYS:
                metric_per_recthresh = data[cls][k]

                interp_metric = np.interp(cls_eval_recalls, np.array(recalls)[::-1], np.array(metric_per_recthresh)[::-1])
                if k in ['ids', 'fp', 'fn']:
                    interp_metric /= data[cls]['gt'][-1]
                mota_metrics[cls][k] = interp_metric.tolist()

    return mota_metrics

def compute_motas(metric_details_path, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    mota_vals = load_info(metric_details_path)
    with open(os.path.join(outdir, "mota_samples.json"), "w") as f:
        json.dump(mota_vals, f)
    print("MOTA info:")
    print(mota_vals)


if __name__ == "__main__":
    compute_motas("/home/colton/Documents/simpletrack-out/cp-mdis_kf/mdis_kf/results/official-eval/metrics_details.json",
                  "/home/colton/Documents/simpletrack-out/cp-mdis_kf/mdis_kf/results/official-eval")
