# Configurations

This documentation provides a brief guide for writing and reading the configuration files. We specify the tracker behaviors with a single `.yaml` file as in the folder of `./configs/`. 

## Explanation with the Example from Waymo Open Dataset

Take the configs for Waymo Open Dataset (vehicles) as example, we explain with `configs/waymo_configs/vc_kf_giou.yaml`.

We provide a thorough annotation below, but it may be confusing if you delve deep directly. So the following are the key points for specifying the model.

* **Data Preprocessing** is controlled by the `nms` and `nms_thres` arguments in `data_loader` field. They specify whether to use NMS and what is the IoU threshold for NMS.

* **Association Metric** is specified in `running-asso`, where you can choose from IoU and GIoU for now. To specify the threshold for successful association, we treat `1-IoU` or `1-GIoU` as the distance, and it must be smaller than the corresponding numbers in `asso_thres`. If you are interested in L2 distance or Mahalanobis distance, please refer to the `dev` branch. We are still cleaning up and adding documentations.
* **Motion Model** is by default Kalman filter. We are adding more options, such as velocity model. Please refer to the `dev` branch if you wish to explore on your own.
* **Two-stage Association** is specified by the `redundancy` module. `mm` denotes two-stage and `default` denotes the conventional single-stage association.
* NOTE on **nuScenes 10Hz two-stage association**. We are trying to incorporate this part into the config file instead of hard coding everything. Please temporarily go to the function `non_key_frame_mot` in `mot_3d/mot.py` for how we deal with the tracking on no-key frames.


```yaml
running:
  covariance: default         # not used
  score_threshold: 0.7        # detection score threshold for first-stage association and output
  max_age_since_update: 2     # count for death, same as AB3DMOT
  min_hits_to_birth: 3        # count for birth, same as AB3DMOT
  match_type: bipartite       # use hungarian (biparitite) or greedy algorithm (greedy) for association
  asso: giou                  # association metric, we support GIoU (giou) and IoU (iou) for now
  has_velo: false
  motion_model: kf            # Kalman filter (kf) as motion model
  asso_thres:
    iou: 0.9                  # association threshold, 1 - IoU has to be smaller than it.
    giou: 1.5                 # association threshold, 1 - GIoU has to be smaller than it.

redundancy:
  mode: mm                    # (mm) denotes two-stage association, (default) denotes one stage association (see nuScenes configs)
  det_score_threshold:        # detection score threshold for two-stage association
    iou: 0.1
    giou: 0.1
  det_dist_threshold:         # association threshold for two-stage association
    iou: 0.1                  # IoU has to be greater than this
    giou: -0.5                # GIoU has to be greater than this

data_loader:
  pc: true                    # load point clouds for visualization
  nms: true                   # apply NMS for data preprocessing, True or False
  nms_thres: 0.25             # IoU-3D threshold for NMS
```