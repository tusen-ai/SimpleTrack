# SimpleTrack: Simple yet Effective 3D Multi-object Tracking

This is the repository for our paper [SimpleTrack: Understanding and Rethinking 3D Multi-object Tracking](https://arxiv.org/abs/2111.09621). We are still working on writing the documentations and cleaning up the code. However, we have already moved all of our code onto the `dev` branch, so please feel free to check it out if you really need to delve deep recently. We will try our best to get everything ready as soon as possible.

If you find our paper or code useful for you, please consider cite us by:
```
@article{pang2021simpletrack,
  title={SimpleTrack: Understanding and Rethinking 3D Multi-object Tracking},
  author={Pang, Ziqi and Li, Zhichao and Wang, Naiyan},
  journal={arXiv preprint arXiv:2111.09621},
  year={2021}
}
```
<img src="docs/simpletrack_gif.gif">

## Installation

### Environment Requirements

`SimpleTrack` requires `python>=3.6` and the packages of `pip install -r requiremens.txt`. For the experiments on Waymo Open Dataset, please install the devkit following the instructions at [waymo open dataset devkit](https://github.com/waymo-research/waymo-open-dataset).

### Installation

We implement the `SimpleTrack` algorithm as a library `mot_3d`. Please run `pip install -e ./` to install it locally.

## Demo and API Example

## Inference on Waymo Open Dataset and nuScenes

Refer to the documentation of [Waymo Open Dataset Inference](./docs/waymo.md) and [nuScenes Inference](./docs/nuScenes.md).

The detailed metrics on Waymo Open Dataset is at, nuScenes is at.

For the metrics on test set, please refer to our paper or the leaderboard.

## Related Documentations

To enable the better usages of our `mot_3d` library, we provide a list useful documentations, and will add more in the future.

* [Read and Use the Configurations](./docs/config.md). We explain how to specify the behaviors of trackers in this documentation, such as two-stage association, the thresholds for association, etc.
* [Format of Output](./docs/output_format.md). We explain the output format for the APIs in `SimpleTrack`, so that you may directly use the functions provided.
* Visualization with `mot_3d` (in progress)
* Structure of `mot_3d` (in progress)
