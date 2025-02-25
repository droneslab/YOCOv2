# You Only Crash Once v2

### [Paper Link](https://arxiv.org/abs/2501.13725)

Open source release of *You Only Crash Once v2* (YOCO), a domain adaptative approach to planetary, lunar, and small body surface terrain landmark detection built on top of the [ultralytics](https://github.com/ultralytics/ultralytics) object detection library.

## Installation
This repository was tested using Ubuntu 20.04 and Python 3.10, but should be applicable to a wide range of unix/Python setups. To get started, install the required Python packages:
```
pip install -r requirements.txt
```
## Dataset Format
YOCO uses the ultralytics format for training and testing datasets, integrated as `yaml` files in the `datasets/` directory. Please refer to their [documentation](https://docs.ultralytics.com/datasets/detect/) for more details on how to configure your data. Example config files are included for convenience.

## Training a YOCO Model
YOCO can be trained using the default parameters by executing
```
python train.py --source_data <PATH/TO/SOURCE_DATA.yaml> --target_data <PATH/TO/TARGET_DATA.yaml>
``` 
in the `src/` directory, where `--source_data` and `--target_data` point to the `.yaml` configs of labeled and unlabeled data, respectively.


## Training Parameters
The default parameters should work out for many environments and terrain types, but tweaks may be needed for more challenging scenarios. The main parameters affecting domain adaptation and detection performance are:

* `--arch`: String indicator for which YOLO architecture to attach the domain adaptation procedure to. This should be the first methodically set and most impactful parameter. Any architecture version (e.g., `v5`) and size variant (e.g., `v5n` for "YOLO V5 Nano") that is implemented within ultralytics is supported. Please see their [documentation](https://docs.ultralytics.com/models/) for a list of supported models. The following models were evaluated in the paper, based on their real-time performance on NASA SpaceCube hardware (dual-core ARM Cortex A9 with Google TPU attached via USB):
  * `v5[n/s/m]`: YOLO V5 Nano, Small, Medium
  * `v6[n/s]`: YOLO V6 Nano, Small
  * `v8[n/s/m]` YOLO V8 Nano, Small, Medium

* `--da_type`: High level algorithmic approach to domain adaptation, which is one of `[roi, sff, kmeans, ptap, yocov0]`. `sff` (default) should be valid in 99% of scenearios, but other approaches can be played with. `[roi, sff]` represent a region-of-interest (i.e., bounding box detection crops) alignment approach while `[kmeans, ptap, yocov0]` attempt alignment based on dependencies found within full feature maps. Generally speaking, `sff > roi > kmeans > yocov0 > ptap`, but your mileage may vary.
* `--da_loss`: The biggest factor in performance - the method used to calculate domain adaptation loss in `[roi, sff]` approaches (other schemes have dedicated techniques). This can be one of `[disc, cont]` for "adversarial discrimination" and "contrastive learning" respectively. Contrastive (`cont`) is set as default, although both approaches have trade-offs (see the paper for more details).

Other selectable training parameters:

* `--epochs`: Training epochs.
* `--batch_size`: Training batch size.
* `--yolo`: Disable YOCO domain adaptation and train a regular YOLO model.
* `--no-pc`: Disable perceptual consistency regularization.
* `--qual`: Generate detection plots on target data.
* `--logdir`: Logging directory for model weights, tensorboard, WandB, etc.
* `--wb`: Enable WandB logging.
* `--exp`: Experiment name to append to the logging string.
* `--pretrained`: Weights to load before training.
