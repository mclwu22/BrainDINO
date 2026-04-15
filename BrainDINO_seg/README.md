# BrainDINO Segmentation

This repository is a clean nnU-Net v2 based code release for the BrainDINO segmentation experiments.

It keeps only the BrainDINO-specific trainer and the model code it depends on, instead of the larger internal training tree used during development.

## What Is Included

- `nnunetv2/`: nnU-Net v2 code used for segmentation training and inference
- `nnunetv2/training/nnUNetTrainer/dinov3_base_trainer.py`: the nnU-Net trainer base used by BrainDINO
- `nnunetv2/training/nnUNetTrainer/braindino_trainer.py`: the released BrainDINO trainer
- `dinov3/`: the minimal BrainDINO model components required by the trainer

## Active Trainer

The main trainer used for the released experiments is:

- `BrainDINO_Primus_Multiscale_HighRes_Trainer_DataRatio`

The non-subsampled variant is also included:

- `BrainDINO_Primus_Multiscale_HighRes_Trainer`

## Pretrained Checkpoint

Before training, set the environment variable below to the converted BrainDINO teacher checkpoint:

```bash
export BRAINDINO_PRETRAINED_CKPT=/path/to/model.pth
```

For dataset-ratio experiments, also set:

```bash
export TRAIN_DATA_RATIO=0.8
```

## Example Training Command

```bash
nnUNetv2_train Dataset008_BraTS2023_MEN 2d_512 0 -tr BrainDINO_Primus_Multiscale_HighRes_Trainer_DataRatio
```

## Notes

- The Python package name remains `nnunetv2` for compatibility with nnU-Net tooling.
- This repository is based on nnU-Net v2 and keeps the original Apache 2.0 license.
- BrainDINO-specific code was separated from the larger research codebase to make public release easier to inspect and maintain.
