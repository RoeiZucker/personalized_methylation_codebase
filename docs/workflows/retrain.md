# Retrain

## Purpose

Adapt a model to the target sample after the base datasets and pretrain outputs already exist.

## Main flow

- Create a `no_pretraining` retrain config for the target sample.
- Create retrain configs from each available pretrain checkpoint.
- Train and save retrain checkpoints in per-run output directories.

## Outputs

- Retrain training YAML files.
- Retrained model directories with saved checkpoints.

## LoRA variant

- Retrain can also run with LoRA settings.
- The project supports regular LoRA retrain and LoRA-over-LoRA retrain.
