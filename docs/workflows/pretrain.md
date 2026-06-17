# Pretrain

## Purpose

Train a base model for each held-out sample using the other samples as training data.

## Main flow

- Create one pretrain training config per sample.
- Point each config at the train datasets of the other samples.
- Train the model and save `epoch-*` checkpoints in the output directory.

## Outputs

- Pretrain training YAML files.
- Pretrained model directories with saved checkpoints.

## LoRA variant

- The project can also create LoRA pretrain configs.
- This is the same workflow shape with different model settings.
