# Extraction

## Purpose

Build the datasets used later by training and evaluation.

## Main flow

- Create variability extraction configs.
- Create pretrain extraction configs.
- Create retrain extraction configs.
- Run extraction scripts to materialize dataset files on disk.

## Outputs

- Variability CSV files.
- Pretrain train datasets.
- Retrain train and test datasets.

## Notes

- Extraction is the first stage of the project.
- Later stages assume these datasets already exist and are in the expected paths.
