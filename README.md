# Individual-level methylation prediction

Code for *How predictable is an individual's DNA methylation?*

Nucleotide Transformer is fine-tuned first on reference donors and then on part of a target donor's methylation data. Comparisons use a mean atlas, local class averaging and a class-frequency random reference.

## Setup

Python 3.10 or newer. Model training and prediction are intended for Linux with a CUDA GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

For model work, install `pip install '.[model]'` with a PyTorch wheel suitable for your CUDA version. Coordinate conversion needs `pip install '.[mapping]'`.

## Figures and tables

```bash
methylation figures --data /path/to/donor_scores --output results
```

Given the study's donor-score files, this draws Figures 2–5 and S6 and writes Tables 1, S1 and S2. Figure 1 is an overview diagram. Donor-level results are not included in this code release.

## Run on methylation data

Obtain the processed hg38 tracks and donor metadata from [GSE186458](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186458), along with the hg38 FASTA. Tracks, prepared datasets and trained weights are not bundled.

```bash
methylation prepare --track target.bigwig --fasta hg38.fa --donor TARGET \
  --mode personalized --output datasets/TARGET

methylation train --stage donor-excluded --donor TARGET \
  --datasets datasets/REFERENCE1/TRAIN datasets/REFERENCE2/TRAIN \
  --output models/reference

methylation train --stage personalized --donor TARGET \
  --datasets datasets/TARGET/TRAIN --model models/reference/epoch-10 \
  --output models/personalized
```

Prepare reference donors with `--mode reference`. Include all other donors of the same cell type in the reference dataset list. The study uses epoch 10 for the donor-excluded model and epoch 3 for the personalized model. Main training and evaluation use chromosomes 1–5 and 18–22; `--mode chromosomes` prepares the separate 6–17 evaluation.

Use `predict` for saved models, `join` to align predictions, `reference` and `attach-reference` for the atlas and bins, and `local` for the 300-bp class-average comparison. Preserve frozen bin assignments when comparing methods. Export local training labels from the rebalanced TRAIN dataset with `training-sites`; the ten-site window filter is applied on export.

`annotate` reads BED intervals. `bimodal` reads donor-labelled regions and optionally an hg38-to-hg19 chain. `report`, `matching` and `chromosome-checks` calculate the summaries. Input columns are listed in `data/schema.json`; run `methylation COMMAND --help` for arguments. Run regular TEST and chromosomes 6–17 separately.

The model is `InstaDeepAI/nucleotide-transformer-2.5b-multi-species`. Classes are ≤0.2, between 0.2 and 0.8, and ≥0.8. Bootstrap intervals use 20,000 donor resamples. Sign tests exclude ties. New local-comparison q-values use subgroup families rather than the paper's earlier broad control screen.

Tests: `pip install '.[test]'`, then `pytest`.

## Acknowledgements

This research was made possible through a GPU compute resource grant funded by the Association of University Heads, the Council for Higher Education and the AI Research Compute Center.

MIT licence. The foundation model and source data retain their own terms.
