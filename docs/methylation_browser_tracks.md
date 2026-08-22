# Methylation genome-browser tracks

`scripts/generate_methylation_browser_tracks.py` turns paired token-classification
predictions into window-level BED and bedGraph tracks. Positive scores mean the
retrained/personalized model has lower loss than the named baseline.

## Reproduce the standard analysis

The discovery command reads every `<participant>/test_eval_objects_dict.txt`,
uses the one pretrain prediction, and selects the largest numeric retraining
checkpoint. This is the third ordered training stage in the standard runs and
does not inspect test metrics.

```bash
python3 scripts/generate_methylation_browser_tracks.py \
  --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
  --split test \
  --output-dir scripts/results/methylation_browser_tracks_full \
  --bin-group bin3 --bin-group bin4 --bin-group bin5 \
  --bin-group bins3-5 --bin-group bins4-5 \
  --max-regions-per-track 1000
```

For a focused, repeatable run, filters may be repeated:

```bash
python3 scripts/generate_methylation_browser_tracks.py \
  --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
  --split test \
  --output-dir scripts/results/methylation_browser_tracks_breast_chr22 \
  --tissue Breast-Basal-Epithelial \
  --chrom chr22 \
  --bin-group bin3 --bin-group bin4 --bin-group bin5 \
  --bin-group bins3-5 --bin-group bins4-5
```

Supported bin groups are `bin1` through `bin5`, `bins3-5`, `bins4-5`, and
`all`. Omit `--bin-group` to generate every group. `--participant` can also be
repeated. `--max-regions-per-track` caps only the browser metadata BED (default
1000 per tissue/bin/comparison); complete compressed audit tables and continuous
bedGraphs are not capped. A hand-written TSV/CSV manifest can be supplied with `--manifest`;
required columns are `participant`, `tissue`, `retrained_path`,
`pretrained_path`, and `variability_path`, with optional `mean_path`.

## Output files

- `tracks/*.bedGraph`: sorted, non-overlapping continuous tracks by tissue,
  baseline comparison, and bin group. Overlapping windows are split into
  disjoint segments and averaged.
- `candidate_regions_top.bed`: bounded BED9+ high-variability candidate track.
  Extra columns contain tissue, bin group, comparison, improvement, standard
  error, exploratory p/q values, individual and observation counts, site count,
  sign-only favors-retrained/favors-baseline/no-difference class, and evidence
  status. It is capped independently for each tissue/bin/comparison.
- `ucsc_custom_tracks.txt`: one self-contained UCSC custom-track upload. Its
  region track is BED9 for UCSC compatibility; the complete BED9+ metadata
  remains in `candidate_regions_top.bed` for IGV and in the TSV audit files.
- `igv_session.xml`: IGV session using paths relative to the output directory.
- `window_summary.tsv.gz`: complete aggregate-window audit behind the continuous
  tracks. `participant_summary.tsv` and `aggregate_summary.tsv` contain the
  participant- and tissue/bin-level claims.
- `candidate_regions.tsv.gz`: complete high-variability candidate audit, including
  participant-direction consistency fields.
- `candidate_regions_top.tsv`: bounded ranked subset represented in
  `candidate_regions_top.bed`. Ranking prioritizes participant coverage,
  supported evidence, lower q-values, then larger absolute effects.
- `qc_summary.tsv`: per-participant model pairing, variability-match, and
  mean-baseline coverage counts/rates.
- `selected_manifest.tsv`: exact model paths, split, checkpoint number, and
  deterministic selection rule used.
- `REPORT.md`: scope, methods, observed evidence, and interpretation limits.

For a directory containing complete compressed summaries from an interrupted or
older run, regenerate only bounded browser artifacts without recomputing model
comparisons:

```bash
python3 scripts/finalize_methylation_browser_tracks.py \
  scripts/results/methylation_browser_tracks_full \
  --max-regions-per-track 1000
```

The finalizer uses the same ranking helper as the generator and refuses to
replace existing final browser artifacts.

## Coordinate and statistical assumptions

The source extraction targets hg38. The exporter treats window starts/ends as
0-based, half-open intervals and validates them against `window_id` and token
bounds. This is an explicit provenance assumption, not an independent
known-locus validation. `genomic_position` is only a join anchor; the track
generator does not claim it is the exact CpG cytosine coordinate. Variability
matching reproduces the
current evaluator's mean across matching positions from -1 through +5 bp.
Each participant's five bins are equal-width between zero and that
participant's maximum variability, so nominal bin boundaries can differ.

Accuracy improvement is `baseline 0/1 loss - retrained 0/1 loss`; `--metric
mae --prediction-column prediction` instead compares absolute numeric error.
Sites are averaged within participant/window before participants are averaged.
Standard error is SD/sqrt(n) across participants. Aggregate and window p-values
use two-sided participant-level sign-flip tests; FDR values are exploratory
Benjamini-Hochberg corrections. Mean-cache rows may be missing, so the mean
comparison uses matched rows while the pretrained comparison retains all paired
model rows.

The full standard output is written to
`scripts/results/methylation_browser_tracks_full`; the smaller real pilot in
`scripts/results/methylation_browser_tracks_pilot` remains useful for quick
inspection. Reports separate descriptive effects from FDR-supported evidence.
Tracks are exploratory region-nomination aids and do not imply optimized
hyperparameters or biological mechanism.

Generation refuses to write into a non-empty output directory, preventing stale
tracks from surviving a rerun. Use a new output directory for each analysis.
