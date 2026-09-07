# Broad-regulatory-union ordinary-TEST weighted-F1 analysis

## Result

- Retained tokens: **671,930**.
- Filtered personalization gain (`F1_E3 - F1_E10`): **-0.01540980**.
- Unfiltered gain: **-0.01762267**.
- Filtered-minus-unfiltered gain: **+0.00221287**.
- Donor directions for that gain difference: 1/0/0 positive/negative/zero; nominal unadjusted sign `p=1`.

## Filter and calculation

The broad regulatory union overlaps a GENCODE v50 promoter (2 kb upstream to 500 bp downstream of a gene TSS) or any SCREEN Registry v4 cCRE; each token is retained once. Gene bodies and CpG-context annotations are excluded from this union. The union is boolean, so a token overlapping multiple annotations appears once.

Every hg38 prediction token is the BED0 half-open 6-base interval `[genomic_position, genomic_position+6)`. At least one base of overlap is required. E10 and E3 ordinary/regular TEST rows were paired exactly, and sklearn-compatible K=3 class-support-weighted F1 was calculated per participant. Overall results average the 31 participants equally.

This is an exploratory annotation analysis. The nominal donor sign test is descriptive and unadjusted; tokens are not independent biological replicates. Filtering changes class balance, so the personalization-gain comparison is more informative than comparing absolute filtered and unfiltered F1. All pairing, geometry, and class QC passed. Source data and prior outputs were not modified.
