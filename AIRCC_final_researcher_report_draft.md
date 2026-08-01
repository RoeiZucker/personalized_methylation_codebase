# AIRCC Final Researcher Report Draft

Principal Investigator Name: [PI name]

Academic Institute: [Institute]

Additional Researcher 1: Roei Zucker

Additional Researcher 2: [Optional]

Research Topic: GPU-accelerated token-classification models for predicting genomic signal classes across GTEx tissue/cell-type datasets

Research Period: [Start date] - [End date]

Report Date: [Submission date]

AIRCC usage note: This draft refers to GPU runs performed on AIRCC servers other than moriah or phoenix. Exact server names, GPU model names, job counts, and GPU-hours should be filled from the AIRCC/accounting logs before submission.

## Scientific Report

During the reporting period, AIRCC GPU resources were used to develop, train, and evaluate a refactored deep-learning pipeline for tissue- and cell-type-specific genomic token classification. The project studies whether sequence-context models can learn per-token class labels from genomic windows and whether pretraining on related individuals improves prediction for a held-out individual.

The work focused on a leave-one-individual-out workflow. For each target individual, a base model was pretrained using the remaining samples, then retrained/adapted to the target sample and evaluated on held-out genomic regions. The pipeline includes data extraction, configuration generation, full model training, LoRA-based training variants, checkpoint prediction, grouped evaluation, and automated report generation. The main evaluated tissues/cell types were Bladder-Epithelial, Breast-Basal-Epithelial, Cortex-Neuron, Heart-Cardiomyocyte, Heart-Fibroblasts, and Liver-Hepatocytes, totaling 33 individual-level evaluations in the current aggregate report.

The GPU allocation enabled experiments that are not practical on CPU: repeated pretraining over large genomic token datasets, checkpoint sweeps over multiple epochs, per-individual retraining, hyperparameter/downsampling probes, and large batched inference for evaluation. This GPU-supported work set the groundwork for the current stage of the project: we now have trained checkpoints, evaluation outputs, and analysis code that make it possible to compare tissues, individuals, and genomic variability regimes systematically. A shared pretrain-only checkpoint selection analysis found that epoch 10 was a strong common checkpoint across the six evaluated tissue/cell-type groups. The best pretrain-only weighted F1 values in the generated aggregate report were approximately 0.807 for Bladder-Epithelial, 0.878 for Breast-Basal-Epithelial, 0.898 for Cortex-Neuron, 0.838 for Heart-Cardiomyocyte, 0.794 for Heart-Fibroblasts, and 0.866 for Liver-Hepatocytes.

For liver-focused runs, pretrain-only performance increased steadily across training epochs. In one grouped comparison, the mean weighted F1 across evaluated samples rose from about 0.760 at pretrain epoch 1 to about 0.840 at epoch 10 and 0.843 at epoch 11. Liver-Hepatocytes variability-free evaluations showed that the learned classifier clearly outperformed a trivial all-class baseline: weighted F1 averaged about 0.824 for model predictions compared with about 0.552 for the all-class baseline across six samples.

The analysis also examined performance as a function of inter-site variability. For a Liver-Hepatocytes sample, the pretrained model achieved weighted F1 of about 0.865 over all evaluated sites. The key improvement over mean-label aggregation appeared in higher-variability sites, where the task is scientifically more important and simple averaging is less reliable. For example, the pretrained model reached about 0.759 versus 0.725 weighted F1 in the 0.1068-0.2136 variability bin, about 0.648 versus 0.512 in the 0.2136-0.3204 bin, about 0.395 versus 0.339 in the 0.3204-0.4272 bin, and about 0.439 versus 0.332 in the 0.4272-0.534 bin. These results suggest that pretraining is especially valuable at high-variability genomic sites, where it captures local/contextual structure beyond the mean-label baseline.

Current work is focused on final tweaking of the retraining stage: improving the adaptation from pretrained checkpoints to held-out individuals, comparing pretraining and no-pretraining settings, evaluating variability-free and variability-bin outcomes, and expanding grouped evaluations. We plan to use the remaining allocated GPU hours for these final retraining improvements and for the corresponding evaluation sweeps. The next research steps are to complete additional tissue/cell-type sweeps, improve robustness in high-variability genomic sites, finalize the comparison between full fine-tuning and LoRA-style adaptation, and prepare the results for a manuscript or conference submission. The expected publication will report the pretraining/retraining methodology, individual-level generalization setting, variability-stratified evaluation, and compute-enabled scaling across multiple tissue/cell-type groups.

## Benefit And Importance Of AIRCC GPU Cluster

AIRCC GPU resources were crucial to this project. The experiments required repeated training and inference over large tokenized genomic windows, many held-out individuals, multiple checkpoints, and several tissue/cell-type groups. The GPU cluster made it possible to run pretraining, retraining, evaluation, and checkpoint-selection experiments at a scale and turnaround time that would not have been feasible on local CPU resources.

The most important contribution of the cluster was enabling systematic comparison rather than isolated single runs. Because GPU jobs could be distributed across the AIRCC environment, the project could evaluate multiple samples, pretraining epochs, fine-tuning epochs, and analysis modes, including variability-free evaluations, variability-bin evaluations, and downsampling/hyperparameter probes. This directly improved the scientific reliability of the conclusions by making it possible to separate stable trends from individual-run noise.

The cluster was also useful for practical research iteration. Intermediate checkpoints could be generated and re-used for downstream prediction and grouped evaluation, which supported a more modular workflow: train once, evaluate multiple ways, then aggregate results into tissue-level and individual-level reports. The resulting artifacts now include automated evaluation inventories, grouped metric CSVs, visual summaries, and a shared project report for review and signature by the PI and researcher.

Overall, AIRCC support substantially accelerated the project and enabled a more complete experimental design. The quality of the environment was sufficient for long-running model training and large evaluation jobs, and the available GPU capacity made the work possible within the research period.

## Improvement Suggestions

The main improvement that would help future reporting is easier access to per-user and per-project GPU accounting, including job counts, wall time, GPU-hours, GPU model type, and server names. Having a standard export for this information would make final AIRCC reports more accurate and easier to prepare.

It would also be useful to have clearer examples for recommended job-submission patterns for large multi-run experiments, including array jobs, checkpoint-heavy workflows, and best practices for organizing outputs across servers. Finally, a short checklist for final report preparation, including where to retrieve usage summaries and how to cite AIRCC resources, would reduce administrative overhead at the end of a project.

## Numbers To Fill Before Submission

- AIRCC server names used: [fill from logs]
- GPU types used: [fill from logs]
- Number of jobs: [fill from scheduler/accounting]
- Total GPU-hours: [fill from scheduler/accounting]
- Total wall-clock period of GPU activity: [fill from logs]
- Publication target or planned venue/date: [fill if known]
