#!/usr/bin/env bash
set -euo pipefail

# Build chr12-only comparison dicts needed before chr12 evaluation.
# Example:
#   bash /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/chr12_standardized_split_v1_create_comparison_dicts_sbatch_commands.sh

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_dict_Bladder-Epithelial --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_comparison_dicts.py --tissue-name Bladder-Epithelial --chroms chr12 --output-dir /sci/labs/michall/roeizucker/comparison_dicts_chr12_standardized_v1/Bladder-Epithelial --sample Z000000QM --sample Z000000QP --sample Z0000043F --sample Z0000044L --sample Z00000450"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_dict_Breast-Basal-Epithelial --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_comparison_dicts.py --tissue-name Breast-Basal-Epithelial --chroms chr12 --output-dir /sci/labs/michall/roeizucker/comparison_dicts_chr12_standardized_v1/Breast-Basal-Epithelial --sample Z000000V6 --sample Z000000VG --sample Z000000VL --sample Z0000043E"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_dict_Cortex-Neuron --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_comparison_dicts.py --tissue-name Cortex-Neuron --chroms chr12 --output-dir /sci/labs/michall/roeizucker/comparison_dicts_chr12_standardized_v1/Cortex-Neuron --sample Z000000TD --sample Z000000TF --sample Z0000042F --sample Z0000042H --sample Z0000042J --sample Z0000042K --sample Z0000042M --sample Z0000042P"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_dict_Heart-Cardiomyocyte --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_comparison_dicts.py --tissue-name Heart-Cardiomyocyte --chroms chr12 --output-dir /sci/labs/michall/roeizucker/comparison_dicts_chr12_standardized_v1/Heart-Cardiomyocyte --sample Z0000044G --sample Z0000044N --sample Z0000044P --sample Z0000044Q --sample Z0000044R"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_dict_Heart-Fibroblasts --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_comparison_dicts.py --tissue-name Heart-Fibroblasts --chroms chr12 --output-dir /sci/labs/michall/roeizucker/comparison_dicts_chr12_standardized_v1/Heart-Fibroblasts --sample Z0000041V --sample Z0000041W --sample Z0000041X --sample Z0000043R"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_dict_Liver-Hepatocytes --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_comparison_dicts.py --tissue-name Liver-Hepatocytes --chroms chr12 --output-dir /sci/labs/michall/roeizucker/comparison_dicts_chr12_standardized_v1/Liver-Hepatocytes --sample Z000000T3 --sample Z00000431 --sample Z0000043Q --sample Z0000044H --sample Z0000044M"
