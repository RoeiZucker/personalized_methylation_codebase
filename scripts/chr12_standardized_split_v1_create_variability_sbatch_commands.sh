#!/usr/bin/env bash
set -euo pipefail

# Create chr12-only per-variant variability CSVs.
# Run this before chr12_standardized_split_v1_evaluation_sbatch_commands.sh.
# Example:
#   bash /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/chr12_standardized_split_v1_create_variability_sbatch_commands.sh

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z000000QM --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Bladder-Epithelial --chroms chr12 --sample Z000000QM --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z000000QP --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Bladder-Epithelial --chroms chr12 --sample Z000000QP --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000043F --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Bladder-Epithelial --chroms chr12 --sample Z0000043F --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000044L --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Bladder-Epithelial --chroms chr12 --sample Z0000044L --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z00000450 --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Bladder-Epithelial --chroms chr12 --sample Z00000450 --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z000000V6 --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Breast-Basal-Epithelial --chroms chr12 --sample Z000000V6 --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z000000VG --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Breast-Basal-Epithelial --chroms chr12 --sample Z000000VG --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z000000VL --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Breast-Basal-Epithelial --chroms chr12 --sample Z000000VL --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000043E --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Breast-Basal-Epithelial --chroms chr12 --sample Z0000043E --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z000000TD --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Cortex-Neuron --chroms chr12 --sample Z000000TD --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z000000TF --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Cortex-Neuron --chroms chr12 --sample Z000000TF --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000042F --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Cortex-Neuron --chroms chr12 --sample Z0000042F --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000042H --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Cortex-Neuron --chroms chr12 --sample Z0000042H --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000042J --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Cortex-Neuron --chroms chr12 --sample Z0000042J --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000042K --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Cortex-Neuron --chroms chr12 --sample Z0000042K --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000042M --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Cortex-Neuron --chroms chr12 --sample Z0000042M --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000042P --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Cortex-Neuron --chroms chr12 --sample Z0000042P --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000044G --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Heart-Cardiomyocyte --chroms chr12 --sample Z0000044G --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000044N --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Heart-Cardiomyocyte --chroms chr12 --sample Z0000044N --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000044P --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Heart-Cardiomyocyte --chroms chr12 --sample Z0000044P --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000044Q --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Heart-Cardiomyocyte --chroms chr12 --sample Z0000044Q --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000044R --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Heart-Cardiomyocyte --chroms chr12 --sample Z0000044R --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000041V --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Heart-Fibroblasts --chroms chr12 --sample Z0000041V --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000041W --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Heart-Fibroblasts --chroms chr12 --sample Z0000041W --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000041X --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Heart-Fibroblasts --chroms chr12 --sample Z0000041X --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000043R --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Heart-Fibroblasts --chroms chr12 --sample Z0000043R --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z000000T3 --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Liver-Hepatocytes --chroms chr12 --sample Z000000T3 --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z00000431 --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Liver-Hepatocytes --chroms chr12 --sample Z00000431 --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000043Q --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Liver-Hepatocytes --chroms chr12 --sample Z0000043Q --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000044H --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Liver-Hepatocytes --chroms chr12 --sample Z0000044H --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"

sbatch --time=1-23 --mem=45g -c4 --killable --requeue --job-name=chr12_var_Z0000044M --wrap="python3 /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_chr12_variability_files.py --tissue-name Liver-Hepatocytes --chroms chr12 --sample Z0000044M --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1"
