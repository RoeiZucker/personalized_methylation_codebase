from pathlib import Path
from pprint import pformat
import sys

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from variability_free_evaluation import evaluate_sample_predictions


RESULTS_DIR = Path("/sci/labs/michall/roeizucker/cls_downsample_liver")
VARIABILITY_DIR = Path(
    "/sci/labs/michall/roeizucker/huggingface_datasets_dir/"
    "huggingface_datasets_dir/_Liver-Hepatocytes_kmer"
)
BIGWIG_FILES_BY_SAMPLE = {
    "Z000000R3": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652233_Liver-Hepatocytes-Z000000R3.hg38.bigwig",
    "Z000000T3": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652234_Liver-Hepatocytes-Z000000T3.hg38.bigwig",
    "Z0000043Q": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652235_Liver-Hepatocytes-Z0000043Q.hg38.bigwig",
    "Z0000044H": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652236_Liver-Hepatocytes-Z0000044H.hg38.bigwig",
    "Z0000044M": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652237_Liver-Hepatocytes-Z0000044M.hg38.bigwig",
    "Z00000431": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652238_Liver-Hepatocytes-Z00000431.hg38.bigwig",
}

FULL_POS_NAME = "full_pos"
RANGES = [0, 0.2, 0.8, 1]
LABELS = [0, 1, 2]
LABEL_A = "mean_label"
LABEL_B = "predicted_class"
COMPARISON_TYPES = [LABEL_A, LABEL_B]
ALL_TWO = True
NUMBER_OF_BINS = 5
OUTPUT_PATH = PROJECT_DIR / "scripts" / "results" / "cls_downsample_liver_variability_free_eval_objects_dict.txt"


def sample_name_from_result_path(result_file_path):
    run_name = result_file_path.parents[1].name
    return run_name.split("_", 1)[0]


def variability_file_path(sample_name):
    return (
        VARIABILITY_DIR
        / f"{sample_name}_per_varaint_variability_Liver-Hepatocytes_kmer_seq_5400_datasets.csv"
    )


def get_chroms(result_files_path):
    chroms = set()
    for result_file_path in result_files_path:
        window_ids = pd.read_csv(result_file_path, usecols=["window_id"])["window_id"]
        chroms.update(window_ids.str.split(":", n=1).str[0].dropna().unique())
    return sorted(chroms, key=lambda chrom: int(chrom[3:]) if chrom[3:].isdigit() else chrom)


def discover_result_files_by_sample():
    result_files_by_sample = {}
    for result_file_path in sorted(RESULTS_DIR.glob("*/epoch-*/eval_predictions.csv.gitbackup")):
        sample_name = sample_name_from_result_path(result_file_path)
        result_files_by_sample.setdefault(sample_name, []).append(str(result_file_path))
    return result_files_by_sample


def main():
    result_files_by_sample = discover_result_files_by_sample()
    eval_objects_dict = {}

    for sample_name, result_files_path in result_files_by_sample.items():
        print(sample_name)
        chroms = get_chroms(result_files_path)
        comparison_bigiwg_files = [
            bigwig_file
            for curr_sample, bigwig_file in BIGWIG_FILES_BY_SAMPLE.items()
            if curr_sample != sample_name
        ]

        eval_objects_dict[sample_name] = evaluate_sample_predictions(
            result_files_path=result_files_path,
            chroms=chroms,
            comparison_bigiwg_files=comparison_bigiwg_files,
            full_pos_name=FULL_POS_NAME,
            ranges=RANGES,
            labels=LABELS,
            comparison_types=COMPARISON_TYPES,
            all_two=ALL_TWO,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(pformat(eval_objects_dict, sort_dicts=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
