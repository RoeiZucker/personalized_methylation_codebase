import argparse
import ast
import os
from pathlib import Path

import yaml

def _default_repo_root():
    override = os.environ.get("TOKEN_CLASSIFICATION_REPO_ROOT")
    if override:
        return Path(override)

    preferred_root = Path("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code")
    if preferred_root.exists():
        return preferred_root

    pwd = Path(os.environ.get("PWD", os.getcwd()))
    for candidate in (pwd / "Tom_Hope_Project" / "refactored_code", pwd):
        if (candidate / "scripts").is_dir() and (candidate / "src").is_dir():
            return candidate

    return Path(__file__).absolute().parents[1]


try:
    from .constants import KMER_SAMPLE_TRAIN_TEST_FILTRATION, RANDOM_SAMPLE_TRAIN_TEST_FILTRATION
    from .token_classification_project_creator_script import (
        DEFAULT_TOKEN_LABEL_BINNING,
        DEFAULT_TOKEN_LABEL_DOWNSAMPLING,
        build_master_project_config,
        create_token_classification_project_from_master_config,
    )
except ImportError:
    from constants import KMER_SAMPLE_TRAIN_TEST_FILTRATION, RANDOM_SAMPLE_TRAIN_TEST_FILTRATION
    from token_classification_project_creator_script import (
        DEFAULT_TOKEN_LABEL_BINNING,
        DEFAULT_TOKEN_LABEL_DOWNSAMPLING,
        build_master_project_config,
        create_token_classification_project_from_master_config,
    )


REPO_ROOT = _default_repo_root()
DEFAULT_EXISTING_UBER_SCRIPT = Path(__file__).with_name("uber_project_creator_script.py")
DEFAULT_BASE_FILE_PATH = "/sci/archive/michall/roeizucker/downloaded_datasets"
DEFAULT_PROJECT_CONFIG_PATH = REPO_ROOT / "configs" / "token_classification_project_configs" / "auto_created"
DEFAULT_BASE_CONFIG_PATH = REPO_ROOT / "configs" / "token_classification_project_configs"
DEFAULT_DATASET_BASE_DIR = "/sci/archive/michall/roeizucker/huggingface_datasets_dir"
DEFAULT_BASE_MODEL_DIR = "/sci/labs/michall/roeizucker/trained_huggingface_models_location"


def _parse_csv_values(value, cast=str):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _extract_files_names_from_python_source(source_path):
    source_text = Path(source_path).read_text()
    tree = ast.parse(source_text)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "FILES_NAMES" for target in node.targets):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "split"
            and isinstance(value.func.value, ast.Constant)
            and isinstance(value.func.value.value, str)
        ):
            split_arg = "\n"
            if value.args and isinstance(value.args[0], ast.Constant):
                split_arg = value.args[0].value
            return [name for name in value.func.value.value.split(split_arg) if name]
    raise ValueError(f"Could not find FILES_NAMES assignment in {source_path}")


def load_file_names(files_list_path=None, existing_uber_script=DEFAULT_EXISTING_UBER_SCRIPT):
    if files_list_path is not None:
        with open(files_list_path, "r") as handle:
            return [line.strip() for line in handle if line.strip()]
    return _extract_files_names_from_python_source(existing_uber_script)


def group_bigwig_files(file_names):
    grouped_files = {}
    for file_name in file_names:
        sample_part = "_".join(file_name.split("_")[1:])
        suffix = "-".join(sample_part.split("-")[:-1])
        if not suffix:
            print(file_name)
            continue
        grouped_files.setdefault(suffix, []).append(file_name)
    return grouped_files


def _write_yaml(path, config, overwrite=False, dry_run=False):
    if os.path.exists(path) and not overwrite:
        return "skipped"
    if dry_run:
        print(f"would write {path}")
        return "created"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
    return "created"


def _selected_splits(split):
    if split == "kmer":
        return [("kmer", KMER_SAMPLE_TRAIN_TEST_FILTRATION)]
    if split == "window":
        return [("window", RANDOM_SAMPLE_TRAIN_TEST_FILTRATION)]
    return [
        ("kmer", KMER_SAMPLE_TRAIN_TEST_FILTRATION),
        ("window", RANDOM_SAMPLE_TRAIN_TEST_FILTRATION),
    ]


def create_token_classification_uber_project_configs(
    file_names,
    base_file_path=DEFAULT_BASE_FILE_PATH,
    base_project_config_path=DEFAULT_PROJECT_CONFIG_PATH,
    base_config_path=DEFAULT_BASE_CONFIG_PATH,
    dataset_base_dir=DEFAULT_DATASET_BASE_DIR,
    base_model_dir=DEFAULT_BASE_MODEL_DIR,
    group_min_size=4,
    split="both",
    project_prefix="_token_cls_",
    tokenizer_name="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    token_label_binning=None,
    token_label_downsampling=DEFAULT_TOKEN_LABEL_DOWNSAMPLING,
    chromosomes=None,
    learning_rates=None,
    batch_sizes=None,
    per_device_eval_batch_size=None,
    seq_sizes=None,
    test_sizes=None,
    num_train_epoch=5,
    num_pretrain_epoch=2,
    batch_size_overrides=None,
    limit_groups=None,
    overwrite=False,
    dry_run=False,
    expand_projects=False,
):
    if token_label_binning is None:
        token_label_binning = DEFAULT_TOKEN_LABEL_BINNING
    if chromosomes is None:
        chromosomes = ["chr5"]
    if learning_rates is None:
        learning_rates = [1e-6]
    if batch_sizes is None:
        batch_sizes = [1]
    if per_device_eval_batch_size is None:
        per_device_eval_batch_size = batch_sizes[0]
    if seq_sizes is None:
        seq_sizes = [5400]
    if test_sizes is None:
        test_sizes = [0.2]

    grouped_files = group_bigwig_files(file_names)
    created_paths = []
    created = 0
    skipped = 0
    considered_groups = 0

    for suffix in sorted(grouped_files):
        if len(grouped_files[suffix]) < group_min_size:
            continue
        considered_groups += 1
        if limit_groups is not None and considered_groups > limit_groups:
            break

        tissue_dir = os.path.join(str(base_project_config_path), suffix)
        if dry_run:
            print(f"mkdir -p {tissue_dir}")
        else:
            os.makedirs(tissue_dir, exist_ok=True)

        for split_name, filtration_method in _selected_splits(split):
            project_name = f"{suffix}_{split_name}"
            master_config = build_master_project_config(
                name=project_name,
                grouped_files=grouped_files[suffix],
                base_file_path=base_file_path,
                base_config_path=str(base_config_path),
                dataset_base_dir=dataset_base_dir,
                base_model_location=base_model_dir,
                filtration_method=filtration_method,
                project_prefix=project_prefix,
                tokenizer_name=tokenizer_name,
                token_label_binning=token_label_binning,
                token_label_downsampling=token_label_downsampling,
                chromosomes=chromosomes,
                learning_rates=learning_rates,
                per_device_train_batch_sizes=batch_sizes,
                per_device_eval_batch_size=per_device_eval_batch_size,
                seq_sizes=seq_sizes,
                test_sizes=test_sizes,
                num_train_epoch=num_train_epoch,
                num_pretrain_epoch=num_pretrain_epoch,
                batch_size_overrides=batch_size_overrides,
            )
            output_path = os.path.join(tissue_dir, f"{project_name}.yaml")
            status = _write_yaml(output_path, master_config, overwrite=overwrite, dry_run=dry_run)
            if status == "created":
                created += 1
            else:
                skipped += 1
            created_paths.append(output_path)

            if expand_projects and dry_run:
                print(f"would expand {output_path}")
            elif expand_projects:
                create_token_classification_project_from_master_config(
                    output_path,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )

    return {
        "created": created,
        "skipped": skipped,
        "master_config_paths": created_paths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Create many token-classification master project configs.")
    parser.add_argument("--files-list-path", help="Optional newline-delimited bigWig file-name list.")
    parser.add_argument("--existing-uber-script", default=str(DEFAULT_EXISTING_UBER_SCRIPT))
    parser.add_argument("--base-file-path", default=DEFAULT_BASE_FILE_PATH)
    parser.add_argument("--base-project-config-path", default=str(DEFAULT_PROJECT_CONFIG_PATH))
    parser.add_argument("--base-config-path", default=str(DEFAULT_BASE_CONFIG_PATH))
    parser.add_argument("--dataset-base-dir", default=DEFAULT_DATASET_BASE_DIR)
    parser.add_argument("--base-model-dir", default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--group-min-size", type=int, default=4)
    parser.add_argument("--split", choices=["both", "kmer", "window"], default="both")
    parser.add_argument("--project-prefix", default="_token_cls_")
    parser.add_argument("--tokenizer-name", default="InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
    parser.add_argument("--binning-method", choices=["fixed", "quantile"], default="fixed")
    parser.add_argument("--low", type=float, default=0.2)
    parser.add_argument("--high", type=float, default=0.8)
    parser.add_argument(
        "--downsampling-minority-to-majority-ratio",
        type=float,
        help="Optional token-label downsampling ratio. Example: 1.0 balances classes; 0.5 allows 2x majority.",
    )
    parser.add_argument("--chromosomes", default="chr5")
    parser.add_argument("--learning-rates", default="1e-6")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument("--seq-sizes", default="5400")
    parser.add_argument("--test-sizes", default="0.2")
    parser.add_argument("--num-train-epoch", type=int, default=5)
    parser.add_argument("--num-pretrain-epoch", type=int, default=2)
    parser.add_argument(
        "--seq-batch-size-overrides",
        default="",
        help="Optional comma list like 5400:14,600:128.",
    )
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expand-projects", action="store_true")
    return parser.parse_args()


def _parse_batch_size_overrides(value):
    if not value:
        return {}
    overrides = {}
    for item in value.split(","):
        if not item.strip():
            continue
        seq_size, batch_size = item.split(":", 1)
        overrides[int(seq_size.strip())] = int(batch_size.strip())
    return overrides


def main():
    args = parse_args()
    file_names = load_file_names(
        files_list_path=args.files_list_path,
        existing_uber_script=args.existing_uber_script,
    )
    token_label_binning = {
        "method": args.binning_method,
        "low": args.low,
        "high": args.high,
    }
    token_label_downsampling = None
    if args.downsampling_minority_to_majority_ratio is not None:
        token_label_downsampling = {
            "enabled": True,
            "minority_to_majority_ratio": args.downsampling_minority_to_majority_ratio,
        }
    summary = create_token_classification_uber_project_configs(
        file_names=file_names,
        base_file_path=args.base_file_path,
        base_project_config_path=args.base_project_config_path,
        base_config_path=args.base_config_path,
        dataset_base_dir=args.dataset_base_dir,
        base_model_dir=args.base_model_dir,
        group_min_size=args.group_min_size,
        split=args.split,
        project_prefix=args.project_prefix,
        tokenizer_name=args.tokenizer_name,
        token_label_binning=token_label_binning,
        token_label_downsampling=token_label_downsampling,
        chromosomes=_parse_csv_values(args.chromosomes, str),
        learning_rates=_parse_csv_values(args.learning_rates, float),
        batch_sizes=_parse_csv_values(args.batch_sizes, int),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        seq_sizes=_parse_csv_values(args.seq_sizes, int),
        test_sizes=_parse_csv_values(args.test_sizes, float),
        num_train_epoch=args.num_train_epoch,
        num_pretrain_epoch=args.num_pretrain_epoch,
        batch_size_overrides=_parse_batch_size_overrides(args.seq_batch_size_overrides),
        limit_groups=args.limit_groups,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        expand_projects=args.expand_projects,
    )
    print(f"created {summary['created']} master configs, skipped {summary['skipped']} existing master configs")
    for path in summary["master_config_paths"]:
        print(path)


if __name__ == "__main__":
    main()
