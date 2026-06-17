import os
import sys

import yaml

try:
    from .config_manager import create_base_dictionary, create_model_config
except ImportError:
    from config_manager import create_base_dictionary, create_model_config


ATLAS_EVAL_CONFIG_DIR_NAME = "atlas_eval_configs"
ATLAS_EVAL_SUBTASK = "atlas_evaluation"
ATLAS_EVAL_TASK_TYPE = "cpg_evaluation"
ATLAS_VARIANT_GROUPING_METHOD = "bins"


def _load_project_config(config_path):
    if not config_path.endswith(".yaml") or not os.path.exists(config_path):
        raise ValueError(f"invalid config path: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict) or "params" not in config:
        raise ValueError("project config must contain a top-level 'params' section")

    return config["params"]


def _split_tokenizer_name(tokenizer_name):
    if "/" not in tokenizer_name:
        return "", tokenizer_name
    return tokenizer_name.split("/", 1)


def _resolve_base_suffix(params):
    return params.get("base_suffix", params["project_suffix"])


def _resolve_seq_sizes(params):
    if "seq_size" in params and params["seq_size"] is not None:
        return [params["seq_size"]]

    seq_sizes = params.get("seq_sizes")
    if not seq_sizes:
        raise ValueError("project config must define either 'seq_size' or 'seq_sizes'")
    return seq_sizes


def _resolve_effective_base_dir(base_dir, base_suffix):
    normalized_base = os.path.basename(os.path.normpath(base_dir))
    if normalized_base == base_suffix:
        return base_dir
    return os.path.join(base_dir, base_suffix)


def _resolve_datasets_suffix(params, base_suffix, seq_size, seq_sizes):
    explicit_suffix = params.get("datasets_suffix")
    if explicit_suffix and len(seq_sizes) == 1:
        return explicit_suffix
    return f"{base_suffix}_seq_{seq_size}_datasets"


def _build_analysis_name(sample_name, datasets_suffix):
    if datasets_suffix.endswith("_datasets"):
        datasets_suffix = datasets_suffix[: -len("_datasets")]
    return f"{sample_name}_atlas_eval{datasets_suffix}"


def _build_model_config(params):
    model_repo, model_name = _split_tokenizer_name(params["tokenizer_name"])
    freeze_model = params.get("freeze_model", params.get("freeze_mode", False))
    use_lora = params.get("use_lora", False)
    num_labels = params.get("num_labels", 1)
    return create_model_config(
        params["model_type"],
        use_lora,
        freeze_model,
        num_labels,
        model_name,
        model_repo,
    )


def _build_sample_bigwig_map(params):
    names = params["names"]
    bigwig_files = params["bigwig_files"]
    if len(names) != len(bigwig_files):
        raise ValueError("project config must provide the same number of names and bigwig_files")
    return dict(zip(names, bigwig_files))


def _build_atlas_eval_config(params, sample_name, seq_size, seq_sizes):
    names = params["names"]
    if len(names) < 2:
        raise ValueError("atlas evaluation requires at least two sample names")

    sample_bigwig_map = _build_sample_bigwig_map(params)
    base_suffix = _resolve_base_suffix(params)
    dataset_base_dir = _resolve_effective_base_dir(params["dataset_base_dir"], base_suffix)
    datasets_suffix = _resolve_datasets_suffix(params, base_suffix, seq_size, seq_sizes)

    target_bigwig_path = sample_bigwig_map[sample_name]
    atlas_bigwig_paths = [
        sample_bigwig_map[other_name]
        for other_name in names
        if other_name != sample_name
    ]
    variant_file_path = os.path.join(
        dataset_base_dir,
        f"{sample_name}_per_varaint_variability{datasets_suffix}.csv",
    )

    base_dict = create_base_dictionary()
    base_dict["paths"]["variant_file_path"] = variant_file_path
    base_dict["paths"]["target_bigwig_path"] = target_bigwig_path
    base_dict["paths"]["atlas_bigwig_paths"] = atlas_bigwig_paths
    base_dict["paths"]["model_paths"] = []

    base_dict["task"]["task_type"] = ATLAS_EVAL_TASK_TYPE
    base_dict["task"]["sub_task"] = ATLAS_EVAL_SUBTASK
    base_dict["task"]["analysis_name"] = _build_analysis_name(sample_name, datasets_suffix)
    base_dict["task"]["top_rows"] = params.get("top_rows", -1)
    base_dict["task"]["use_variant_file"] = True
    base_dict["task"]["vriant_grouping_method"] = ATLAS_VARIANT_GROUPING_METHOD
    base_dict["task"]["number_of_bins"] = params.get("number_of_bins", 10)
    base_dict["task"]["base_suffix"] = base_suffix
    if params.get("chromosomes") is not None:
        base_dict["task"]["chromosomes"] = params["chromosomes"]

    base_dict["model"] = _build_model_config(params)
    return base_dict


def create_atlas_eval_configs(project_config_path):
    params = _load_project_config(project_config_path)
    seq_sizes = _resolve_seq_sizes(params)
    output_root = os.path.join(params["created_configs_path"], ATLAS_EVAL_CONFIG_DIR_NAME)
    os.makedirs(output_root, exist_ok=True)

    written_paths = []
    for seq_size in seq_sizes:
        seq_output_dir = os.path.join(output_root, f"seq_{seq_size}")
        os.makedirs(seq_output_dir, exist_ok=True)

        for sample_name in params["names"]:
            atlas_config = _build_atlas_eval_config(params, sample_name, seq_size, seq_sizes)
            output_path = os.path.join(
                seq_output_dir,
                f"{sample_name}_atlas_eval.yaml",
            )
            with open(output_path, "w", encoding="utf-8") as handle:
                yaml.dump(atlas_config, handle, default_flow_style=False, sort_keys=False)
            written_paths.append(output_path)

    return written_paths


def main():
    project_config_path = sys.argv[1] if len(sys.argv) > 1 else input("enter project config path: ").strip()
    written_paths = create_atlas_eval_configs(project_config_path)
    print("created atlas evaluation configs:")
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
