from datasets import Dataset, load_from_disk
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from transformers import AutoTokenizer

try:
    from .utils.metrics_utils import mae, mse, pearsonr_eval
    from .utils.tissue_model_utils import get_fine_tuned_model
    from .utils.tissue_trainer_utils import get_trainer
    from .utils.atlas_bigwig_utils import evaluate_atlas_from_bigwigs
    from .utils.label_transform_utils import (
        apply_label_transform_to_dataset,
        get_task_label_transform,
        maybe_decode_regression_predictions,
    )
except ImportError:
    from utils.metrics_utils import mae, mse, pearsonr_eval
    from utils.tissue_model_utils import get_fine_tuned_model
    from utils.tissue_trainer_utils import get_trainer
    from utils.atlas_bigwig_utils import evaluate_atlas_from_bigwigs
    from utils.label_transform_utils import (
        apply_label_transform_to_dataset,
        get_task_label_transform,
        maybe_decode_regression_predictions,
    )


def create_window_id_dataset_dict(dataset):
    keys = []
    index = 0
    dataset_dict = {}
    for val in dataset:
        dataset_dict[val["window_id"]] = val
        keys.append(val["window_id"])
        index += 1
    return dataset_dict, keys


def evaluate_atlas(dataset, dataset_labels, variability_positions_dict, bins_to_use, atlas_datasets_to_use, verbose):
    dataset_dicts_to_use = []
    counter = 0
    for val in atlas_datasets_to_use:
        if verbose:
            print("creating atlas dict:", counter, "/", len(atlas_datasets_to_use), end="\r")
        counter += 1
        dic, temp_keys = create_window_id_dataset_dict(val)
        dataset_dicts_to_use.append(dic)
    if verbose:
        print("")
    res = []

    if verbose:
        print("length of dataset:", len(dataset), flush=True)
    for curr_bin in bins_to_use:
        if verbose:
            print("curr_bin:", curr_bin)
        labels_from_dataset = []
        labels_from_prediction = []
        for curr_dataset in range(len(dataset)):
            curr_window_id = dataset[curr_dataset]["window_id"]
            if curr_window_id not in variability_positions_dict:
                continue
            if curr_bin not in variability_positions_dict[curr_window_id]:
                continue
            viable_positions = variability_positions_dict[curr_window_id][curr_bin]
            filtered_labels_from_dataset = dataset_labels[curr_window_id][curr_bin]
            other_vals = []
            for dic in dataset_dicts_to_use:
                if curr_window_id in dic:
                    other_vals.append(np.array(dic[curr_window_id]["labels"]))

            if len(other_vals) == 0:
                continue
            other_vals = np.array(other_vals)
            other_vals[other_vals == -100] = np.nan
            if np.isnan(other_vals).all():
                print("all nans!!", flush=True)
            predictions = list(np.nanmean(other_vals, axis=0))
            filtered_from_prediction = [predictions[i + 1] for i in viable_positions]
            labels_from_dataset.extend(filtered_labels_from_dataset)
            labels_from_prediction.extend(filtered_from_prediction)

        if len(labels_from_prediction) <= 1:
            continue

        res_r, res_mse, res_mae = get_cpg_reesults_for_labels(labels_from_dataset, labels_from_prediction, verbose)
        bin_results = [curr_bin, res_r, res_mse, res_mae]
        if verbose:
            print(bin_results, flush=True)
        res.append(bin_results)
    return res


def evaluate_checkpoint(
    model_repo: str,
    model_name: str,
    is_lora: bool,
    num_labels: int,
    dataset: Dataset,
    task_type,
    model_path: str,
    model_type,
    variability_positions_dict,
    use_variant_file: bool,
    vriant_grouping_method,
    bins_to_use,
    dataset_labels,
    verbose: bool,
    tissue_prompt_config=None,
    task_level="token",
    per_device_eval_batch_size: int = 1,
    eval_accumulation_steps: int = 4,
    label_transform="none",
) -> pd.DataFrame:
    if verbose:
        print("Evaluating model at path:", model_path)
    base_model_name = model_repo + "/" + model_name
    if use_variant_file:
        prediction = predict_checkpoint(
            dataset,
            dataset,
            model_type,
            is_lora,
            num_labels,
            model_repo,
            model_name,
            model_path,
            tissue_prompt_config=tissue_prompt_config,
            task_level=task_level,
            per_device_eval_batch_size=64,
            eval_accumulation_steps=10,
        )
        res = []
        if verbose:
            print("length of dataset:", len(dataset), flush=True)
        for curr_bin in bins_to_use:
            if verbose:
                print("curr_bin:", curr_bin)
            labels_from_dataset = []
            labels_from_prediction = []
            for curr_dataset in range(len(prediction.predictions)):
                curr_window_id = dataset[curr_dataset]["window_id"]
                if curr_window_id not in variability_positions_dict:
                    continue
                if curr_bin not in variability_positions_dict[curr_window_id]:
                    continue
                viable_positions = variability_positions_dict[curr_window_id][curr_bin]
                filtered_labels_from_dataset = dataset_labels[curr_window_id][curr_bin]
                labels_from_dataset.extend(filtered_labels_from_dataset)
                filtered_from_prediction = maybe_decode_regression_predictions(
                    [prediction.predictions[curr_dataset][i + 1][0] for i in viable_positions],
                    label_transform,
                )
                labels_from_prediction.extend(filtered_from_prediction)

            if len(labels_from_prediction) <= 1:
                continue

            res_r, res_mse, res_mae = get_cpg_reesults_for_labels(labels_from_dataset, labels_from_prediction, verbose)
            bin_results = [curr_bin, res_r, res_mse, res_mae]
            if verbose:
                print(bin_results, flush=True)
            res.append(bin_results)
        return res
    model = get_fine_tuned_model(
        is_lora,
        num_labels,
        base_model_name,
        model_path,
        for_inference=True,
        model_type=model_type,
        tissue_prompt_config=tissue_prompt_config,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    trainer = get_trainer(
        dataset,
        model,
        tokenizer,
        model_type,
        per_device_train_batch_size=per_device_eval_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_accumulation_steps=eval_accumulation_steps,
        task_level=task_level,
    )
    eval_results = trainer.evaluate()

    del model
    del trainer
    del tokenizer
    return eval_results


def get_cpg_reesults_for_labels(labels_from_dataset, labels_from_prediction, verbose):
    labels_from_dataset = np.array(labels_from_dataset)
    labels_from_prediction = np.array(labels_from_prediction)
    mask = labels_from_dataset != -100
    labels_from_dataset = labels_from_dataset[mask]
    labels_from_prediction = labels_from_prediction[mask]
    mask = ~(np.isnan(labels_from_dataset) | np.isnan(labels_from_prediction))
    labels_from_prediction = labels_from_prediction[mask]
    labels_from_dataset = labels_from_dataset[mask]
    if verbose:
        print("legnth_labels:", len(labels_from_dataset), flush=True)
    res_r = pearsonr_eval.compute(predictions=labels_from_prediction, references=labels_from_dataset)
    res_mse = mse.compute(predictions=labels_from_prediction, references=labels_from_dataset)
    res_mae = mae.compute(predictions=labels_from_prediction, references=labels_from_dataset)
    return res_r, res_mse, res_mae


def perform_tissue_evaluation(cfg: dict):
    verbose = cfg.get("verbose", False)
    paths = cfg["paths"]
    task = cfg["task"]
    model = cfg["model"]
    test_params = cfg["testing_params"]

    test_mode = test_params.get("test_mode", False)
    jump_sample = test_params.get("jump_sample", -1)
    per_device_eval_batch_size = test_params.get("per_device_eval_batch_size", 1)
    eval_accumulation_steps = test_params.get("eval_accumulation_steps", 4)

    dataset_path = paths.get("dataset_path", None)
    variant_file_path = paths.get("variant_file_path", False)
    atlas_dataset_paths = paths.get("atlas_dataset_paths", None)
    target_bigwig_path = paths.get("target_bigwig_path", None)
    atlas_bigwig_paths = paths.get("atlas_bigwig_paths", None)

    sub_task = task["sub_task"]
    use_variant_file = task.get("use_variant_file", False)
    vriant_grouping_method = task.get("vriant_grouping_method", None)
    number_of_bins = task.get("number_of_bins", -1)
    top_rows = task.get("top_rows", -1)
    label_transform = get_task_label_transform(task)

    use_lora = model.get("use_lora", model.get("is_lora", False))
    tissue_prompt_config = model.get("tissue_prompt")
    task_level = model.get("task_level", (tissue_prompt_config or {}).get("task_level", "token"))
    lora_over_finetuned = model.get("lora_over_finetuned", False)
    model_repo = model.get("model_repo", None)
    model_name = model.get("model_name", None)

    if verbose:
        print("performing,", sub_task + " task")
        if sub_task == "atlas_evaluation":
            print("for target bigwig:", target_bigwig_path)
        else:
            print("for dataset:", dataset_path)
        print("with model:", model_repo, model_name)

    if sub_task == "atlas_evaluation":
        if target_bigwig_path is None or atlas_bigwig_paths is None or len(atlas_bigwig_paths) == 0:
            raise ValueError(
                "atlas evaluation now expects paths.target_bigwig_path and paths.atlas_bigwig_paths. "
                "Regenerate the atlas evaluation configs."
            )
        return pd.DataFrame(
            evaluate_atlas_from_bigwigs(
                target_bigwig_path=target_bigwig_path,
                atlas_bigwig_paths=atlas_bigwig_paths,
                number_of_bins=number_of_bins,
                chroms=task.get("chromosomes"),
                top_rows=top_rows,
                test_mode=test_mode,
                jump_sample=jump_sample,
                verbose=verbose,
            )
        )

    if dataset_path is None:
        raise ValueError("missing dataset_path for non-atlas evaluation")

    dataset = load_from_disk(dataset_path)
    selected_dataset = dataset.select(range(top_rows)) if top_rows > 0 else dataset
    if test_mode:
        print("running test mode", flush=True)
        if jump_sample > 0:
            print("using jump sample of:", jump_sample, flush=True)
        selected_dataset = dataset.select(range(0, len(selected_dataset), jump_sample)) if jump_sample > 0 else selected_dataset

    if verbose:
        print("finished loading dataset", flush=True)
        print("with size", len(selected_dataset))

    selected_dataset = apply_label_transform_to_dataset(
        selected_dataset,
        label_transform,
        verbose=verbose,
        dataset_name="evaluation dataset",
    )

    if "tissue_ids" not in selected_dataset.column_names:
        raise ValueError(
            "This config uses tissue prompts, but the dataset has no 'tissue_ids' column. "
            "Recreate the HF datasets after adding tissue-aware extraction."
        )

    if task_level != "token" and (use_variant_file or sub_task == "atlas_evaluation"):
        raise ValueError("Variant-bin and atlas evaluation paths currently assume token-level outputs.")

    variability_positions_dict = {}
    bins_to_use = []
    dataset_labels = None
    if use_variant_file:
        variant_file_dataframe = pd.read_csv(variant_file_path).dropna()

        if verbose:
            print("loaded variant file", flush=True)
        add_std_bins_to_dataframe(number_of_bins, variant_file_dataframe)
        bins_to_use = variant_file_dataframe["std_bin"].unique()
        window_ids_in_variant_file = set(variant_file_dataframe["window_id"].unique())
        selected_dataset = selected_dataset.filter(lambda example: example["window_id"] in window_ids_in_variant_file)
        variability_positions_dict = create_variability_positions_dict(verbose, selected_dataset, bins_to_use, variant_file_dataframe)
        set_of_non_empty_window_ids = create_non_empty_window_ids_set(variability_positions_dict)
        dataset_labels = create_dataset_labels(selected_dataset, variability_positions_dict, bins_to_use)
        if verbose:
            print("finished filtering dataset", flush=True)
            print("with size", len(selected_dataset))
            print("finished creating variability positions dict", flush=True)

    if sub_task == "evaluate_single_checkpoint":
        res = evaluate_checkpoint(
            model_repo,
            model_name,
            use_lora,
            cfg["model"]["num_labels"],
            selected_dataset,
            cfg["model"]["model_type"],
            cfg["paths"]["model_path"],
            cfg["model"]["model_type"],
            variability_positions_dict,
            use_variant_file,
            vriant_grouping_method,
            bins_to_use,
            dataset_labels,
            verbose,
            tissue_prompt_config=tissue_prompt_config,
            task_level=task_level,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
            label_transform=label_transform,
        )
        return res
    elif sub_task == "predict_single_checkpoint":
        model_type = cfg["model"]["model_type"]
        return predict_checkpoint(
            selected_dataset,
            selected_dataset,
            model_type,
            use_lora,
            cfg["model"]["num_labels"],
            model_repo,
            cfg["model"]["model_name"],
            cfg["paths"]["model_path"],
            tissue_prompt_config=tissue_prompt_config,
            task_level=task_level,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
        )
    elif sub_task == "evaluate_multiple_checkpoints":
        results = []
        use_non_lora_first_iter = use_lora
        if lora_over_finetuned:
            use_non_lora_first_iter = False
        for model_path in cfg["paths"]["model_paths"]:
            res = evaluate_checkpoint(
                model_repo,
                model_name,
                use_non_lora_first_iter,
                cfg["model"]["num_labels"],
                selected_dataset,
                cfg["model"]["model_type"],
                model_path,
                cfg["model"]["model_type"],
                variability_positions_dict,
                use_variant_file,
                vriant_grouping_method,
                bins_to_use,
                dataset_labels,
                verbose,
                tissue_prompt_config=tissue_prompt_config,
                task_level=task_level,
                per_device_eval_batch_size=per_device_eval_batch_size,
                eval_accumulation_steps=eval_accumulation_steps,
                label_transform=label_transform,
            )
            use_non_lora_first_iter = use_lora
            results.append(res)
        if use_variant_file:
            final_results = []
            for i in range(len(results)):
                model_results = {}
                keys = []
                for bin_value in results[i]:
                    bin_name = bin_value[0]
                    for val in bin_value[1:]:
                        key = list(val)[0]
                        value = val[key]
                        model_results[bin_name + "_" + key] = value
                        keys.append(bin_name + "_" + key)
                final_results.append(model_results)
            results = final_results
        df = pd.DataFrame(results)
        df["paths"] = cfg["paths"]["model_paths"]
        return df
    else:
        raise ValueError(f"Unknown sub_task: {sub_task}")


def create_non_empty_window_ids_set(variability_positions_dict):
    set_of_non_empty_window_ids = set()
    for window_id in variability_positions_dict:
        for bin in variability_positions_dict[window_id]:
            if len(variability_positions_dict[window_id][bin]):
                set_of_non_empty_window_ids.add(window_id)
    return set_of_non_empty_window_ids


def create_variability_positions_dict(verbose, selected_dataset, bins_to_use, variant_file_dataframe):
    by_window = {wid: g.copy() for wid, g in variant_file_dataframe.groupby("window_id", sort=False)}
    if verbose:
        print("created variants bin based variant dictionary", flush=True)
    variability_positions_dict = {}
    for index in range(len(selected_dataset)):
        if verbose:
            if index % 50 == 0:
                print("mapping viable variants", index, "/", len(selected_dataset), end="\r", flush=True)
        curr_window_id = selected_dataset[index]["window_id"]
        if curr_window_id not in variability_positions_dict:
            variability_positions_dict[curr_window_id] = {}
        if curr_window_id not in by_window:
            continue
        windiw_id_df = by_window[curr_window_id]
        for curr_bin in bins_to_use:
            limited_df = windiw_id_df[windiw_id_df["std_bin"] == curr_bin]
            starts_of_viable = set(limited_df["full_position"].str.split(":").str[1].str.split("-").str[0].astype(int))
            viable_positions = [((x // 6 * 6) - selected_dataset[index]["start"]) // 6 for x in starts_of_viable]
            variability_positions_dict[curr_window_id][curr_bin] = list(sorted(viable_positions))

    if verbose:
        print()
    return variability_positions_dict


def add_std_bins_to_dataframe(number_of_bins, variant_file_dataframe):
    max_val = variant_file_dataframe["std"].max()
    edges = list(np.linspace(0, max_val, number_of_bins + 1))
    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"{edges[i]}-{edges[i + 1]}")
    variant_file_dataframe["std_bin"] = pd.cut(
        variant_file_dataframe["std"],
        bins=edges,
        labels=labels,
        right=True,
        include_lowest=True,
    )


def create_dataset_labels(selected_dataset, variability_positions_dict, bins_to_use):
    dataset_labels = {}
    for dataset_index in range(len(selected_dataset)):
        curr_window_id = selected_dataset[dataset_index]["window_id"]
        if curr_window_id not in dataset_labels:
            dataset_labels[curr_window_id] = {}
        for curr_bin in bins_to_use:
            if curr_bin not in variability_positions_dict[curr_window_id]:
                continue
            viable_positions = variability_positions_dict[curr_window_id][curr_bin]
            filtered_labels_from_dataset = [selected_dataset[dataset_index]["labels"][i + 1] for i in viable_positions]
            dataset_labels[curr_window_id][curr_bin] = filtered_labels_from_dataset
    return dataset_labels


def predict_checkpoint(
    dataset,
    selected_dataset,
    model_type,
    is_lora,
    num_labels,
    model_repo,
    model_name,
    model_path,
    tissue_prompt_config=None,
    task_level="token",
    per_device_eval_batch_size: int = 1024,
    eval_accumulation_steps: int = 128,
):
    base_model_name = model_repo + "/" + model_name

    model = get_fine_tuned_model(
        is_lora,
        num_labels,
        base_model_name,
        model_path,
        for_inference=True,
        model_type=model_type,
        tissue_prompt_config=tissue_prompt_config,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    trainer = get_trainer(
        dataset,
        model,
        tokenizer,
        model_type,
        per_device_train_batch_size=per_device_eval_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_accumulation_steps=eval_accumulation_steps,
        task_level=task_level,
    )
    return trainer.predict(selected_dataset)
