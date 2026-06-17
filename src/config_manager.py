import yaml
import os, sys
import math
# sys.path.insert(0, os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code"))
from datetime import datetime
import os
import yaml
import shutil
import csv

try:
    from utils.label_transform_utils import normalize_label_transform
except ImportError:
    from .utils.label_transform_utils import normalize_label_transform

from constants import (RAW_INPUT_NAME, 
                           CPG_SEPERATING_SITES_TASK_NAME,
                           INTERMEDIATE_INPUT_NAME, 
                           PREPROCESSED_INPUT_NAME, 
                           CPG_TRAINING_TASK_TYPE, 
                           CPG_RETRAINING_TASK_TYPE,
                           CPG_EVALUATION_TASK_TYPE,
                           SAVED_EPOCH_PREFIX,
                           EVALUATE_MULTIPLE_CHCEKPOINTS_SUBTASK_TYPE,
                           BINS_GROUPING_METHOD,
                           DEFAULT_FULL_POSITION_COLUMN_NAME,
                           REGRESSION_ANALYSIS_SYMBOL,
                           BLANK_LABEL_VALUE,
                           NO_PRETRAINING_CONFIG_NAME,
                           STD_VARIABILITY_TYPE,
                           QUANTILE_SEPERATION_TYPE
                           )
# from utils.formatting import (STD_VARIABILITY_TYPE,
#                                   QUANTILE_SEPERATION_TYPE) #TODO: move to constants


CREATE_PROJECT_TASK_NUM = 0
DATA_EXTRQACTION_TASK_NUM = 1
MODEL_TRAINING_TASK_NUM = 2
MODEL_EVALUATION_TASK_NUM = 3
CREATE_PROJECT_HP_OPTIMIZATION_TASK_NUM = 4
CPG_DATA_TYPE = 0

RAW_INPUT_MODE_NUMBER = 0

CPG_EXTRACTION_TASK_TYPE = "cpg_extraction"

RAW_INPUT_MODE_NAME = "raw"

# GENERATOR_CACHE_DIR = "/sci/archive/michall/roeizucker/huggingface_modles_cache/datasets/generator"
GENERATOR_CACHE_DIR = "/home/users/roeizucker/.cache/huggingface/datasets/generator"

# /sci/archive/michall/roeizucker/trained_huggingfacce_models_location
TRAINED_HUGGINGFACE_MODELS_LOCATION = "/home/users/roeizucker/tests/trained_huggingface_models_location"

# HUGGINGFACE_DATASET_BASE_DIR = "/sci/archive/michall/roeizucker/huggingface_datasets_dir/"
HUGGINGFACE_DATASET_BASE_DIR = "/home/users/roeizucker/tests/huggingface_datasets_dir"

# TODO: change so it is not a constant, but a parameter in the create base dict function
ASSEMBLIES =   {
    "HG38": "/home/users/roeizucker/tests/reference_genome/hg38.fa",
    "MM10": "/sci/archive/michall/roeizucker/reference_genome/mm10.fa"
}

# ASSEMBLIES =   {
#     "HG38": "/sci/archive/michall/roeizucker/reference_genome/hg38.fa",
#     "MM10": "/sci/archive/michall/roeizucker/reference_genome/mm10.fa"
# }

DEFAULT_CONFIG_PATH = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/config_default.yaml"
CONFIG_DIR_PATH = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/"
DEFAULT_MULTIPLE_TRAINING_PATH = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/test_without_minus1"
DEFAULT_CREATE_BASIC_EXTRACTION_FILES = '''/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652233_Liver-Hepatocytes-Z000000R3.hg38.bigwig,/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652234_Liver-Hepatocytes-Z000000T3.hg38.bigwig,/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652235_Liver-Hepatocytes-Z0000043Q.hg38.bigwig,/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652236_Liver-Hepatocytes-Z0000044H.hg38.bigwig,/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652237_Liver-Hepatocytes-Z0000044M.hg38.bigwig,/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652238_Liver-Hepatocytes-Z00000431.hg38.bigwig'''
DEFAULT_CREATE_BASIC_EXTRACTION_PATH = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/kaplan_files_chr1/regular_extract"


DEFAULT_CHROMS = [
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
]


# presets:
CONVERT_INTERMEDIATE_FILE_EXTRACTION_TO_DS_EXTRACTION_PRESET = 0
CONVERT_DS_EXTRACTION_TO_MULTIPLE_TRAINING_PRESET = 1
CREATE_MULTIPLE_CONFIGS = 2
CONVERT_DS_EXTRACTION_TO_RETRAINING_PRESET = 3
CONVERT_TRAINING_TO_EVALUATION_PRESET = 4

# TODO: move to constants
# TODO: needs change to parameter
RESULTS_DIR = "/home/users/roeizucker/tests/jupyter_notebooks/Tom_Hope_Project/results"

def _result_row_count(csv_path):
    # Count data rows (exclude header). Returns -1 on read failure.
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return 0
        return max(0, len(rows) - 1)
    except Exception:
        return -1

def _expected_eval_rows(eval_cfg):
    # For eval-multiple-checkpoints, expected rows should match number of model paths in that eval config.
    model_paths = (eval_cfg or {}).get("paths", {}).get("model_paths", []) or []
    if model_paths:
        return len(model_paths)
    return None

def _result_exists(eval_cfg, expected_rows=None):
    task = (eval_cfg or {}).get("task", {})
    analysis = task.get("analysis_name")
    base_suffix = task.get("base_suffix")

    candidates = []

    # new format: results/<base_suffix>/<analysis>_result.csv
    if analysis and base_suffix:
        candidates.append(os.path.join(RESULTS_DIR, base_suffix, f"{analysis}_result.csv"))

    # # fallback to old formats
    # if analysis:
    #     legacy_dir = os.path.join(RESULTS_DIR, analysis)
    #     candidates.append(os.path.join(legacy_dir, "result.csv"))
    #     candidates.append(os.path.join(legacy_dir, "resilt.csv"))
    for p in candidates:
        if not os.path.exists(p):
            continue
        if expected_rows is None:
            return True
        row_count = _result_row_count(p)
        if row_count == expected_rows:
            return True
        else:
            print("#check if deletion required:")
            print("#",p)
    # print("check if deletion required:")
    # print(candidates[0])
    return False

def _yaml_files(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".yaml")
    )

def _is_true_pretrain_dir(path):
    name = os.path.basename(path).lower()
    # Keep true pretrain outputs out of auto-deletion.
    # no_pretraining belongs to retrain variants and should not be treated as pretrain.
    if "no_pretraining" in name:
        return False
    return ("pretrain" in name) and ("_retrain" not in name)

def _project_dirs(config_path):
    cfg = yaml.safe_load(open(config_path))
    p = cfg["params"]
    base_suffix = p["project_suffix"]
    root = p["created_configs_path"]
    if not os.path.isdir(root):
        return []
    return sorted(
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith(base_suffix + "_lr_")
    )

def collect_finished_training_dirs(config_path, include_pretrain=False):
    done_dirs = set()
    blocked_dirs = set()

    for proj_dir in _project_dirs(config_path):
        eval_dir = os.path.join(proj_dir, "eval_configs")
        for yp in _yaml_files(eval_dir):
            cfg = yaml.safe_load(open(yp))
            analysis = cfg.get("task", {}).get("analysis_name")
            model_paths = cfg.get("paths", {}).get("model_paths", []) or []
            expected_rows = _expected_eval_rows(cfg)
            target = done_dirs if _result_exists(cfg, expected_rows=expected_rows) else blocked_dirs
            
            
            for mp in model_paths:
                # model path is usually .../<training_output_dir>/epoch-...
                parent = os.path.dirname(mp)

                if not os.path.isdir(parent):
                    
                    continue
                if _is_true_pretrain_dir(parent):

                    # Pretrain outputs are always manual-delete only.
                    continue
            
                target.add(parent)
    # never delete dirs still referenced by pending evals
    return sorted(done_dirs - blocked_dirs)

def _dir_size_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def print_deletion_plan(config_path, include_pretrain=False,keep_one_alive=False):
    dirs = collect_finished_training_dirs(config_path, include_pretrain=include_pretrain)
    if keep_one_alive:
        with_one_alive = set()
        for parent in dirs:
            for child in os.listdir(parent):
                mp = os.path.join(parent,child)
                if str(child).startswith("epoch"):
                    with_one_alive.add(mp)
            if len([x for x in  os.listdir(parent) if x.startswith("checkpoint")]) > 1:
                with_one_alive.add(os.path.join(parent,min([x for x in  os.listdir(parent) if x.startswith("checkpoint")])))
        for d in with_one_alive:
            print(f"rm -rf {d}")
        return
    total = 0
    print(f"Found {len(dirs)} finished training dirs to delete")
    for d in dirs:
        sz = _dir_size_bytes(d) if os.path.exists(d) else 0
        total += sz
        print(f"rm -rf {d}")
    print(f"Approx bytes (very rough): {total}")
    return dirs

def delete_finished_trainings(config_path, include_pretrain=False):
    dirs = collect_finished_training_dirs(config_path, include_pretrain=include_pretrain)
    for d in dirs:
        if os.path.isdir(d):
            shutil.rmtree(d)
            print("deleted", d)
    print("done")



def _safe_load_yaml(path):
    try:
        return yaml.safe_load(open(path))
    except Exception:
        return None


def _epoch_ckpts(model_dir):
    if not os.path.isdir(model_dir):
        return []
    return [x for x in os.listdir(model_dir) if x.startswith("epoch-")]


def _dataset_dir_ready(path):
    if not path or not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except Exception:
        return False
    if not entries:
        return False
    markers = {"dataset_info.json", "state.json", "data", "dataset_dict.json"}
    if any(m in entries for m in markers):
        return True
    return any(name.endswith(".arrow") or name.endswith(".parquet") for name in entries)


def build_job_name(project_prefix, stage, yaml_path):
    cfg = _safe_load_yaml(yaml_path) or {}
    p = cfg.get("paths", {})
    t = cfg.get("task", {})
    tr = cfg.get("train", {})

    analysis = t.get("analysis_name")

    sample = None
    out_dir = p.get("output_dir")
    if out_dir:
        sample = os.path.basename(out_dir).split("_")[0]
    if not sample and analysis:
        sample = analysis.split("_")[0]
    if not sample:
        sample = os.path.basename(yaml_path).split("_")[0]

    lr = tr.get("learning_rate")
    bs = tr.get("per_device_train_batch_size")
    seq = t.get("seq_size")

    def _fmt_lr(x):
        if x is None:
            return "na"
        try:
            return f"{float(x):.0e}".replace("+0", "").replace("+", "")
        except Exception:
            return str(x)

    prefix = str(project_prefix).replace(" ", "_").replace("/", "-")
    analysis_part = str(analysis) if analysis else "na"

    parts = [   
        str(stage),
        str(sample),
        analysis_part,
        f"lr{_fmt_lr(lr)}",
        f"bs{bs if bs is not None else 'na'}",
        f"s{seq if seq is not None else 'na'}",
    ]
    name = ":".join(parts).replace(" ", "_").replace("/", "-")
    return name[:120]


def _extraction_done(cfg):
    if not cfg:
        return False

    task = cfg.get("task", {})
    paths = cfg.get("paths", {})
    task_type = task.get("type", "")

    if task_type == CPG_SEPERATING_SITES_TASK_NAME:
        out_path = paths.get("per_variant_data_path")
        return bool(out_path and os.path.exists(out_path))

    train_path = paths.get("hf_dataset_train_path")
    test_path = paths.get("hf_dataset_test_path")
    test_size = float(task.get("test_size", 0.0) or 0.0)

    train_ok = _dataset_dir_ready(train_path)
    if test_size == 0:
        return train_ok

    test_ok = _dataset_dir_ready(test_path)
    return train_ok and test_ok


def _training_status(cfg, completed_model_dirs=None):
    """
    returns: 'not_started' | 'started_not_finished' | 'finished' | 'done_deleted'
    """
    if not cfg:
        return "not_started"

    out_dir = cfg.get("paths", {}).get("output_dir")
    expected = int(cfg.get("train", {}).get("num_train_epochs", 0) or 0)
    if expected <= 0:
        expected = 1

    if not out_dir or not os.path.isdir(out_dir):
        if completed_model_dirs and out_dir in completed_model_dirs:
            return "done_deleted"
        return "not_started"

    ckpts = _epoch_ckpts(out_dir)
    if len(ckpts) == 0:
        if completed_model_dirs and out_dir in completed_model_dirs:
            return "done_deleted"
        return "not_started"
    if len(ckpts) < expected:
        return "started_not_finished"
    return "finished"


def _eval_done(eval_cfg):
    return _result_exists(eval_cfg, expected_rows=_expected_eval_rows(eval_cfg))


def _build_eval_completion_index(eval_yaml_paths):
    completed = set()
    for yp in eval_yaml_paths:
        cfg = _safe_load_yaml(yp)
        if not cfg or not _eval_done(cfg):
            continue
        for mp in (cfg.get("paths", {}).get("model_paths", []) or []):
            parent = os.path.dirname(mp)
            if not parent:
                continue
            if "_retrain" not in os.path.basename(parent):
                continue
            completed.add(parent)
    return completed


def _build_expected_epochs_map(training_yaml_paths):
    expected = {}
    for yp in training_yaml_paths:
        cfg = _safe_load_yaml(yp)
        if not cfg:
            continue
        out_dir = cfg.get("paths", {}).get("output_dir")
        n = cfg.get("train", {}).get("num_train_epochs")
        if out_dir and n is not None:
            expected[out_dir] = int(n)
    return expected


def _all_eval_models_ready(eval_cfg, expected_epochs_by_output_dir):
    model_paths = (eval_cfg or {}).get("paths", {}).get("model_paths", []) or []
    if not model_paths:
        return False
    if not all(os.path.exists(p) for p in model_paths):
        return False

    retrain_dirs = {os.path.dirname(p) for p in model_paths if os.path.basename(p).startswith("epoch-")}
    retrain_dirs = {d for d in retrain_dirs if "_retrain" in os.path.basename(d)}
    if not retrain_dirs:
        return False

    for d in retrain_dirs:
        required = expected_epochs_by_output_dir.get(d)
        if required is None:
            return False
        if len(_epoch_ckpts(d)) < required:
            return False
    return True


def _eval_can_run(eval_cfg, expected_epochs_by_output_dir):
    return (not _eval_done(eval_cfg)) and _all_eval_models_ready(eval_cfg, expected_epochs_by_output_dir)


def _build_train_cmd(cfg, yaml_path, stage, project_prefix, train_script, is_for_script,sbatch_command_prefix):
    bs = int((cfg or {}).get("train", {}).get("per_device_train_batch_size", 2))
    gpus = max(2, min(math.ceil(bs / 3), 8))
    job_name = build_job_name(project_prefix, stage, yaml_path)
    
    if sbatch_command_prefix is None:
        sbatch_command_prefix = f"sbatch --mem=45g -c15 --gres=gg:g4:{gpus} --time=5-23 --killable --requeue"
    if is_for_script:
        return (
            f'sbatch --mem=45g -c15 --gres=gg:g4:{gpus} --time=5-23 -A tomhope --job-name="{job_name}" '
            f'--wrap="{train_script} {yaml_path}"'
        )
    
    return (
        
        f'{sbatch_command_prefix} --job-name="{job_name}" '
        f'--wrap="{train_script} {yaml_path}"'
    )


def _build_eval_cmd(yaml_path, project_prefix, eval_script, is_for_script,sbatch_command_prefix):
    job_name = build_job_name(project_prefix, "eval", yaml_path)
    if sbatch_command_prefix is None:
        sbatch_command_prefix = f"sbatch --mem=45g -c15 --gres=gg:g4:1 --time=5-23 --killable --requeue"
    if is_for_script:
        return (
            f'sbatch --mem=45g -c8 --gres=gg:g4:1 --time=5-23 -A tomhope --job-name="{job_name}" '
            f'--wrap="{eval_script} {yaml_path}"'
        )
    return (
        f'{sbatch_command_prefix} --job-name="{job_name}" '
        f'--wrap="{eval_script} {yaml_path}"'
    )


def _print_extract_pending(yaml_paths, extract_script):
    for p in yaml_paths:
        cfg = _safe_load_yaml(p)
        if not _extraction_done(cfg):
            print(f"{extract_script} {p}")


def _print_train_pending_split(yaml_paths, stage, project_prefix, train_script, is_for_script,
                               completed_model_dirs=None, ignore=None,sbatch_command_prefix=None):
    not_started_cmds = []
    started_not_finished_cmds = []
    ignore = set(ignore or [])

    for p in yaml_paths:
        cfg = _safe_load_yaml(p)
        status = _training_status(cfg, completed_model_dirs=completed_model_dirs)
        if status in ("finished", "done_deleted"):
            continue

        cmd = _build_train_cmd(cfg, p, stage, project_prefix, train_script, is_for_script,sbatch_command_prefix)
        job_name = build_job_name(project_prefix, stage, p)
        if job_name in ignore:
            continue
        if status == "started_not_finished":
            started_not_finished_cmds.append(cmd)
        else:
            not_started_cmds.append(cmd)

    print("# training not started")
    for c in not_started_cmds:
        if is_for_script:
            print("wait_for_it")
            print("echo", c.split('--job-name="')[1].split('"')[0])
        print(c)

    print("# training started but not finished")
    for c in started_not_finished_cmds:
        if is_for_script:
            print("wait_for_it")
            print("echo", c.split('--job-name="')[1].split('"')[0])
        print(c)


def _print_eval_pending(yaml_paths, expected_epochs_by_output_dir, project_prefix, eval_script,
                        is_for_script, ignore=None,sbatch_command_prefix=None):
    ignore = set(ignore or [])
    for p in yaml_paths:
        cfg = _safe_load_yaml(p)
        job_name = build_job_name(project_prefix, "eval", p)
        if job_name in ignore:
            continue

        if _eval_can_run(cfg, expected_epochs_by_output_dir):
            if is_for_script:
                print("wait_for_it")
                print("echo", job_name)
            print(_build_eval_cmd(p, project_prefix, eval_script, is_for_script,sbatch_command_prefix))


def _project_dirs_for_combo(configs_base_dir, base_suffix, lr, batch_size):
    prefix = f"{base_suffix}_lr_{lr}_bs_{batch_size}"
    if not os.path.isdir(configs_base_dir):
        return []
    return sorted(
        os.path.join(configs_base_dir, d)
        for d in os.listdir(configs_base_dir)
        if os.path.isdir(os.path.join(configs_base_dir, d)) and d.startswith(prefix)
    )


def print_commands_for_roject_config(project_dto, lr, batch_size, extract_script, train_script,
                                     eval_script, is_for_script=False, ignore=None,sbatch_command_prefix=None):
    base_suffix = project_dto["project_suffix"]
    configs_base_dir = project_dto["created_configs_path"]
    ignore = set(ignore or [])

    project_dirs = _project_dirs_for_combo(configs_base_dir, base_suffix, lr, batch_size)

    for created_configs_path in project_dirs:
        variability_extraction = os.path.join(created_configs_path, "variability_extraction")
        pretrain_extraction = os.path.join(created_configs_path, "pretrain_extraction")
        pretrain_training = os.path.join(created_configs_path, "pretrain_training")
        lora_pretrain_training = os.path.join(created_configs_path, "lora_pretrain_training")
        retrain_extraction = os.path.join(created_configs_path, "retrain_extraction")
        retrain_training = os.path.join(created_configs_path, "retrain_training")
        eval_configs = os.path.join(created_configs_path, "eval_configs")

        print(f"\n# project: {created_configs_path}")

        _print_extract_pending(_yaml_files(variability_extraction), extract_script)

        print("# run extraction:")
        _print_extract_pending(_yaml_files(pretrain_extraction), extract_script)
        
        # return
        completed_model_dirs = _build_eval_completion_index(_yaml_files(eval_configs))

        print("# pretraining run")
        _print_train_pending_split(
            _yaml_files(pretrain_training),
            stage="pre",
            project_prefix=base_suffix,
            train_script=train_script,
            is_for_script=is_for_script,
            completed_model_dirs=completed_model_dirs,
            ignore=ignore,
            sbatch_command_prefix=sbatch_command_prefix
        )

        if os.path.isdir(lora_pretrain_training):
            print("# lora pretraining run")
            _print_train_pending_split(
                _yaml_files(lora_pretrain_training),
                stage="pre_lora",
                project_prefix=base_suffix,
                train_script=train_script,
                is_for_script=is_for_script,
                completed_model_dirs=completed_model_dirs,
                ignore=ignore,
                sbatch_command_prefix=sbatch_command_prefix
            )

        print("# run extraction:")
        _print_extract_pending(_yaml_files(retrain_extraction), extract_script)

        print("# run retrain")
        _print_train_pending_split(
            _yaml_files(retrain_training),
            stage="ret",
            project_prefix=base_suffix,
            train_script=train_script,
            is_for_script=is_for_script,
            completed_model_dirs=completed_model_dirs,
            ignore=ignore,
            sbatch_command_prefix=sbatch_command_prefix
        )

        train_yaml_paths = (
            _yaml_files(pretrain_training)
            + _yaml_files(lora_pretrain_training)
            + _yaml_files(retrain_training)
        )
        expected_epochs_by_output_dir = _build_expected_epochs_map(train_yaml_paths)

        print("##### run eval #####")
        _print_eval_pending(
            _yaml_files(eval_configs),
            expected_epochs_by_output_dir,
            project_prefix=base_suffix,
            eval_script=eval_script,
            is_for_script=is_for_script,
            ignore=ignore,
            sbatch_command_prefix=sbatch_command_prefix
        )


def create_base_dictionary():
    dic = {}
    dic["paths"] = {
    "assemblies" : ASSEMBLIES,
    "generetor_cache_dir" : GENERATOR_CACHE_DIR
    }
    dic["task"] = {}
    dic["testing_params"] = {
        "test_mode" : False
    }
    dic["random_state"] = 42
    dic["verbose"] = True
    return dic


def convert_ds_extraction_configs_to_leave_one_out_training(multiple_configs_directory_path, base_name):
    configs = []
    for file in os.listdir(multiple_configs_directory_path):
        if file.endswith(".yaml"):
            config_file = os.path.join(multiple_configs_directory_path, file)
            data = yaml.safe_load(open(config_file))
            task_type = data["task"].get("type",None)
            if task_type != CPG_EXTRACTION_TASK_TYPE:
                continue
            configs.append(data)
    for config in configs:
        analysis_name = config["paths"]["hf_dataset_train_path"].split("/")[-1].replace("_train","")
        curr_configs = [
                    curr_config for curr_config in configs if analysis_name not in curr_config["paths"]["hf_dataset_train_path"]
                    ]
                
        result_project_name = f"{base_name}_{analysis_name}"
        result_model_path = os.path.join(TRAINED_HUGGINGFACE_MODELS_LOCATION,f"{result_project_name}_pretrain")
        curr_datasets_paths = [conf["paths"]["hf_dataset_train_path"] for conf in curr_configs]
        tokenizer_name = config["task"]["tokenizer_name"]
        base_dict = create_base_dictionary()
        create_regression_analysis_train_task(analysis_name, result_model_path, curr_datasets_paths, tokenizer_name, base_dict)
        with open(os.path.join(multiple_configs_directory_path,analysis_name + "_train.yaml"), 'w') as file:
            yaml.dump(base_dict, file, default_flow_style=False, sort_keys=False)
                
            # print(cfg)
    print("created", len(configs),"files in ",multiple_configs_directory_path)

def create_regression_analysis_train_task(analysis_name, result_model_path, datasets_paths, tokenizer_name, base_dict):
    task = base_dict["task"]
    task["task_type"] ="cpg_training"
    task["analysis_name"] =  analysis_name
    # NOTE: removed this
    # task["sub_task"] ="evaluate_multiple_checkpoints"
    task["sequence_col"] = "sequence"
    task["label_col"] = "labels"
    task["top_rows"] = -1
    # TODO: add save_stratagy

    model = create_model_config(REGRESSION_ANALYSIS_SYMBOL,True,True,1,tokenizer_name.split("/")[1],tokenizer_name.split("/")[0])
    base_dict["model"] = model

    train = {}
    base_dict["train"] = train
    train["load_best_model_at_end"] = False
    train["num_train_epochs"] = 3
    train["per_device_train_batch_size"] = 2
    train["per_device_eval_batch_size"] = 2
    train["learning_rate"] = 1e-5
    train["metric_for_best_model"]= "mse"

    # TODO: add this as param
    train["max_grad_norm"] = 1
    base_dict["paths"]["train_dataset_path"] = datasets_paths
    base_dict["paths"]["output_dir"] = result_model_path

def convert_intermediate_file_extraction_to_ds_extraction(source_path, destination_path):
    cfg = yaml.safe_load(open(source_path))
            
    cfg["task"]["output_intermediate_data"] = False
    cfg["task"]["output_hf_dataset"] = True
    cfg["task"]["input_mode"] = INTERMEDIATE_INPUT_NAME 
    with open(destination_path, 'w') as file:
        yaml.dump(cfg, file, default_flow_style=False, sort_keys=False)
    print(f"sucessfully rewritten {destination_path}")

def create_data_extraction_config_dict(base_dict,test_size = 0.00,window_split = None,
                    data_type=None,input_mode = None,created_datasets_base_name=None,raw_data_path = None,tokenizer_name = None,
                    use_variant_filtering=False,variant_filtering_upper_bound=-1,variant_filtering_lower_bound=-1,variant_file_path=None,
                    replace_min1=False,chromosomes=None,train_test_seperation=None,seq_size=5400,dataset_base_dir=HUGGINGFACE_DATASET_BASE_DIR,
                    override_dataset=False):
    if window_split is None:
        window_split = int(input("create window split? 1/0 (yes/no)"))
        
    if window_split == 1:
        raise NotImplementedError("this option was not implemented yet")
    elif window_split == 0:
        base_dict["task"]["type"] = CPG_EXTRACTION_TASK_TYPE
        base_dict["task"]["assembly"] = "HG38"
        base_dict["task"]["value_column"] = "methyl_rate"
        if data_type is None:
            print(f'''what is the data type?
cpg data: {CPG_DATA_TYPE}''')
            data_type = int(input())
        if data_type == CPG_DATA_TYPE:
                cpg_data_extraction(base_dict,None,input_mode,created_datasets_base_name,raw_data_path=raw_data_path,dataset_base_dir=dataset_base_dir)
        else:
            raise NotImplementedError()
        # TODO: change this to function that asks for parameters
        base_dict["task"]["seq_size"] = seq_size
        base_dict["task"]["test_size"] = test_size
        base_dict["task"]["blank_label"] = BLANK_LABEL_VALUE
        base_dict["task"]["use_fasta"] = True
        base_dict["task"]["shuffle"]= True
        base_dict["task"]["chromosomes"] = chromosomes
        if tokenizer_name is None:
            tokenizer_name = ask_for_tokenizer_name()
        base_dict["task"]["tokenizer_name"] = tokenizer_name
        base_dict["task"]["output_preprocessed_data"] =  False
        base_dict["task"]["output_intermediate_data"] = True
        base_dict["task"]["output_hf_dataset"] = True
        base_dict["task"]["window_type"] =  "standart_no_overlap"
        base_dict["task"]["override_dataset"] = override_dataset
        if train_test_seperation is None:
            base_dict["task"]["train_test_seperation"] = "random_sample_filtration"
        else:
            base_dict["task"]["train_test_seperation"] = train_test_seperation
        base_dict["task"]["clear_generator_cache"] = True
        base_dict["task"]["replace_min1"] = replace_min1

        variant_filtering = {}
        variant_filtering["use_variant_filtering"] = use_variant_filtering
        variant_filtering["variant_filtering_upper_bound"] = variant_filtering_upper_bound
        variant_filtering["variant_filtering_lower_bound"] = variant_filtering_lower_bound
        if use_variant_filtering:
            base_dict["paths"]["variant_file_path"] = variant_file_path

        base_dict["variant_filtering"] = variant_filtering
        
    else:
        raise ValueError("Incorrect Value")

def ask_for_tokenizer_name(default = "InstaDeepAI/nucleotide-transformer-500m-1000g"):
    tokenizer_name = input("Enter tpkenizer name, enter nothing for " + default)
    if len(tokenizer_name) == 0 or tokenizer_name == "\n":
                # TODO: not working
        tokenizer_name = default
    return tokenizer_name

def cpg_data_extraction(base_dict,temp_prefix = None,input_mode = None,created_datasets_base_name=None,raw_data_path=None,dataset_base_dir=HUGGINGFACE_DATASET_BASE_DIR):
    if temp_prefix is None:
        temp_prefix = str(datetime.now()).replace(" ","")
    if input_mode is None:
        print(f'''What is the input mode?
    raw: {RAW_INPUT_MODE_NUMBER}''',flush=True)
        input_mode = int(input())
    if input_mode == RAW_INPUT_MODE_NUMBER:
        if raw_data_path is None:
            raw_data_path = input("please enter raw data path: ")
        base_dict["task"]["input_mode"] = RAW_INPUT_NAME
        base_dict["paths"]["raw_data_path"] = raw_data_path
        # NOTE: if I want to add costume intermidate paths and the like, do it here
        base_dict["paths"]["intermediate_train_data_path"] = os.path.join(dataset_base_dir,f"temp_train{temp_prefix}.csv")
        base_dict["paths"]["intermediate_test_data_path"] =  os.path.join(dataset_base_dir,f"temp_test{temp_prefix}.csv")
        if created_datasets_base_name is None:
            created_datasets_base_name = input("please enter create datasets base name")
        base_dict["paths"]["hf_dataset_train_path"] = os.path.join(dataset_base_dir,created_datasets_base_name + "_train")
        base_dict["paths"]["hf_dataset_test_path"] = os.path.join(dataset_base_dir,created_datasets_base_name + "_test")
    else:
        raise NotImplementedError()



# TODO: remove variant_file_path parameter
def create_multiple_basic_cpg_extraction_configs(bigwig_files_paths, names,created_configs_path,test_size = 0.2,
    created_configs_path_name_suffix="",created_datasets_base_name_suffix="",tokenizer_name="InstaDeepAI/nucleotide-transformer-500m-1000g",
    use_variant_filtering=False,variant_filtering_upper_bound=-1,variant_filtering_lower_bound=-1,variant_file_path=None,replace_min1=False,
    created_datasets_variability_base_name_suffix=None,variability_extraction_config_path=None,chromosomes=None,
    train_test_seperation=None,seq_size=None,dataset_base_dir=HUGGINGFACE_DATASET_BASE_DIR,override_dataset=False):

    # print("# run extraction:")
    for name,file in zip(names,bigwig_files_paths):
        base_dict = create_base_dictionary()
        variability_file_path =  os.path.join(dataset_base_dir,name + created_datasets_variability_base_name_suffix + ".csv")
        create_data_extraction_config_dict(base_dict,test_size = test_size,window_split = 0,data_type=CPG_DATA_TYPE,
            input_mode = RAW_INPUT_MODE_NUMBER,created_datasets_base_name=name + created_datasets_base_name_suffix,raw_data_path=file,
            tokenizer_name = tokenizer_name,use_variant_filtering=use_variant_filtering,
            variant_filtering_upper_bound=variant_filtering_upper_bound,variant_filtering_lower_bound=variant_filtering_lower_bound,
            variant_file_path=variability_file_path,replace_min1=replace_min1,chromosomes=chromosomes,
            train_test_seperation=train_test_seperation,seq_size=seq_size,dataset_base_dir=dataset_base_dir,override_dataset=override_dataset)
        path = os.path.join(created_configs_path,name + created_configs_path_name_suffix + ".yaml")
        # print("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_data_extraction_params.sh",path)
        if os.path.exists(path):
            continue
        with open(path, 'w') as file:
            yaml.dump(base_dict, file, default_flow_style=False, sort_keys=False)


def convert_ds_extraction_to_retraining_configs(multiple_configs_directory_path, extraction_path,dataset_base_dir=HUGGINGFACE_DATASET_BASE_DIR):
    configs = []
    for file in os.listdir(multiple_configs_directory_path):
        if file.endswith(".yaml"):
            config_file = os.path.join(multiple_configs_directory_path, file)
            data = yaml.safe_load(open(config_file))
            task_type = data["task"].get("task_type",None)
            if task_type != CPG_TRAINING_TASK_TYPE:
                continue
            configs.append(data)
        # create_regression_analysis_train_task(analysis_name, result_model_path, curr_datasets_paths, tokenizer_name, base_dict)
    train_path_base = dataset_base_dir
    for curr_config in configs:
            # print(curr_config["task"])
        base_model_path = curr_config["paths"]["output_dir"]
        old_analysis_name = curr_config["task"]["analysis_name"]
        train_path = os.path.join(train_path_base,old_analysis_name + "_train")
        curr_config["paths"]["train_dataset_path"] = [train_path.replace("_train","_basic_train")]
        curr_config["paths"]["trained_model_path"] = base_model_path  + "/checkpoint-105458"
            # TODO: this is specific, change that!
        curr_config["paths"]["output_dir"] = base_model_path.replace("pretrain","retrain")
        curr_config["task"]["task_type"] = CPG_RETRAINING_TASK_TYPE
        curr_config["task"]["analysis_name"]+="_retrain"
        curr_config["train"]["save_stratagy"] = "epoch"
            # DEFAULT_CREATE_BASIC_EXTRACTION_PATH
        path = os.path.join(extraction_path,old_analysis_name + "_basic_retrain.yaml")
            # path = os.path.join(DEFAULT_CREATE_BASIC_EXTRACTION_PATH,name + "_basic.yaml")
        if not os.path.exists:
            with open(path, 'w') as file:
                yaml.dump(curr_config, file, default_flow_style=False, sort_keys=False)

def get_files_names_from_file_list(sep_char, file_list_input):
    files = file_list_input.split(sep_char)
    names = [x.split("/")[-1].replace(".hg38.bigwig","") for x in file_list_input.split(sep_char)]
    return files,names


def apply_label_transform_to_config_directory(config_dir, label_transform):
    normalized = normalize_label_transform(label_transform)
    if not os.path.isdir(config_dir):
        return
    for config_name in os.listdir(config_dir):
        if not config_name.endswith('.yaml'):
            continue
        config_path = os.path.join(config_dir, config_name)
        with open(config_path, 'r') as handle:
            config = yaml.safe_load(handle) or {}
        task_config = config.setdefault('task', {})
        task_config['label_transform'] = normalized
        with open(config_path, 'w') as handle:
            yaml.dump(config, handle, default_flow_style=False, sort_keys=False)


def create_model_config(model_type, use_lora, freeze_model, num_labels, model_name, model_repo):
    model_config = {}
    model_config["model_name"] = model_name
    model_config["model_repo"] = model_repo
    model_config["model_type"] = model_type
    model_config["blank_label"] = BLANK_LABEL_VALUE
    model_config["use_lora"] = use_lora
    model_config["freeze_mode"] = freeze_model
    model_config["num_labels"] = num_labels
    return model_config

def change_base_dict_to_training_dict(base_dict, tokenizer_name, name_suffix, base_model_location, task_type, 
        model_type, use_lora, freeze_model, num_labels, load_best_model_at_end, num_train_epoch, 
        per_device_train_batch_size, per_device_eval_batch_size, learning_rate, metric_for_best_model, save_stratagy, 
        add_epoch_end_save_callback, save_at_end, continue_from_last, curr_base_name, 
        train_dataset_path,number_of_steps = None, save_total_limit = None,trained_model_path=None,top_rows = -1,
        max_grad_norm = None,lora_over_finetuned=False,load_dataset_to_memory=False,min_number_of_cpg_sites=-1,OVERRITE_BATCH_SIZE=None,add_epoch_end_prediction=True,eval_dataset_path=None):
    base_dict["paths"]["train_dataset_path"] = train_dataset_path
    base_dict["paths"]["output_dir"] = os.path.join(base_model_location,curr_base_name + name_suffix)
    if trained_model_path is not None:
        base_dict["paths"]["trained_model_path"] = trained_model_path

    if eval_dataset_path is not None:
        base_dict["paths"]["eval_dataset_path"] = eval_dataset_path

    base_dict["task"]["task_type"] = task_type
    base_dict["task"]["analysis_name"] = curr_base_name + name_suffix
    base_dict["task"]["top_rows"] = top_rows

    base_dict["testing_params"] = {}
    base_dict["testing_params"]["test_mode"] = False

    model_name = tokenizer_name.split("/")[1]
    model_repo = tokenizer_name.split("/")[0]
    base_dict["model"] = create_model_config(model_type, use_lora, freeze_model, num_labels, model_name, model_repo)

    base_dict["train"] = {}
    base_dict["train"]["load_best_model_at_end"] = load_best_model_at_end
    base_dict["train"]["num_train_epochs"] = num_train_epoch
    base_dict["train"]["per_device_train_batch_size"] = per_device_train_batch_size
    base_dict["train"]["per_device_eval_batch_size"] = per_device_eval_batch_size
    if OVERRITE_BATCH_SIZE is not None:
        base_dict["train"]["per_device_train_batch_size"] = OVERRITE_BATCH_SIZE
        base_dict["train"]["per_device_eval_batch_size"] = OVERRITE_BATCH_SIZE
    base_dict["train"]["learning_rate"] = learning_rate
    base_dict["train"]["metric_for_best_model"]=  metric_for_best_model
    base_dict["train"]["save_stratagy"] = save_stratagy
    base_dict["train"]["lora_over_finetuned"] = lora_over_finetuned
    base_dict["train"]["load_dataset_to_memory"] = load_dataset_to_memory
    base_dict["train"]["min_number_of_cpg_sites"] = min_number_of_cpg_sites
    # TODO : add max_grad_norm as param
    if max_grad_norm is not None:
        base_dict["train"]["max_grad_norm"] = max_grad_norm
    if number_of_steps is not None:
        base_dict["train"]["number_of_steps"] = number_of_steps
            
    if save_total_limit is not None:
        base_dict["train"]["save_total_limit"] = save_total_limit
    base_dict["train"]["add_epoch_end_save_callback"] = add_epoch_end_save_callback
    base_dict["train"]["add_epoch_end_prediction"] = add_epoch_end_prediction
    base_dict["train"]["save_at_end"] = save_at_end
    base_dict["train"]["continue_from_last"] = continue_from_last



def create_project_config(project_suffix, bigwig_files, names, created_configs_path, tokenizer_name, dataset_base_dir, 
    base_model_location, model_type, use_lora, freeze_model, num_labels, load_best_model_at_end, num_train_epoch, num_pretrain_epoch,
    per_device_train_batch_size, per_device_eval_batch_size, learning_rate, metric_for_best_model, save_stratagy,
    number_of_steps, save_total_limit, add_epoch_end_save_callback, save_at_end, continue_from_last, 
    use_variant_filtering, pretraining_variant_filtering_upper_bound, pretraining_variant_filtering_lower_bound,retraining_variant_filtering_upper_bound,retraining_variant_filtering_lower_bound, max_grad_norm=None, 
    variant_file_path=None,top_rows=-1,replace_min1=True,chromosomes=None,datasets_suffix=None,
    train_test_seperation=None,seq_size=None,number_of_bins=10,base_suffix="",test_size=0.2,load_dataset_to_memory=False,
    override_dataset=False,OVERRITE_BATCH_SIZE=None,min_number_of_cpg_sites=-1,label_transform="none"):
    label_transform = normalize_label_transform(label_transform)
    if datasets_suffix is None:
        datasets_suffix = project_suffix    
    if not os.path.exists(created_configs_path):
        os.mkdir(created_configs_path)
    pretrain_extraction_config_path = os.path.join(created_configs_path,"pretrain_extraction")
    if not os.path.exists(pretrain_extraction_config_path):
        os.mkdir(pretrain_extraction_config_path)
    variability_extraction_config_path = os.path.join(created_configs_path,"variability_extraction")
    if not os.path.exists(variability_extraction_config_path):
        os.mkdir(variability_extraction_config_path)

    if chromosomes is None:
        chromosomes = DEFAULT_CHROMS
        
    created_datasets_variability_base_name_suffix="_per_varaint_variability" + datasets_suffix
    create_variability_files_configs(bigwig_files, names, chromosomes, seq_size, variability_extraction_config_path, 
                                     created_datasets_variability_base_name_suffix,dataset_base_dir)
        

    created_datasets_pretrain_base_name_suffix="_pretrain" + datasets_suffix
    # Pretrain extraction should keep all variant labels in the dataset while
    # still carrying the variability file path for downstream atlas steps.
    create_multiple_basic_cpg_extraction_configs(bigwig_files, names, pretrain_extraction_config_path, 
        created_configs_path_name_suffix="_pretrain_data_extraction" + project_suffix,test_size=0,
        created_datasets_base_name_suffix=created_datasets_pretrain_base_name_suffix,tokenizer_name=tokenizer_name,
        use_variant_filtering=use_variant_filtering,variant_filtering_upper_bound=pretraining_variant_filtering_upper_bound,
        variant_filtering_lower_bound=pretraining_variant_filtering_lower_bound,variant_file_path=variant_file_path,replace_min1=replace_min1,
        created_datasets_variability_base_name_suffix=created_datasets_variability_base_name_suffix,
        variability_extraction_config_path=variability_extraction_config_path,chromosomes=chromosomes,
        train_test_seperation=train_test_seperation,seq_size=seq_size,dataset_base_dir=dataset_base_dir,override_dataset=override_dataset)

    pretrain_training_config_path = os.path.join(created_configs_path,"pretrain_training")
    pretrain_name_suffix = "_pretrain" + project_suffix
    created_configs_path_name_suffix="_pretrain_training" + project_suffix



    create_pretrain_training_configs_for_project(names, tokenizer_name, created_datasets_pretrain_base_name_suffix, pretrain_name_suffix, 
            created_configs_path_name_suffix, dataset_base_dir, base_model_location, CPG_TRAINING_TASK_TYPE, model_type, False, 
            False, num_labels, load_best_model_at_end, num_pretrain_epoch, per_device_train_batch_size, per_device_eval_batch_size, 
            learning_rate, metric_for_best_model, save_stratagy, number_of_steps, save_total_limit, True, save_at_end, 
            continue_from_last, pretrain_training_config_path,top_rows,max_grad_norm,load_dataset_to_memory,OVERRITE_BATCH_SIZE,min_number_of_cpg_sites)
    
    if use_lora:
        lora_pretrain_training_config_path = os.path.join(created_configs_path,"lora_pretrain_training")
        lora_pretrain_name_suffix = "_lora_pretrain" + project_suffix
        lora_created_configs_path_name_suffix="_lora_pretrain_training" + project_suffix

        create_pretrain_training_configs_for_project(names, tokenizer_name, created_datasets_pretrain_base_name_suffix, lora_pretrain_name_suffix, 
                lora_created_configs_path_name_suffix, dataset_base_dir, base_model_location, CPG_TRAINING_TASK_TYPE, model_type, True, 
                freeze_model, num_labels, load_best_model_at_end, num_pretrain_epoch, per_device_train_batch_size, per_device_eval_batch_size, 
                learning_rate, metric_for_best_model, save_stratagy, number_of_steps, save_total_limit, True, save_at_end, 
                continue_from_last, lora_pretrain_training_config_path,top_rows,max_grad_norm,load_dataset_to_memory,OVERRITE_BATCH_SIZE,min_number_of_cpg_sites)

    retrain_extraction_config_path = os.path.join(created_configs_path,"retrain_extraction")
    if not os.path.exists(retrain_extraction_config_path):
        os.mkdir(retrain_extraction_config_path)
    created_datasets_retrain_base_name_suffix="_retrain" + datasets_suffix
    create_multiple_basic_cpg_extraction_configs(bigwig_files, names, retrain_extraction_config_path, 
        created_configs_path_name_suffix="_retrain_data_extraction" + project_suffix,test_size=test_size,
        created_datasets_base_name_suffix=created_datasets_retrain_base_name_suffix,
        use_variant_filtering=use_variant_filtering,variant_filtering_upper_bound=retraining_variant_filtering_upper_bound,
        variant_filtering_lower_bound=retraining_variant_filtering_lower_bound,
        tokenizer_name=tokenizer_name,created_datasets_variability_base_name_suffix=created_datasets_variability_base_name_suffix,
        variability_extraction_config_path=variability_extraction_config_path,chromosomes=chromosomes,
        # retraining_variant_filtering_upper_bound,retraining_variant_filtering_lower_bound
        train_test_seperation=train_test_seperation,seq_size=seq_size,dataset_base_dir=dataset_base_dir,override_dataset=override_dataset)
        
    retrain_training_config_path = os.path.join(created_configs_path,"retrain_training")
    if not os.path.exists(retrain_training_config_path):
        os.mkdir(retrain_training_config_path)
    retrain_name_suffix = "_retrain" + project_suffix
    created_configs_path_name_suffix="_retrain_training" + project_suffix #TODO: why do they need to be different?
    create_retrain_training_configs_for_project(names, tokenizer_name, pretrain_name_suffix, created_configs_path_name_suffix, 
            dataset_base_dir, base_model_location, model_type, False, False, num_labels, load_best_model_at_end,
            num_train_epoch, per_device_train_batch_size, per_device_eval_batch_size, learning_rate, metric_for_best_model, save_stratagy, 
            number_of_steps, save_total_limit, False, save_at_end, continue_from_last, 
            created_datasets_retrain_base_name_suffix, retrain_training_config_path, retrain_name_suffix,top_rows,max_grad_norm,False,load_dataset_to_memory,OVERRITE_BATCH_SIZE,min_number_of_cpg_sites)
    

    # print("##### this is lora #####")
    retrain_lora_name_suffix = "_retrain_lora" + project_suffix
    created_configs_lora_path_name_suffix="_retrain_training_lora" + project_suffix #TODO: why do they need to be different?
    # TODO: add lora_over_finetuned parameter
    if use_lora:
        create_retrain_training_configs_for_project(names, tokenizer_name, pretrain_name_suffix, created_configs_lora_path_name_suffix, 
                dataset_base_dir, base_model_location, model_type, True, freeze_model, num_labels, load_best_model_at_end,
                num_train_epoch, per_device_train_batch_size, per_device_eval_batch_size, learning_rate, metric_for_best_model, save_stratagy, 
                number_of_steps, save_total_limit, False, save_at_end, continue_from_last, 
                created_datasets_retrain_base_name_suffix, retrain_training_config_path, retrain_lora_name_suffix,top_rows,max_grad_norm,
                lora_over_finetuned=True,load_dataset_to_memory=load_dataset_to_memory,OVERRITE_BATCH_SIZE=OVERRITE_BATCH_SIZE,
                min_number_of_cpg_sites=min_number_of_cpg_sites)
        
        # does lora retraining over lora pretrained
        retrain_lora_over_lora_name_suffix = "_retrain_lora_over_lora" + project_suffix
        created_configs_lora_over_lora_path_name_suffix="_retrain_training_lora_over_lora" + project_suffix #TODO: why do they need to be different?
        create_retrain_training_configs_for_project(names, tokenizer_name, lora_pretrain_name_suffix, created_configs_lora_over_lora_path_name_suffix, 
                dataset_base_dir, base_model_location, model_type, True, freeze_model, num_labels, load_best_model_at_end,
                num_train_epoch, per_device_train_batch_size, per_device_eval_batch_size, learning_rate, metric_for_best_model, save_stratagy, 
                number_of_steps, save_total_limit, False, save_at_end, continue_from_last, 
                created_datasets_retrain_base_name_suffix, retrain_training_config_path, retrain_lora_over_lora_name_suffix,top_rows,max_grad_norm,
                lora_over_finetuned=False,load_dataset_to_memory=load_dataset_to_memory,OVERRITE_BATCH_SIZE=OVERRITE_BATCH_SIZE,
                min_number_of_cpg_sites=min_number_of_cpg_sites)
                # yaml.dump(base_dict, sys.stdout)
        # eval section:
    # print("##### run eval #####")
    created_configs_path_name_suffix="_eval" + project_suffix
    eval_name_suffix = created_configs_path_name_suffix

    # TODO: I added the change "send num_train_epoch as parameter, and if the number of retrain epochs is not the same as the number of existing model paths, don't create config file" make sure I didn't create any bugs
    create_retrain_evaluation_configs_for_project(names, created_configs_path, tokenizer_name, pretrain_name_suffix, 
        created_configs_path_name_suffix, dataset_base_dir, base_model_location, model_type, False, False, num_labels, 
        created_datasets_retrain_base_name_suffix, retrain_name_suffix, eval_name_suffix, created_datasets_variability_base_name_suffix,
        number_of_bins,base_suffix,False,num_train_epoch) #TODO: change to created_datasets_variability_base_name_suffix
    # TODO: neeed to create eval for each type of lora training, need to add an if use lora
    if use_lora:
        # print("##### this is lora eval #####")
        created_configs_path_name_suffix="_eval_lora" + project_suffix
        eval_name_suffix = created_configs_path_name_suffix
        create_retrain_evaluation_configs_for_project(names, created_configs_path, tokenizer_name, pretrain_name_suffix, 
            created_configs_path_name_suffix, dataset_base_dir, base_model_location, model_type, use_lora, freeze_model, num_labels, 
            created_datasets_retrain_base_name_suffix, retrain_lora_name_suffix, eval_name_suffix, 
            created_datasets_variability_base_name_suffix,number_of_bins,base_suffix,True,num_train_epoch) #TODO: change to created_datasets_variability_base_name_suffix
        # print("##### this is lora over lora eval #####")

        created_lora_over_lora_configs_path_name_suffix="_eval_lora_over_lora" + project_suffix
        lora_over_lora_eval_name_suffix = created_lora_over_lora_configs_path_name_suffix
        create_retrain_evaluation_configs_for_project(names, created_configs_path, tokenizer_name, lora_pretrain_name_suffix, 
        created_lora_over_lora_configs_path_name_suffix, dataset_base_dir, base_model_location, model_type, use_lora, freeze_model, num_labels, 
        created_datasets_retrain_base_name_suffix, retrain_lora_over_lora_name_suffix, lora_over_lora_eval_name_suffix, 
        created_datasets_variability_base_name_suffix,number_of_bins,base_suffix,False,num_train_epoch) 

    apply_label_transform_to_config_directory(pretrain_training_config_path, label_transform)
    if use_lora:
        apply_label_transform_to_config_directory(lora_pretrain_training_config_path, label_transform)
    apply_label_transform_to_config_directory(retrain_training_config_path, label_transform)
    apply_label_transform_to_config_directory(os.path.join(created_configs_path, "eval_configs"), label_transform)


def create_variability_files_configs(bigwig_files, names, chromosomes, seq_size, variability_extraction_config_path, 
                                     created_datasets_variability_base_name_suffix,dataset_base_dir=HUGGINGFACE_DATASET_BASE_DIR):
    for name,file in zip(names,bigwig_files):
        base_dict = create_base_dictionary()
        paths = base_dict["paths"]
        task = base_dict["task"]
        files_without = bigwig_files[:]
        files_without.remove(file)
        task["type"] = CPG_SEPERATING_SITES_TASK_NAME
        paths["raw_data_paths"] = files_without
        task["chromosomes"] = chromosomes
        task["test_size"] = 0
        # paths["train_path"] = os.path.join("/sci/archive/michall/roeizucker/huggingface_datasets_dir/",name + created_datasets_variability_base_name_suffix + "csv") 
        paths["per_variant_data_path"] = os.path.join(dataset_base_dir,name + created_datasets_variability_base_name_suffix + ".csv") 
        task['variability_type'] = STD_VARIABILITY_TYPE
        task['variant_seperation_type'] = QUANTILE_SEPERATION_TYPE
        task['variant_seperation_threshold'] = 0.09
        task['output_per_variant_data'] = True
        task['output_window_data'] = False
        task["seq_size"] = seq_size
        path = os.path.join(variability_extraction_config_path,name + created_datasets_variability_base_name_suffix + ".yaml")
        # print("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_data_extraction_params.sh",path)
        if os.path.exists(path):
            continue
        with open(path, 'w') as file:
            yaml.dump(base_dict, file, default_flow_style=False, sort_keys=False)


def create_retrain_evaluation_configs_for_project(names, created_configs_path, tokenizer_name, pretrain_name_suffix, 
        created_configs_path_name_suffix, dataset_base_dir, base_model_location, model_type, use_lora, freeze_model, num_labels, 
        created_datasets_retrain_base_name_suffix, retrain_name_suffix, eval_name_suffix, created_datasets_variability_base_name_suffix,
        number_of_bins,base_suffix,lora_over_fine_tuned,num_train_epoch):
    eval_configs_path = os.path.join(created_configs_path,"eval_configs")
    if not os.path.exists(eval_configs_path):
        os.mkdir(eval_configs_path)

    for index in range(len(names)):
        curr_base_name = names[index]
            
        eval_dataset_path = os.path.join(dataset_base_dir,names[index]  + created_datasets_retrain_base_name_suffix + "_test")
        base_models_path =  os.path.join(base_model_location,curr_base_name + pretrain_name_suffix)
        
        existing_model_names = []

        # NOTE: The no_pretraining eval path is accidentally blocked by the pretrain directory check. In config_manager.py (line 1041), if base_models_path does not exist, the function continues before it reaches the no_pretraining branch at lines 1049-1063. So even if the no_pretraining retrain checkpoints exist, their eval config will never be created unless the unrelated pretrain directory also exists.
        if not os.path.exists(base_models_path):
            continue
        for file in os.listdir(base_models_path):
                # TODO: change epoch to constant
            if file.startswith(SAVED_EPOCH_PREFIX):
                existing_model_names.append(file)
        # print("# do eval")
        no_pretrain_retrain_name_suffix = "_" + NO_PRETRAINING_CONFIG_NAME + retrain_name_suffix
        no_pretrain_eval_name_suffix = "_" + NO_PRETRAINING_CONFIG_NAME + eval_name_suffix
        no_pretrain_base_model_path = os.path.join(base_model_location,curr_base_name + no_pretrain_retrain_name_suffix)
        no_pretrain_created_configs_path_name_suffix = "_" + NO_PRETRAINING_CONFIG_NAME + created_configs_path_name_suffix
        if os.path.exists(no_pretrain_base_model_path):
            no_pretrain_models_for_evaluation = []
            for file in os.listdir(no_pretrain_base_model_path):
                if file.startswith(SAVED_EPOCH_PREFIX):
                    no_pretrain_models_for_evaluation.append(os.path.join(no_pretrain_base_model_path,file))
            eval_path = os.path.join(eval_configs_path,curr_base_name + no_pretrain_created_configs_path_name_suffix + ".yaml")
            if len(no_pretrain_models_for_evaluation) == num_train_epoch:
                create_eval_config_file(tokenizer_name, dataset_base_dir, model_type, use_lora, freeze_model, num_labels, 
                                        created_datasets_variability_base_name_suffix, number_of_bins, base_suffix, lora_over_fine_tuned, 
                                        curr_base_name, eval_dataset_path, no_pretrain_eval_name_suffix, no_pretrain_models_for_evaluation,
                                        eval_path)
        for name in existing_model_names:
            curr_retrain_name_suffix = "_" + name + retrain_name_suffix
            curr_eval_name_suffix = "_" + name + eval_name_suffix
            curr_created_configs_path_name_suffix = "_" + name + created_configs_path_name_suffix
            pretrained_model_path = os.path.join(base_models_path,name)
            curr_models_base_path = os.path.join(base_model_location,curr_base_name + curr_retrain_name_suffix)
            if not os.path.exists(curr_models_base_path):
                continue
            models_for_evaluation = [pretrained_model_path]
            for file in os.listdir(curr_models_base_path):
                if file.startswith(SAVED_EPOCH_PREFIX):
                    models_for_evaluation.append(os.path.join(curr_models_base_path,file))
            if len(models_for_evaluation) == num_train_epoch + 1:
                eval_path = os.path.join(eval_configs_path,curr_base_name + curr_created_configs_path_name_suffix + ".yaml")
                create_eval_config_file(tokenizer_name, dataset_base_dir, model_type, use_lora, freeze_model, num_labels, 
                                        created_datasets_variability_base_name_suffix, number_of_bins, base_suffix, lora_over_fine_tuned, 
                                        curr_base_name, eval_dataset_path, curr_eval_name_suffix, models_for_evaluation,
                                        eval_path)

def create_eval_config_file(tokenizer_name, dataset_base_dir, model_type, use_lora, freeze_model, num_labels, 
                            created_datasets_variability_base_name_suffix, number_of_bins, base_suffix, lora_over_fine_tuned, 
                            curr_base_name, eval_dataset_path, eval_name_suffix, models_for_evaluation, eval_path):
    base_dict = create_eval_config_dict(tokenizer_name, model_type, use_lora, freeze_model, num_labels,
                                created_datasets_variability_base_name_suffix, number_of_bins, curr_base_name, eval_dataset_path, 
                                eval_name_suffix, models_for_evaluation,base_suffix,dataset_base_dir,lora_over_fine_tuned)
    # print(f'sbatch --mem=45g -c15 --gres=gg:g4:2 --time=5-23 --killable --requeue --wrap="/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_eval_params.sh {eval_path}"')
    with open(eval_path, 'w') as file:
        yaml.dump(base_dict, file, default_flow_style=False, sort_keys=False)

def create_eval_config_dict(tokenizer_name, model_type, use_lora, freeze_model, num_labels, created_datasets_variability_base_name_suffix, 
        number_of_bins, curr_base_name, eval_dataset_path, curr_eval_name_suffix, models_for_evaluation,base_suffix,dataset_base_dir,
        lora_over_finetuned):
    base_dict = create_base_dictionary()
    paths_config = base_dict["paths"]
    paths_config["variant_file_path"] = os.path.join(dataset_base_dir,curr_base_name + created_datasets_variability_base_name_suffix + ".csv") 
    paths_config["model_paths"] = models_for_evaluation
    paths_config["dataset_path"] = eval_dataset_path

    task_config = base_dict["task"]
    task_config["task_type"] = CPG_EVALUATION_TASK_TYPE
    task_config["sub_task"] = EVALUATE_MULTIPLE_CHCEKPOINTS_SUBTASK_TYPE
    task_config["analysis_name"] = curr_base_name + curr_eval_name_suffix
    task_config["top_rows"] = -1
    task_config["use_variant_file"] = True   # might need to change that name
    task_config["vriant_grouping_method"] = BINS_GROUPING_METHOD
    task_config["number_of_bins"] = number_of_bins
    task_config["base_suffix"] = base_suffix  #TODO: change base suffix to results dir

    model_name = tokenizer_name.split("/")[1]
    model_repo = tokenizer_name.split("/")[0]

                # TODO: export to function, and apply in the creation of training config
    model_config = create_model_config(model_type, use_lora, freeze_model, num_labels, model_name, model_repo)
    # TODO: move to create_model_config, when function is called from ther places, need to send false instead
    model_config["lora_over_finetuned"] = lora_over_finetuned
    base_dict["model"] = model_config
    return base_dict

def create_retrain_training_configs_for_project(names, tokenizer_name, pretrain_name_suffix, created_configs_path_name_suffix, 
    dataset_base_dir, base_model_location, model_type, use_lora, freeze_model, num_labels, load_best_model_at_end, num_train_epoch, 
    per_device_train_batch_size, per_device_eval_batch_size, learning_rate, metric_for_best_model, save_stratagy, number_of_steps, 
    save_total_limit, add_epoch_end_save_callback, save_at_end, continue_from_last, created_datasets_retrain_base_name_suffix, 
    retrain_training_config_path, retrain_name_suffix,top_rows,max_grad_norm,lora_over_finetuned,load_dataset_to_memory,OVERRITE_BATCH_SIZE,min_number_of_cpg_sites=-1):
    for index in range(len(names)):
        curr_base_name = names[index]
        dataset_path = []
            
        dataset_path.append(os.path.join(dataset_base_dir,names[index]  + created_datasets_retrain_base_name_suffix + "_train"))
        eval_dataset_path = os.path.join(dataset_base_dir,names[index]  + created_datasets_retrain_base_name_suffix + "_test")
        # TODO: there is an issue here
        base_models_path =  os.path.join(base_model_location,curr_base_name + pretrain_name_suffix)
        
        # print("# run retrain")
        # TODO: extract function 
        base_dict = create_base_dictionary()
        name = NO_PRETRAINING_CONFIG_NAME
        curr_retrain_name_suffix = "_" + name + retrain_name_suffix
        curr_created_configs_path_name_suffix = "_" + name + created_configs_path_name_suffix
        change_base_dict_to_training_dict(base_dict, tokenizer_name, curr_retrain_name_suffix, base_model_location, CPG_TRAINING_TASK_TYPE,
                model_type, use_lora, freeze_model, num_labels, load_best_model_at_end, num_train_epoch,
                per_device_train_batch_size, per_device_eval_batch_size, learning_rate, metric_for_best_model, 
                save_stratagy, add_epoch_end_save_callback, save_at_end, continue_from_last, curr_base_name, dataset_path, 
                number_of_steps, save_total_limit,top_rows=top_rows,max_grad_norm=max_grad_norm,
                lora_over_finetuned=lora_over_finetuned,load_dataset_to_memory=load_dataset_to_memory,
                min_number_of_cpg_sites=min_number_of_cpg_sites,OVERRITE_BATCH_SIZE=OVERRITE_BATCH_SIZE,add_epoch_end_prediction=True,eval_dataset_path=eval_dataset_path)
        # print("#",name)
        path = os.path.join(retrain_training_config_path,curr_base_name + curr_created_configs_path_name_suffix + ".yaml")
        # print(f'sbatch --mem=45g -c15 --gres=gg:g4:{max(2,min(math.ceil( per_device_train_batch_size / 3), 8))} --time=5-23 --killable --requeue --wrap="/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_training_params.sh',path + '"')
        if not os.path.exists(path):
            with open(path, 'w') as file:
                yaml.dump(base_dict, file, default_flow_style=False, sort_keys=False)

        
        existing_model_names = []
        if not os.path.exists(base_models_path):
            # print(base_models_path, "does not exist")
            continue
        for file in os.listdir(base_models_path):
            if file.startswith("epoch"):
                existing_model_names.append(file)

        for name in existing_model_names:
            # print("#",name)
            base_dict = create_base_dictionary()
            curr_retrain_name_suffix = "_" + name + retrain_name_suffix
            pretrained_model_path = os.path.join(base_models_path,name)
            curr_created_configs_path_name_suffix = "_" + name + created_configs_path_name_suffix
            change_base_dict_to_training_dict(base_dict, tokenizer_name, curr_retrain_name_suffix, base_model_location, CPG_RETRAINING_TASK_TYPE,
                    model_type, use_lora, freeze_model, num_labels, load_best_model_at_end, num_train_epoch,
                    per_device_train_batch_size, per_device_eval_batch_size, learning_rate, metric_for_best_model, 
                    save_stratagy, add_epoch_end_save_callback, save_at_end, continue_from_last, curr_base_name, dataset_path, 
                    number_of_steps, save_total_limit,pretrained_model_path,top_rows=top_rows,max_grad_norm=max_grad_norm,
                    lora_over_finetuned=lora_over_finetuned,min_number_of_cpg_sites=min_number_of_cpg_sites,
                    OVERRITE_BATCH_SIZE=OVERRITE_BATCH_SIZE,add_epoch_end_prediction=True,eval_dataset_path=eval_dataset_path)
            path = os.path.join(retrain_training_config_path,curr_base_name + curr_created_configs_path_name_suffix + ".yaml")
            # print(f'sbatch --mem=45g -c15 --gres=gg:g4:{max(2,min(math.ceil( per_device_train_batch_size / 3), 8))} --time=5-23 --killable --requeue --wrap="/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_training_params.sh',path + '"')
            if os.path.exists(path):
                continue
            with open(path, 'w') as file:
                yaml.dump(base_dict, file, default_flow_style=False, sort_keys=False)

def create_pretrain_training_configs_for_project(names, tokenizer_name, created_datasets_pretrain_base_name_suffix, pretrain_name_suffix,
    created_configs_path_name_suffix, dataset_base_dir, base_model_location, task_type, model_type, use_lora, freeze_model, num_labels, 
    load_best_model_at_end, num_train_epoch, per_device_train_batch_size, per_device_eval_batch_size, learning_rate, metric_for_best_model,
    save_stratagy, number_of_steps, save_total_limit, add_epoch_end_save_callback, save_at_end, continue_from_last, 
    pretrain_training_config_path,top_rows,max_grad_norm,load_dataset_to_memory,OVERRITE_BATCH_SIZE,min_number_of_cpg_sites=-1):
    if not os.path.exists(pretrain_training_config_path):
        os.mkdir(pretrain_training_config_path)
    # print("# pretraining run")
    for index in range(len(names)):
        curr_base_name = names[index]
        dataset_path = []
        base_dict = create_base_dictionary()

        for i in range(len(names)):
            if i != index:
                dataset_path.append(os.path.join(dataset_base_dir,names[i]  + created_datasets_pretrain_base_name_suffix + "_train"))
            
        change_base_dict_to_training_dict(base_dict, tokenizer_name, pretrain_name_suffix, base_model_location, task_type,
                model_type, use_lora, freeze_model, num_labels, load_best_model_at_end, num_train_epoch,
                per_device_train_batch_size, per_device_eval_batch_size, learning_rate, metric_for_best_model, 
                save_stratagy, add_epoch_end_save_callback, save_at_end, continue_from_last, curr_base_name, dataset_path, number_of_steps,
                  save_total_limit,top_rows=top_rows,max_grad_norm=max_grad_norm,load_dataset_to_memory=load_dataset_to_memory,
                  min_number_of_cpg_sites=min_number_of_cpg_sites,OVERRITE_BATCH_SIZE=OVERRITE_BATCH_SIZE,add_epoch_end_prediction=False)

        path = os.path.join(pretrain_training_config_path,curr_base_name + created_configs_path_name_suffix + ".yaml")
        # print(f'sbatch --mem=45g -c15 --gres=gg:g4:{max(2,min(math.ceil( per_device_train_batch_size / 3), 8))} --time=5-23 --killable --requeue --wrap="/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_training_params.sh',path + '"')
        if os.path.exists(path):
            continue
        with open(path, 'w') as file:
            yaml.dump(base_dict, file, default_flow_style=False, sort_keys=False)

