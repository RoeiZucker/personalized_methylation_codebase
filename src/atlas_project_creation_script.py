import yaml
import os, sys
from datetime import datetime
import subprocess

# Environment mode flag used by generated command cells
# False -> test scripts, True -> production scripts


from config_manager import (CREATE_PROJECT_TASK_NUM,
                                DATA_EXTRQACTION_TASK_NUM,
                                MODEL_TRAINING_TASK_NUM,
                                MODEL_EVALUATION_TASK_NUM,
                                create_base_dictionary,
                                create_data_extraction_config_dict,
                                DEFAULT_CONFIG_PATH,
                                create_project_config,
                                DEFAULT_CREATE_BASIC_EXTRACTION_FILES,
                                ask_for_tokenizer_name,
                                get_files_names_from_file_list,
                                CONFIG_DIR_PATH,
                                HUGGINGFACE_DATASET_BASE_DIR,
                                TRAINED_HUGGINGFACE_MODELS_LOCATION,
                                CREATE_PROJECT_HP_OPTIMIZATION_TASK_NUM,
                                print_deletion_plan,
                                print_commands_for_roject_config
                                )




import os

SCRIPTS_DIR = "/home/users/roeizucker/tests/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts"
IS_PRODUCTION = False
IS_FOR_SCRIPT = False
# print(os.run())
# IGNORE = os.popen('squeue -u roeizucker -o "%.125j"').read().replace(" ","").split("\n")

_USE_PROD = bool(globals().get("IS_PRODUCTION", False))
if _USE_PROD:
    EXTRACT_SCRIPT = os.path.join(SCRIPTS_DIR, "run_data_extraction_params_prod.sh")
    TRAIN_SCRIPT = os.path.join(SCRIPTS_DIR, "run_training_params_prod.sh")
    EVAL_SCRIPT = os.path.join(SCRIPTS_DIR, "run_eval_params_prod.sh")
else:
    EXTRACT_SCRIPT = os.path.join(SCRIPTS_DIR, "run_data_extraction_params.sh")
    TRAIN_SCRIPT = os.path.join(SCRIPTS_DIR, "run_training_params.sh")
    EVAL_SCRIPT = os.path.join(SCRIPTS_DIR, "run_eval_params.sh")


COMMAND_PREFIX = "sbatch --container-mounts=/shared:/shared --partition=compute-gpu --qos=owner_95 --gres=gpu:1 --time=45:10:00 --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3 "
# usage
path = "/home/users/roeizucker/tests/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/kol_kora_configs/project_configs/_full_liver.yaml"

default_learning_rates = [1e-05, 1e-06, 1e-07]
default_batch_sizes = [2, 4]

OVERRITE_BATCH_SIZE = None


# TODO: run 0 0 and all enters for remaking old run 

default_learning_rates = [1e-05,1e-06,1e-07]
default_batch_sizes = [2,4]
default_seq_sizes = [5400]
default_test_sizes = [0.2]

ATLAS_AUTO_GENERATED_CONFIGS_DIR = (
    "/cs/usr/roeizucker/new_storage/jupyter_notebooks/"
    "Tom_Hope_Project/refactored_code/configs/atlas_configs/auto_generated"
)

def _yaml_files(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, file_name)
        for file_name in os.listdir(path)
        if file_name.endswith(".yaml")
    )

paths = sorted(
    os.path.join(root, file_name)
    for root, _, file_names in os.walk(ATLAS_AUTO_GENERATED_CONFIGS_DIR)
    for file_name in file_names
    if file_name.endswith(".yaml")
)

variability_extraction_commands = []
dataset_creation_commands = []
retrain_creation_commands = []

for master_config_path in paths:
    if not master_config_path.endswith(".yaml") or not os.path.exists(master_config_path):
        raise ValueError("invalid config path")
    config = yaml.safe_load(open(master_config_path))
    base_suffix = config["params"]["project_suffix"]
    configs_base_dir = config["params"]["created_configs_path"]
    if not os.path.exists(configs_base_dir):
        os.mkdir(configs_base_dir)
    learning_rates = default_learning_rates
    batch_sizes = default_batch_sizes
    seq_sizes = default_seq_sizes
    test_sizes = default_test_sizes
    if "learning_rates" in config["params"]:
        learning_rates = config["params"]["learning_rates"]
        del config["params"]["learning_rates"   ]
    if "per_device_train_batch_sizes" in config["params"]:
        batch_sizes = config["params"]["per_device_train_batch_sizes"]
        del config["params"]["per_device_train_batch_sizes"]
    if "seq_sizes" in config["params"]:
        seq_sizes = config["params"]["seq_sizes"]
        del config["params"]["seq_sizes"]
    if "test_sizes" in config["params"]:
        test_sizes = config["params"]["test_sizes"]
        del config["params"]["test_sizes"]
    project_dataset_location = os.path.join(config["params"]["dataset_base_dir"],config["params"]["project_suffix"])
    if not os.path.exists(project_dataset_location):
        if not _USE_PROD:
            os.mkdir(project_dataset_location)
        else:
            print("mkdir",project_dataset_location)

    project_model_location = os.path.join(config["params"]["base_model_location"],config["params"]["project_suffix"])
    if not os.path.exists(project_model_location):
        if not _USE_PROD:
            os.mkdir(project_model_location)
        else:
            print("mkdir",project_model_location)
    # default_numbers_of_steps
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for seq_size in seq_sizes:
                for test_size in test_sizes:
                    curr_config = config.copy()
                    if test_sizes is default_test_sizes:
                        curr_suffix = f"{base_suffix}_lr_{lr}_bs_{batch_size}_seq_{seq_size}"
                    else:
                        curr_suffix = f"{base_suffix}_lr_{lr}_bs_{batch_size}_seq_{seq_size}_testsize_{test_size}"
                    curr_config["params"]["base_suffix"] = base_suffix
                    datasets_suffix = f"{base_suffix}_seq_{seq_size}_datasets"
                    curr_config["params"]["project_suffix"] = curr_suffix
                    curr_config["params"]["datasets_suffix"] = datasets_suffix
                    curr_config["params"]["created_configs_path"] = os.path.join(configs_base_dir,curr_suffix)
                    curr_config["params"]["learning_rate"] = lr
                    curr_config["params"]["per_device_train_batch_size"] = batch_size
                    curr_config["params"]["seq_size"] = seq_size
                    curr_config["params"]["dataset_base_dir"] = project_dataset_location
                    curr_config["params"]["base_model_location"] = project_model_location
                    curr_config["params"]["test_size"] = test_size
                    if OVERRITE_BATCH_SIZE is not None:
                        curr_config["params"]["OVERRITE_BATCH_SIZE"] = OVERRITE_BATCH_SIZE
                    if seq_size == 5400:
                        curr_config["params"]["OVERRITE_BATCH_SIZE"] = 14
                    elif seq_size == 600:
                        curr_config["params"]["OVERRITE_BATCH_SIZE"] = 128
                    # print(lr,batch_size,seq_size)
                    create_project_config(**curr_config["params"])
                    created_configs_path = curr_config["params"]["created_configs_path"]
                    variability_extraction_commands.extend(
                        (EXTRACT_SCRIPT, yaml_path)
                        for yaml_path in _yaml_files(
                            os.path.join(created_configs_path, "variability_extraction")
                        )
                    )
                    dataset_creation_commands.extend(
                        (EXTRACT_SCRIPT, yaml_path)
                        for yaml_path in _yaml_files(
                            os.path.join(created_configs_path, "pretrain_extraction")
                        )
                    )
                    retrain_creation_commands.extend(
                        (EXTRACT_SCRIPT, yaml_path)
                        for yaml_path in _yaml_files(
                            os.path.join(created_configs_path, "retrain_extraction")
                        )
                    )

print("# run variability extraction first:")
for script_path, yaml_path in variability_extraction_commands:
    config = yaml.safe_load(open(yaml_path))
    if os.path.exists(config["paths"]["per_variant_data_path"]):
        continue
    
    print(script_path, yaml_path)

print("# then run dataset creation:")
for script_path, yaml_path in dataset_creation_commands:
    config = yaml.safe_load(open(yaml_path))
    if os.path.exists(config["paths"]["hf_dataset_train_path"]):
        continue
    print(script_path, yaml_path)

print("# then run retrain dataset creation:")
for script_path, yaml_path in retrain_creation_commands:
    config = yaml.safe_load(open(yaml_path))
    if os.path.exists(config["paths"]["hf_dataset_train_path"]):
        continue
    print(script_path, yaml_path)
