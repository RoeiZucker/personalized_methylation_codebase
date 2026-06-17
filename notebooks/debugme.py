import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code"))
import yaml
import importlib
# from src.training import run_training
import src.training as train
from src.training import run_training
from src.evaluator import perform_evaluation
file = yaml.safe_load(open("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/virt_token_500m_human_ref.yaml"))
run_training(file)


# file = yaml.safe_load(open("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/_sixteen_shards_of_adonalsium_liver/_sixteen_shards_of_adonalsium_liver_lr_1e-06_bs_2_seq_5400_testsize_0.2/eval_configs/3Q_no_pretraining_eval_sixteen_shards_of_adonalsium_liver_lr_1e-06_bs_2_seq_5400_testsize_0.2.yaml"))
# perform_evaluation(file)