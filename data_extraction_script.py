print("started loading libraries",flush=True)
import yaml
import sys, os
# sys.path.insert(0, os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code"))

# from utils.dataset_utils import create_dataset, create_dataset_dict, create_tokenizer
from src.data_extractor import extract_data
cfg_path = sys.argv[1]
if not os.path.exists(cfg_path):
    raise Exception("Config file does not exist")
print("#"*30)
print("script started with config file:",cfg_path,flush=True)
print("#"*30)

# print(cfg_path)
cfg = yaml.safe_load(open(cfg_path))
# cfg = yaml.safe_load(open("../configs/config_extraction_full_window_filtration_ENCFF713LYH.yaml"))
extract_data(cfg)
