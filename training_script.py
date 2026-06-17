print("started loading libraries",flush=True)

import sys, os
# sys.path.insert(0, os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src"))
import yaml
from src.training import run_training


cfg_path = sys.argv[1]
print("#"*30)
print("training started with file",cfg_path,flush=True)
print("#"*30)

if not os.path.exists(cfg_path):
    raise Exception("Config file does not exist")
# print(cfg_path)
cfg = yaml.safe_load(open(cfg_path))


run_training(cfg)