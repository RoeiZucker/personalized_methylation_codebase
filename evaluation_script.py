print("started loading libraries",flush=True)
import sys, os
# sys.path.insert(0, os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src"))
import yaml
from src.evaluator import perform_evaluation

cfg_path = sys.argv[1]
# print(cfg_path)
cfg = yaml.safe_load(open(cfg_path))
print("#"*30)
print("script started with config file:",cfg_path,flush=True)
print("#"*30,flush=True)
res = perform_evaluation(cfg)
# TODO: check if this is necessery, caused scripts to fail!!!
# res["checkpoint"] = res["paths"].str.strip("/").str.split('/').str[-1]
analysis_name = cfg["task"]["analysis_name"]
base_suffix = cfg["task"]["base_suffix"]


# TODO: change the results location to be a parameter!
dev_dir = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/results"
prod_dir = "/home/users/roeizucker/tests/jupyter_notebooks/Tom_Hope_Project/results"
base_dir = None
if os.path.exists(prod_dir):
    base_dir = prod_dir
elif os.path.exists(dev_dir):
    base_dir = dev_dir

if base_dir is None:
    raise Exception("Not in correct enviroment")

project_dir = os.path.join(base_dir,base_suffix)
if not os.path.exists(project_dir):
    os.mkdir(project_dir)

def write_result_dataframe(df, res_path):
    df.to_csv(res_path, index=None)
    df.to_csv(res_path + ".gitbackup", index=None)
    print("wrote to", res_path)


if isinstance(res, dict):
    for result_name, curr_df in res.items():
        res_path = os.path.join(project_dir, analysis_name + f"_{result_name}.csv")
        write_result_dataframe(curr_df, res_path)
else:
    res_path = os.path.join(project_dir,analysis_name + "_result.csv")
    # if not os.path.exists(res_path):
    #     os.mkdir(res_path)
    write_result_dataframe(res, res_path)