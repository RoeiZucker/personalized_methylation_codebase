. /cs/usr/roeizucker/new_storage/new_python_env/bin/activate

python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/evaluation_script.py  $1

# how to run:
# sbatch --mem=45g -c15 --gres=gg:g4:2 --time=5-23 --killable --requeue --wrap="/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_eval_params.sh 