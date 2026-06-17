. /home/users/roeizucker/tests/my_env/bin/activate
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID

python3 /home/users/roeizucker/tests/jupyter_notebooks/Tom_Hope_Project/refactored_code/evaluation_script.py  $1

# how to run:
# sbatch --mem=45g -c15 --gres=gg:g4:2 --time=5-23 --killable --requeue --wrap="/cs/usr/roeizucker/new_storage/code_production_dir/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_eval_params.sh 