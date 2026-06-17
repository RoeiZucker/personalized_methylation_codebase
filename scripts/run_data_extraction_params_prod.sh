
. /home/users/roeizucker/tests/my_env/bin/activate
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID

python3 /home/users/roeizucker/tests/jupyter_notebooks/Tom_Hope_Project/refactored_code/data_extraction_script.py  $1
