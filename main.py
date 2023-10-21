from train import *
import subprocess


config = utils.read_json('config.json')
params_list = ['T',"eps","MH"]
root_path = config["root_path"]

for param in params_list:
    root_path = root_path + param + "=" + str(config[param]) + "||"
root_path = root_path[:len(root_path)-2] + "/"

## Make Directory if it doesn't Exist 
if not os.path.exists(root_path):
    os.makedirs(root_path)
else:
    if config['reset'] == "True":
        utils.delete_contents(root_path)

utils.copy_folders(['train.py','utils.py','config.json'],root_path)


    
sbatch_script = f"""#!/bin/bash
#SBATCH -A standby
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:58:00
#SBATCH --output={root_path}result.out

module --force purge 
module load anaconda/2020.11-py38
conda activate refh
python -c 'from train import train;train("{root_path}",False)'

    """
with open("temp_sbatch_script.sbatch", "w") as f:
    f.write(sbatch_script)


output  = subprocess.run(['sbatch','temp_sbatch_script.sbatch'], stdout=subprocess.PIPE).stdout.decode('utf-8').split()

curr_job = utils.read_json('curr_job.json')
curr_job[root_path] = output[-1]
utils.write_json(curr_job,'curr_job.json')