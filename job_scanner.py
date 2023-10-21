import time
import json
import subprocess
import os
import utils



while True:
    
    time.sleep(5)
    required_job_dic = utils.read_json('curr_job.json')     ### of the form dic = {123456:'/scratch/gilrbeth/abdulsal/file.py'} 
    output = subprocess.run(['squeue','-u','abdulsal'], stdout=subprocess.PIPE).stdout.decode('utf-8').split()  ## Gives the entire output upon running command squeue -u abdulsal
    current_jobs_list = [value for index, value in enumerate(output) if index % 9 == 0][1:] # Retrieves every 9th element to get only jobid excluding the first one since its not jobid
    
    for root_path in required_job_dic:        
        if required_job_dic[root_path] not in current_jobs_list:
            
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
python -c 'from train import train;train("{root_path}",True)'

    """     
            with open("temp_sbatch_script.sbatch", "w") as f:
                f.write(sbatch_script)
            
            output  = subprocess.run(['sbatch','temp_sbatch_script.sbatch'], stdout=subprocess.PIPE).stdout.decode('utf-8').split()
            required_job_dic[root_path] = output[-1]
    
    utils.write_json(required_job_dic,'curr_job.json')
            
        
    
    
    
    