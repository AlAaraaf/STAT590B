#!/bin/bash

#SBATCH --time=01:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 8 processor core(s) per node 
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --job-name="dl3classa"
#SBATCH --mail-user=sjx@iastate.edu   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load r
export R_LIBS_USER=/work/classtmp/carrice/Rlibs
mkdir -p $R_LIBS_USER
Rscript train3class.R
