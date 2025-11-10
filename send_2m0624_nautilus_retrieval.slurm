#!/bin/bash
#SBATCH --job-name=ntls-2m0624  # Job name
#SBATCH --output=2M0624_TEST_outputs/output.txt          # Standard output file
#SBATCH --error=2M0624_TEST_outputs/error.txt            # Standard error file

#SBATCH --time=1-00:00:00             # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=ALL               # Send email at job completion
#SBATCH --mail-user=wbalmer1@jhu.edu  # Email address for notifications

#SBATCH --partition=parallel	      # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --ntasks-per-node=48          # Number of tasks per node

# Load any necessary modules or dependencies
ml anaconda

# Activate the virtual environment
conda activate prt3

# Run your Python script

mpiexec -n 27 python prtshell_2m0624_example.py


# Deactivate the virtual environment
conda deactivate

