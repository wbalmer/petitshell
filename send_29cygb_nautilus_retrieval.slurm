#!/bin/bash
#SBATCH --job-name=ntls-29cygb  # Job name
#SBATCH --output=29cygb_shell_testeqchem_outputs/output.txt          # Standard output file
#SBATCH --error=29cygb_shell_testeqchem_outputs/error.txt            # Standard error file
#SBATCH --partition=parallel	      # Partition or queue name
#SBATCH --nodes=2                     # Number of nodes
#SBATCH --ntasks-per-node=48          # Number of tasks per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --time=12:00:00             # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=ALL               # Send email at job completion
#SBATCH --mail-user=wbalmer1@jhu.edu  # Email address for notifications

# Load any necessary modules or dependencies
ml anaconda

# Activate the virtual environment
conda activate prt3

# Run your Python script

mpiexec -n 96 python prtshell_29cygb_example.py
# mpiexec -n 48 python prtshell_29cygb_example.py


# Deactivate the virtual environment
conda deactivate

