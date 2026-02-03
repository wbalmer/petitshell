#!/bin/bash
#SBATCH --job-name=prts-bin  # Job name
#SBATCH --output=hd47127b_shell_twosurface_full_rvsplit_pmn_outputs/output.txt          # Standard output file
#SBATCH --error=hd47127b_shell_twosurface_full_rvsplit_pmn_outputs/error.txt            # Standard error file
#SBATCH --partition=parallel	      # Partition or queue name
#SBATCH --nodes=4                     # Number of nodes
#SBATCH --ntasks-per-node=48          # Number of tasks per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --time=06:00:00             # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=ALL               # Send email at job completion
#SBATCH --mail-user=wbalmer1@jhu.edu  # Email address for notifications

# Load any necessary modules or dependencies
ml anaconda

# Activate the virtual environment
conda activate prt3

# Run your Python script

# mpiexec -n 96 python prtshell_binbd_twocol.py
mpiexec -n 192 python prtshell_binbd_twocol.py
# mpiexec -n 96 python prtshell_binbd_onecol.py

# Deactivate the virtual environment
conda deactivate

