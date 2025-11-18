#!/bin/bash
#SBATCH --job-name=dyn-test  # Job name
#SBATCH --output=BENCH_DYNESTY_LARGEPARAM_outputs/output_dynesty_dynamic.txt          # Standard output file
#SBATCH --error=BENCH_DYNESTY_LARGEPARAM_outputs/error_dynesty_dynamic.txt            # Standard error file
#SBATCH --partition=parallel	      # Partition or queue name
#SBATCH --nodes=3                     # Number of nodes
#SBATCH --ntasks-per-node=48          # Number of tasks per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --time=2-00:00:00             # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=ALL               # Send email at job completion
#SBATCH --mail-user=wbalmer1@jhu.edu  # Email address for notifications

# Load any necessary modules or dependencies
ml anaconda

# Activate the virtual environment
conda activate prt3

# Run your Python script

mpiexec -n 144 python prtshell_prtbenchmark_example.py
# mpiexec -n 48 python prtshell_prtbenchmark_example.py


# Deactivate the virtual environment
conda deactivate

