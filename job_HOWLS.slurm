#!/bin/bash
#SBATCH --time=10:00:00                     # set max wallclock time
#SBATCH --output=job-%N-%j.out             # set output log name
#SBATCH --error=job-%N-%j.error            # set error log name %N=first node, %j=job number
#SBATCH --job-name=HOWLS_Pipeline         # set job name
#SBATCH -p all                             # set partition (queue)
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --mail-type=ALL                    # set mail alerts
#SBATCH --mail-user=j.bai@ed.ac.uk # set mail address

./HOWLSPipeline.sh
