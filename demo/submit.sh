#!/bin/bash     
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpuq
#SBATCH --job-name=agent_demo
#SBATCH --mem=1GB
#SBATCH --mail-type=NONE  
#SBATCH --output=%x_%j.log

echo; echo "Starting slurm job..."
echo "PWD:  $(pwd)"
echo "HOST: $(hostname)"
echo "DATE: $(date)"; echo

# >>> conda initialize >>>
__conda_setup="$('/facility/nfdata-omics/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<

# load singularity and nextflow
#module load openjdk/17.0.8.1_1
module load openjdk/16.0.2
module load singularity/3.8.5
#conda activate nextflow-25.4.3
conda activate nextflow-24.4.4

# launch pipeline
#nextflow run /home/giulia.scotti/pipelines/rnaseq/ -params-file nf-params.yml -profile ht_cluster -ansi-log false -resume
nextflow run nfdata-omics/rnaseq -r custom_multiqc_test -hub ht_gitlab -params-file nf-params.yml -profile ht_cluster -ansi-log false -resume
#nextflow run nfdata-omics/rnaseq -r 0.3.0 -hub ht_gitlab -params-file nf-params.yml -profile ht_cluster -ansi-log false -resume
# nextflow run /home/giulia.scotti/pipelines/report/rnaseq/ -params-file nf-params.yml -profile ht_cluster -ansi-log false -resume

echo; echo "Terminating slurm job..."
echo "DATE: $(date)"; echo
exit
