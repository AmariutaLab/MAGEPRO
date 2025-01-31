#!/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 14:00:00
#SBATCH -J mismatch_h2_setting_0.1_0.05
#SBATCH -A csd832
#SBATCH -o ../../../../../working_err/mismatch_h2_setting_0.1_0.05.%j.%N.out
#SBATCH -e ../../../../../working_err/mismatch_h2_setting_0.1_0.05.%j.%N.err
#SBATCH --export=ALL
#SBATCH --constraint="lustre"

source ~/.bashrc
conda activate r_py

numafr=$1 # afr sample size to test
heritability=$2 #preset heritability
heritability2=$3 #preset heritability
correlation=$4
eur_geno_prefix=$5
afr_geno_prefix=$6
amr_geno_prefix=$7
num_causal=$8
threads=$9
out=${10}
temp=/scratch/$USER/job_$SLURM_JOBID
#temp=/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_SIMULATIONS_DIR/MAGEPRO_sims_scratch

echo "running simulations with $numafr AFR individuals and a gene with heritability $heritability / $heritability2 and effect size correlation $correlation"

bash run_simulation_mismatch_h2.sh $numafr $heritability $heritability2 $correlation $eur_geno_prefix $afr_geno_prefix $amr_geno_prefix $num_causal $threads $out $temp
