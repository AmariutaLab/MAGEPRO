#!/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 12:00:00
#SBATCH -J reviews_sims
#SBATCH -A csd832
#SBATCH -o ../../../../working_err/reviews_sims.%j.%N.out
#SBATCH -e ../../../../working_err/reviews_sims.%j.%N.err
#SBATCH --export=ALL
#SBATCH --constraint="lustre"

source ~/.bashrc
conda activate r_py

numafr=$1 # afr sample size to test
heritability=$2 #preset heritability
correlation=$3
eur_geno_prefix=$4
afr_geno_prefix=$5
amr_geno_prefix=$6
num_causal=$7
threads=$8
out=$9
temp=/scratch/$USER/job_$SLURM_JOBID
#temp=/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_SIMULATIONS_DIR/MAGEPRO_sims_scratch

echo "running simulations with $numafr AFR individuals and a gene with heritability $heritability and effect size correlation $correlation"

bash run_simulation_corr1000genes.sh $numafr $heritability $correlation $eur_geno_prefix $afr_geno_prefix $amr_geno_prefix $num_causal $threads $out $temp
