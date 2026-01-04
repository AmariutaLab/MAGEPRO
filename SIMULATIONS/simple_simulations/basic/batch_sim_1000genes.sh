#!/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH -t 24:00:00
#SBATCH -J runsims
#SBATCH -A csd832
#SBATCH -o ../../../working_err/runsims.%j.%N.out
#SBATCH -e ../../../working_err/runsims.%j.%N.err
#SBATCH --export=ALL
#SBATCH --constraint="lustre"

source ~/.bashrc
conda activate r_py

numafr=$1 # afr sample size to test
heritability=$2 #preset heritability
eur_geno_prefix=$3
afr_geno_prefix=$4
amr_geno_prefix=$5
threads=$6
out=$7
temp=/scratch/$USER/job_$SLURM_JOBID
#temp=/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_sims_scratch

echo "running simulations with $numafr AFR individuals and a gene with heritability $heritability"

bash run_simulation_1000genes.sh $numafr $heritability $eur_geno_prefix $afr_geno_prefix $amr_geno_prefix $threads $out $temp
