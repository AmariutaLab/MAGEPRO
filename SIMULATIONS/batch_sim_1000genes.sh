#!/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH -t 20:00:00
#SBATCH -J runsims
#SBATCH -A csd832
#SBATCH -o runsims.%j.%N.out
#SBATCH -e runsims.%j.%N.err
#SBATCH --export=ALL
#SBATCH --constraint="lustre"

source ~/.bashrc
conda activate r_py

numafr=$1 # afr sample size to test
heritability=$2 #preset heritability
eur_geno_prefix=$3
afr_geno_prefix=$4
amr_geno_prefix=$5
out=$6

echo "running simulations with $numafr AFR individuals and a gene with heritability $heritability"

bash run_simulation_1000genes.sh $numafr $heritability $eur_geno_prefix $afr_geno_prefix $amr_geno_prefix $out
