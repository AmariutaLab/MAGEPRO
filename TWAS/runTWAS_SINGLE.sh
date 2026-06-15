#!/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH -t 12:00:00
#SBATCH -J RUNTWAS
#SBATCH -A csd832
#SBATCH -o ../working_err/RUNTWAS.%j.%N.out
#SBATCH -e ../working_err/RUNTWAS.%j.%N.err
#SBATCH --export=ALL
#SBATCH --constraint="lustre"

source ~/.bashrc
conda activate r_env

home=/expanse/lustre/projects/ddp412/kakamatsu

pheno=$1
gwas=$2
wgtdir=$3
outbase=$4
ld=$5
wgtname=$6
wgt=${home}/fusion_multipop/magepro_weight_${wgtname}.pos

mkdir ${outbase}
out=${outbase}/${pheno}
#mkdir ${out}

for chr in {1..22}; do \
	Rscript ${home}/fusion_multipop/FUSION.assoc_test_multipop.R --force_model LASSO --chr $chr --ref_ld_chr $ld --sumstats $gwas --weights $wgt --weights_dir $wgtdir --out ${out}.${chr}.dat; \
done
