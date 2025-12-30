#!/bin/bash
#SBATCH --job-name="MAGEPROgenoa_random"
#SBATCH --output="/expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GENOA_target/random/MAGEPRO_genoa_target_random.%j.%N.out"
#SBATCH --error="/expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GENOA_target/random/MAGEPRO_genoa_target_random.%j.%N.err"

#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --account=csd832
#SBATCH --export=ALL
#SBATCH -t 8:00:00

source ~/.bashrc
conda activate magepro_env
module load cpu/0.15.4 
module load parallel/20200822

srun Rscript /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO/RUN_MAGEPRO_PIPELINE.R \
--magepro_path /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO/ \
--bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GENOA/GENOTYPES/phg000927.v1.NHLBI_GENOA_GWAS_AA.genotype-calls-matrixfmt.c1/AA_genotypes_combined/maf_hwe_rate_relatedness_HM3_filtered_people_with_GE_inGEU_YRI_redo/GENOA_AA_chr \
--ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GENOA/GE/GENOA_AA_ge_normalized.bed.gz \
--covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GENOA/COVARS/AA_GENOA_covariates_ready.txt \
\
--out /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GENOA_target/random/out \
--scratch /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GENOA_target/random/scratch \
--intermed_dir /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GENOA_target/random/intermediate \
\
--PATH_plink /expanse/lustre/projects/ddp412/sgolzari/plink/plink \
--PATH_gcta /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO/fusion_twas-master/gcta_nr_robust \
\
--sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE/SUMSTATS_READY \
--sumstats mesahis \
--ss 352 \
\
--models MAGEPRO \
\
--verbose 2 \
--num_covar 38 \
--skip_susie TRUE \
--num_batches 20 \
--batch TRUE \
--hsq_p 1