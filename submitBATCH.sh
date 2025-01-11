#!/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH -t 07:00:00
#SBATCH -J MAGEPROsubmit
#SBATCH -A csd832
#SBATCH -o ../working_err/MAGEPROsubmit.%j.%N.out
#SBATCH -e ../working_err/MAGEPROsubmit.%j.%N.err
#SBATCH --export=ALL
#SBATCH --constraint="lustre"

source ~/.bashrc
conda activate r_py

Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/HIS/maf_hwe_rate_filtered_HM3_filtered/downsampled/MESA_HISchr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/gene_expression/processed/MESA_HIS_ge_normalized.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/covars/ready/MESA_HIS_covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/weights_for_crossanc --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/scratch_ds --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/intermediate_ds --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun FALSE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE/SUMSTATS_READY --sumstats eqtlgen,eurgtex,genoa --models SINGLE,MAGEPRO --ss 31684,574,1032 --hsq_p 1 --verbose 2 --num_covar 42 --batch TRUE --skip_susie TRUE --num_batches 20

Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/CAU/maf_hwe_rate_filtered_HM3_filtered/downsampled/MESA_CAUchr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/gene_expression/processed/MESA_CAU_ge_normalized.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/covars/ready/MESA_CAU_covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_CAU_Monocyte/weights_for_crossanc --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_CAU_Monocyte/scratch_ds --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_CAU_Monocyte/intermediate_ds --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun FALSE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE/SUMSTATS_READY --sumstats eqtlgen,eurgtex,genoa --models SINGLE,MAGEPRO --ss 31684,574,1032 --hsq_p 1 --verbose 2 --num_covar 42 --batch TRUE --skip_susie TRUE --num_batches 20

Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/AA/maf_hwe_rate_filtered_HM3_filtered/downsampled/MESA_AAchr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/gene_expression/processed/MESA_AA_ge_normalized.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/covars/ready/MESA_AA_covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_AA_Monocyte/weights_for_crossanc --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_AA_Monocyte/scratch_ds --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_AA_Monocyte/intermediate_ds --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun FASLE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE/SUMSTATS_READY --sumstats eqtlgen,eurgtex,genoa --models SINGLE,MAGEPRO --ss 31684,574,1032 --hsq_p 1 --verbose 2 --num_covar 42 --batch TRUE --skip_susie TRUE --num_batches 20
