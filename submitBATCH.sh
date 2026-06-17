#!/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH -t 02:00:00
#SBATCH -J MAGEPROsubmit
#SBATCH -A csd832
#SBATCH -o "/expanse/lustre/projects/ddp412/kakamatsu/working_err/MAGEPRO_submit.%j.%N.out"
#SBATCH -e "/expanse/lustre/projects/ddp412/kakamatsu/working_err/MAGEPRO_submit.%j.%N.err"
#SBATCH --export=ALL
#SBATCH --constraint="lustre"

source ~/.bashrc
conda activate r_py


# ---------------------- REVIEWS3, GEUVADIS EUR negative control experiments

srun Rscript /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/RUN_MAGEPRO_PIPELINE.R \
--magepro_path /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/ \
--bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GEUVADIS/1KG_geno/plink_HM3_variants_EUR/maf_hwe_rate_relatedness_HM3_ingenoa_b38_redo/downsampled/GEUVADIS_EUR \
--ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GEUVADIS/GE/GEUVADIS_normalized_processed_ge_v26_b38_EUR_start_end.bed.gz \
--covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GEUVADIS/covar/GEUVADIS_EUR_covariates.txt \
\
--out /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_REVIEWS3/GEUVADIS_target/random/out_redo \
--scratch /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_REVIEWS3/GEUVADIS_target/random/scratch \
--intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_REVIEWS3/GEUVADIS_target/random/intermediate \
\
--PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink \
--PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust \
\
--sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE/SUMSTATS_READY \
--sumstats mesahis \
--ss 352 \
\
--models LASSO,LASSO_OLS,LASSO_RESCALE,MAGEPRO \
\
--verbose 2 \
--num_covar 36 \
--skip_susie TRUE \
--num_batches 20 \
--batch TRUE \
--hsq_p 1 \
--rerun TRUE

# ---------------------- REVIEWS3, GENOA AA negative control experiments

srun Rscript /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/RUN_MAGEPRO_PIPELINE.R \
--magepro_path /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/ \
--bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GENOA/GENOTYPES/phg000927.v1.NHLBI_GENOA_GWAS_AA.genotype-calls-matrixfmt.c1/AA_genotypes_combined/maf_hwe_rate_relatedness_HM3_filtered_people_with_GE_inGEU_YRI_redo/GENOA_AA_chr \
--ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GENOA/GE/GENOA_AA_ge_normalized.bed.gz \
--covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GENOA/COVARS/AA_GENOA_covariates_ready.txt \
\
--out /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_REVIEWS3/GENOA_target/random/out_postlasso_ols_redo2 \
--scratch /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_REVIEWS3/GENOA_target/random/scratch \
--intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_REVIEWS3/GENOA_target/random/intermediate \
\
--PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink \
--PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust \
\
--sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE/SUMSTATS_READY \
--sumstats mesahis \
--ss 352 \
\
--models LASSO,LASSO_OLS,LASSO_RESCALE,MAGEPRO \
\
--verbose 2 \
--num_covar 38 \
--skip_susie TRUE \
--num_batches 20 \
--batch TRUE \
--hsq_p 1 \
--rerun TRUE



# skip_susie FALSE here runs susie on randomly sampled sumstats to see if anything changed 
	# see fine mapping function to comment in or out the override with random sumstats
# in this case --ldref_dir and --ldrefs and --out_susie are required 

#--ldref_dir /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPE_GE_data_cleaned/MESA/GENOTYPES/MESA_HIS_GENOTYPES \
#--ldrefs MESA_HIS \
#--out_susie /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_REVIEWS3/GENOA_target/random/susie_results

# --- old runs below

#Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/HIS/maf_hwe_rate_filtered_HM3_filtered/downsampled/MESA_HISchr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/gene_expression/processed/MESA_HIS_ge_normalized.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/covars/ready/MESA_HIS_covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/weights_for_crossanc --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/scratch_ds --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/intermediate_ds --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun FALSE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE/SUMSTATS_READY --sumstats eqtlgen,eurgtex,genoa --models LASSO,MAGEPRO --ss 31684,574,1032 --hsq_p 1 --verbose 2 --num_covar 42 --batch TRUE --skip_susie TRUE --num_batches 20

#Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/CAU/maf_hwe_rate_filtered_HM3_filtered/downsampled/MESA_CAUchr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/gene_expression/processed/MESA_CAU_ge_normalized.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/covars/ready/MESA_CAU_covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_CAU_Monocyte/weights_for_crossanc --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_CAU_Monocyte/scratch_ds --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_CAU_Monocyte/intermediate_ds --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun FALSE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE/SUMSTATS_READY --sumstats eqtlgen,eurgtex,genoa --models LASSO,MAGEPRO --ss 31684,574,1032 --hsq_p 1 --verbose 2 --num_covar 42 --batch TRUE --skip_susie TRUE --num_batches 20

#Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/AA/maf_hwe_rate_filtered_HM3_filtered/downsampled/MESA_AAchr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/gene_expression/processed/MESA_AA_ge_normalized.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/covars/ready/MESA_AA_covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_AA_Monocyte/weights_for_crossanc --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_AA_Monocyte/scratch_ds --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_AA_Monocyte/intermediate_ds --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun FASLE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE/SUMSTATS_READY --sumstats eqtlgen,eurgtex,genoa --models LASSO,MAGEPRO --ss 31684,574,1032 --hsq_p 1 --verbose 2 --num_covar 42 --batch TRUE --skip_susie TRUE --num_batches 20
