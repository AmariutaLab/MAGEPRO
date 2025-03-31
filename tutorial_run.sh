plink_path=$1
gcta_path=$2
path_to_magepro=$3

Rscript RUN_MAGEPRO_PIPELINE.R --bfile ../DATA/MAGEPRO_data/GEUVADIS_EUR_genotypes/chr/GEUVADIS_EUR_chr \
--magepro_path $path_to_magepro \
--ge ../DATA/MAGEPRO_data/GEUVADIS_normalized_gene_expression_EUR.bed.gz \
--covar ../DATA/MAGEPRO_data/GEUVADIS_EUR_covariates.txt \
--out ../SAMPLE_FILES/OUTPUT \
--scratch ../SAMPLE_FILES/SCRATCH \
--intermed_dir ../SAMPLE_FILES/INTERMED \
--PATH_plink $plink_path \
--PATH_gcta $gcta_path \
--rerun FALSE \
--sumstats_dir ../DATA/MAGEPRO_data/MAGEPRO_SUMSTATS_SuSiE \
--sumstats mesahis \
--models SINGLE,MAGEPRO \
--ss 352 \
--hsq_p 1 \
--verbose 2 \
--num_covar 36 \
--skip_susie TRUE \
--subset_genes SAMPLE_DATA/sample_genes.txt \
--batch F
