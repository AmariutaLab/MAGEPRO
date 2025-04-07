Rscript RUN_MAGEPRO_PIPELINE.R \
 --magepro_path /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO \
 --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/HIS/maf_hwe_rate_filtered_HM3_filtered/MESA_HISchr \
 --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/gene_expression/processed/MESA_HIS_ge_normalized.bed.gz \
 --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/covars/ready/MESA_HIS_covariates_4genoPC.txt \
 --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/weights_4global_1local_ancestry \
 --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/scratch \
 --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/intermediate \
 --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink \
 --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust \
 --rerun TRUE \
 --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE/SUMSTATS_READY \
 --sumstats eqtlgen,eurgtex,genoa,mesaafr \
 --models SINGLE,MAGEPRO,MAGEPRO_localx \
 --ss 31684,574,1032,233 \
 --hsq_p 1 \
 --verbose 2 \
 --num_covar 41 \
 --regress_localpcs TRUE \
 --num_localpcs 1 \
 --skip_susie TRUE \
 --batch TRUE \
 --num_batches 20
