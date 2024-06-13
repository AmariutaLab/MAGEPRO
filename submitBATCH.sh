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

Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/CAU/maf_hwe_rate_filtered_HM3_filtered/MESA_CAUchr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/gene_expression/processed/MESA_CAU_ge_normalized.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/covars/ready/MESA_CAU_covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_CAU_Monocyte/weights_final_benchmark --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_CAU_Monocyte/scratch --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_CAU_Monocyte/intermediate --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun FALSE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE --sumstats eqtlgen,eurgtex,mesahis,mesaafr,genoa --models SINGLE,META,PT,SuSiE,PRSCSx,MAGEPRO_fullsumstats,MAGEPRO --ss 31684,574,352,233,1032 --hsq_p 1 --verbose 2 --num_covar 42 --ldref_pt /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/CAU/maf_hwe_rate_filtered_HM3_filtered/MESA_CAUchr --ldref_PRSCSx /expanse/lustre/projects/ddp412/kakamatsu/eQTLsummary/multipopGE/benchmark/LD_ref --dir_PRSCSx /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/BENCHMARK/PRScsx --phi_shrinkage_PRSCSx 1e-7 --pops EUR,EUR,AMR,AFR,AFR --susie_pip 8 --susie_beta 9 --susie_cs 10


Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/AA/maf_hwe_rate_filtered_HM3_filtered/MESA_AAchr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/gene_expression/processed/MESA_AA_ge_normalized.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/covars/ready/MESA_AA_covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_AA_Monocyte/weights_final_benchmark --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_AA_Monocyte/scratch --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_AA_Monocyte/intermediate --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun FALSE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE --sumstats eqtlgen,eurgtex,mesahis,genoa --models SINGLE,META,PT,SuSiE,PRSCSx,MAGEPRO_fullsumstats,MAGEPRO --ss 31684,574,352,1032 --hsq_p 1 --verbose 2 --num_covar 42 --ldref_pt /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/AA/maf_hwe_rate_filtered_HM3_filtered/MESA_AAchr --ldref_PRSCSx /expanse/lustre/projects/ddp412/kakamatsu/eQTLsummary/multipopGE/benchmark/LD_ref --dir_PRSCSx /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/BENCHMARK/PRScsx --phi_shrinkage_PRSCSx 1e-7 --pops EUR,EUR,AMR,AFR --susie_pip 8 --susie_beta 9 --susie_cs 10


Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GTEX_AFR/maf_hwe_rate_relatedness/GTExAFR_chr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GTEX_ge_covar/GTEx_Analysis_v8_eQTL_ge/GTEx_Analysis_v8_eQTL_expression_matrices/Muscle_Skeletal.v8.normalized_expression.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GTEX_ge_covar/GTEx_Analysis_v8_eQTL_covariates/Muscle_Skeletal.v8.covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GTExAA_MUSCLE/weights --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GTExAA_MUSCLE/scratch --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GTExAA_MUSCLE/intermediate --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun FALSE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE --sumstats eqtlgen,eurgtex,mesahis,mesaafr,genoa --models SINGLE,META,PT,SuSiE,PRSCSx,MAGEPRO_fullsumstats,MAGEPRO --ss 31684,574,352,233,1032 --hsq_p 1 --verbose 2 --ldref_pt /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GTEX_AFR/maf_hwe_rate_relatedness/GTExAFR_chr --ldref_PRSCSx /expanse/lustre/projects/ddp412/kakamatsu/eQTLsummary/multipopGE/benchmark/LD_ref --dir_PRSCSx /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/BENCHMARK/PRScsx --phi_shrinkage_PRSCSx 1e-7 --pops EUR,EUR,AMR,AFR,AFR --susie_pip 8 --susie_beta 9 --susie_cs 10


Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GTEX_AFR/maf_hwe_rate_relatedness/GTExAFR_chr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GTEX_ge_covar/GTEx_Analysis_v8_eQTL_ge/GTEx_Analysis_v8_eQTL_expression_matrices/Lung.v8.normalized_expression.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GTEX_ge_covar/GTEx_Analysis_v8_eQTL_covariates/Lung.v8.covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GTExAA_LUNG/weights --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GTExAA_LUNG/scratch --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GTExAA_LUNG/intermediate --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun FALSE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/SuSiE --sumstats eqtlgen,eurgtex,mesahis,mesaafr,genoa --models SINGLE,META,PT,SuSiE,PRSCSx,MAGEPRO_fullsumstats,MAGEPRO --ss 31684,574,352,233,1032 --hsq_p 1 --verbose 2 --ldref_pt /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GTEX_AFR/maf_hwe_rate_relatedness/GTExAFR_chr --ldref_PRSCSx /expanse/lustre/projects/ddp412/kakamatsu/eQTLsummary/multipopGE/benchmark/LD_ref --dir_PRSCSx /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/BENCHMARK/PRScsx --phi_shrinkage_PRSCSx 1e-7 --pops EUR,EUR,AMR,AFR,AFR --susie_pip 8 --susie_beta 9 --susie_cs 10
