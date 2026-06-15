#!/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J MAGEPRO
#SBATCH -A csd832
#SBATCH -o ../../working_err/RUNTIME.%j.%N.out
#SBATCH -e ../../working_err/RUNTIME.%j.%N.err
#SBATCH --export=ALL
#SBATCH --constraint="lustre"

source ~/.bashrc
conda activate r_py
module load cpu/0.15.4 
module load parallel/20200822 

thread_settings=(1 4 8 16 32)

output_file="runtime_results.txt"

# write header
echo -e "Threads\tDuration(s)" > "$output_file"

# Loop through each thread setting
for n_threads in "${thread_settings[@]}"
do
  cd ../
  echo "Running Rscript with $n_threads threads..."
  start_time=$(date +%s)
  mkdir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/weights_test_8_10_nobatch_$n_threads
  Rscript RUN_MAGEPRO_PIPELINE.R --bfile /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/genotypes/consents_combined/HIS/maf_hwe_rate_filtered_HM3_filtered/MESA_HISchr --ge /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/gene_expression/processed/MESA_HIS_ge_normalized.bed.gz --covar /expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/MESA/covars/ready/MESA_HIS_covariates.txt --out /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/weights_test_8_10_nobatch_$n_threads --scratch /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/scratch --intermed_dir /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/intermediate --PATH_plink /expanse/lustre/projects/ddp412/kakamatsu/plink --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --rerun TRUE --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_datasets_allinfo --sumstats eqtlgen,eurgtex,genoa --models LASSO,MAGEPRO --ss 31684,574,1032 --hsq_p 1 --verbose 2 --num_covar 42 --pops EUR,EUR,AFR --skip_susie FALSE --ldref_dir /expanse/lustre/projects/ddp412/sgolzari/ldref --ldrefs 1000G.EUR.HM3.filtered,1000G.EUR.HM3.filtered,GENOA_AA --out_susie /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_susie --subset_genes /expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/MESA_HIS_Monocyte/intermediate/subset_test_100.txt --batch FALSE --n_threads $n_threads
  end_time=$(date +%s)

  cd TEST_RUNTIME
  duration=$((end_time - start_time))
  echo -e "$n_threads\t$duration" >> "$output_file"
done

