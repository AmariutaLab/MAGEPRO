# bash script to test TWAS simulations 

afr_geno_prefix=/expanse/lustre/projects/ddp412/kakamatsu/ldref/afr/1000G_afr_chr
eur_geno_prefix=/expanse/lustre/projects/ddp412/kakamatsu/ldref/eur/1000G_eur_chr
amr_geno_prefix=/expanse/lustre/projects/ddp412/kakamatsu/ldref/amr/1000G_amr_chr

plink_path=/expanse/lustre/projects/ddp412/kakamatsu/plink
genos_paths=${afr_geno_prefix},${eur_geno_prefix},${amr_geno_prefix}
pops_list=AFR,EUR,AMR
tmp_dir=/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_TWAS_SIMULATIONS/tmp
out=/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_TWAS_SIMULATIONS/results

python sim_twas.py --plink_path ${plink_path} --geno_prefix_list ${genos_paths} --pops_list ${pops_list} --tmpdir ${tmp_dir} --out ${out} 
