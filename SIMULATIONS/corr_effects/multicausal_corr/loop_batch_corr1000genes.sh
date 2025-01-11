# loop through various sample sizes and heritability values and submit jobs 

afr_sizes=80,160,240,400,500
heritability=(0.01 0.025 0.05 0.1 0.2 0.4)
#heritability=(0.1)
correlation=0.8
# genotype subset to hapmap snps already
eur_geno_prefix=/expanse/lustre/projects/ddp412/kakamatsu/ldref/eur/1000G_eur_chr
afr_geno_prefix=/expanse/lustre/projects/ddp412/kakamatsu/ldref/afr/1000G_afr_chr
amr_geno_prefix=/expanse/lustre/projects/ddp412/kakamatsu/ldref/amr/1000G_amr_chr
threads=4
num_causal_list=(1 2 4 8 15 25)

jobs=(1 2 3 4 5 6 7 8 9 10)
#jobs=(1)
for num_causal in "${num_causal_list[@]}"; do
    for job in "${jobs[@]}"; do
        out=/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_SIMULATIONS_DIR/MAGEPRO_REVIEWS_${num_causal}causal_correffects_job${job} # CHANGE IF RUNNING WITH DIFFERENT NUMBER OF CAUSAL VARIANTS
        rm -rf $out
        mkdir $out
        out_results=${out}/results
        rm -rf $out_results
        mkdir $out_results
        for h in "${heritability[@]}"; do
            sbatch batch_sim_corr1000genes.sh $afr_sizes $h $correlation $eur_geno_prefix $afr_geno_prefix $amr_geno_prefix $num_causal $threads $out_results
        done
    done
done

