# simulate a random gene and create plink files for that 1MB region 
numsafr_s=$1
IFS=',' read -ra numsafr <<< $numsafr_s
heritability=$2
heritability2=$3
correlation=$4
eur_geno_prefix=$5
afr_geno_prefix=$6
amr_geno_prefix=$7
num_causal=$8
threads=$9
out=${10}
temp=${11}

export MKL_NUM_THREADS=$threads
export NUMEXPR_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

randomgenes=${temp}/random_genes_multi_h${heritability}
rm -rf $randomgenes
mkdir $randomgenes

#causal snp is the same 
for ((i=1; i<=100; i++))
do 

	rm -rf ${randomgenes}/*

	# choose a random TSS and take +- 500 kb
	
	while true; do

		#random chr
		chr=$((1 + RANDOM % 22))
		echo $chr
	
		#pick a random variant and extract the position number
		file=${eur_geno_prefix}${chr}.bim
		num_variants=$(wc -l < "${file}")
		line=$((1 + RANDOM % num_variants))
		random_variant=$(sed -n "${line}p" "${file}")
		random_position=$(echo "${random_variant}" | awk '{ print $4 }')

		start_pos=$(($random_position - 500000))
		if [ $start_pos -lt 0 ]; then 
			start_pos=0
		fi
		end_pos=$(($random_position + 500000))

		echo $start_pos
		echo $end_pos

		#extract genes for each population

		#EUR 
		/expanse/lustre/projects/ddp412/kakamatsu/plink --bfile ${eur_geno_prefix}${chr} --chr $chr --from-bp $start_pos --to-bp $end_pos --make-bed --out ${randomgenes}/EUR_1KG_chr${chr}_${random_position}

		#AFR
		/expanse/lustre/projects/ddp412/kakamatsu/plink --bfile ${afr_geno_prefix}${chr} --chr $chr --from-bp $start_pos --to-bp $end_pos --make-bed --out ${randomgenes}/AFR_1KG_chr${chr}_${random_position}

		#AMR
		/expanse/lustre/projects/ddp412/kakamatsu/plink --bfile ${amr_geno_prefix}${chr} --chr $chr --from-bp $start_pos --to-bp $end_pos --make-bed --out ${randomgenes}/AMR_1KG_chr${chr}_${random_position}


		if [ ! -e ${randomgenes}/EUR_1KG_chr${chr}_${random_position}.bim ] || [ ! -e ${randomgenes}/AFR_1KG_chr${chr}_${random_position}.bim ] || [ ! -e ${randomgenes}/AMR_1KG_chr${chr}_${random_position}.bim ]; then
    			echo "no snps remaining, pick another gene"
			rm -rf ${randomgenes}/* 
			continue
		fi

		EUR_variants_random_gene=$(wc -l < "${randomgenes}/EUR_1KG_chr${chr}_${random_position}.bim")
		AFR_variants_random_gene=$(wc -l < "${randomgenes}/AFR_1KG_chr${chr}_${random_position}.bim")
		AMR_variants_random_gene=$(wc -l < "${randomgenes}/AMR_1KG_chr${chr}_${random_position}.bim")

		if [ $EUR_variants_random_gene -lt 50 ]; then
			echo "not enough snps in chosen gene, pick another gene"
			rm -rf ${randomgenes}/*
			continue
		elif [ $AFR_variants_random_gene -lt 50 ]; then
			echo "not enough snps in chosen gene, pick another gene"
			rm -rf ${randomgenes}/*
			continue
		elif [ $AMR_variants_random_gene -lt 50 ]; then
			echo "not enough snps in chosen gene, pick another gene"
			rm -rf ${randomgenes}/*
			continue
		else
			echo "successfully picked a random gene"
			break
		fi
	done	

	#choose a snp randomly as causal 
	small_snps=$((EUR_variants_random_gene < AFR_variants_random_gene ? EUR_variants_random_gene : AFR_variants_random_gene))
	min_snps=$((AMR_variants_random_gene < small_snps ? AMR_variants_random_gene : small_snps))
	causal_indices=()
	while [ ${#causal_indices[@]} -lt $num_causal ]; do
		index_causal=$((RANDOM % $min_snps))
		if [[ ! " ${causal_indices[@]} " =~ " ${index_causal} " ]]; then
			causal_indices+=($index_causal)
		fi
	done
	causal_indices=$(IFS=, ; echo "${causal_indices[*]}")
	echo $causal_indices

	for numafr in "${numsafr[@]}"; do

		simgeno=${temp}/simulated_genotypes_${numafr}_h${heritability}

		lddir=${temp}/ld_${numafr}_h${heritability}

		sumstatsdir=${temp}/sumstats_${numafr}_h${heritability}

		tempdir=${temp}/temp_${numafr}_h${heritability}

		if [ $i -eq 1 ]; then
			mkdir $simgeno
			mkdir $sumstatsdir
			mkdir $tempdir 
			mkdir ${tempdir}/PRSCSx # later for PRSCSx intermediate files
			mkdir $lddir
		else 
			rm -rf ${simgeno}/*
			rm -rf ${sumstatsdir}/*
			rm -rf ${tempdir}/*
			mkdir ${tempdir}/PRSCSx # later for PRSCSx intermediate files
			rm -rf ${lddir}/*
		fi

		# simulate genotypes using python script
		python3 ../../../Sim_geno.py ${randomgenes}/EUR_1KG_chr${chr}_${random_position} 500 EUR ${simgeno} ${lddir}
		python3 ../../../Sim_geno.py ${randomgenes}/AMR_1KG_chr${chr}_${random_position} 500 AMR ${simgeno} ${lddir}
		python3 ../../../Sim_geno.py ${randomgenes}/AFR_1KG_chr${chr}_${random_position} $numafr AFR ${simgeno} ${lddir}

		# simulation effects 
		python3 Sim_Effects.py $num_causal $heritability $heritability2 $heritability2 $correlation AFR,EUR,AMR $tempdir/effects.txt

		# simulate eqtl analysis to produce EUR full sum stats 
		#python3 Sim_SumStats.py <path to plink files> <sample size> <population> <simulated genotypes> <number of causal snps> <index of causal> <simulation number> <heritability of gene> <sample size of target pop> <sumstats dir> <tempdir> <outdir> <threads>
		python3 Sim_SumStats_corr.py ${randomgenes}/EUR_1KG_chr${chr}_${random_position} 500 EUR ${simgeno}/simulated_genotypes_EUR.csv $num_causal $causal_indices $i $heritability2 $tempdir/effects.txt $correlation $numafr $sumstatsdir $tempdir ${out} $threads
		python3 Sim_SumStats_corr.py ${randomgenes}/AMR_1KG_chr${chr}_${random_position} 500 AMR ${simgeno}/simulated_genotypes_AMR.csv $num_causal $causal_indices $i $heritability2 $tempdir/effects.txt $correlation $numafr $sumstatsdir $tempdir ${out} $threads

		# EUR - LD created in Sim_geno.py
		#/expanse/lustre/projects/ddp412/kakamatsu/plink --bfile ${randomgenes}/EUR_1KG_chr${chr}_${random_position} --clump ${sumstatsdir}/sumstats_EUR.csv --clump-p1 1 --clump-r2 0.97 --out ${sumstatsdir}/EUR_1KG_chr${chr}_${random_position}_pruned
		#/expanse/lustre/projects/ddp412/kakamatsu/plink --bfile ${randomgenes}/EUR_1KG_chr${chr}_${random_position} --r2 inter-chr --ld-window-r2 0 --extract ${sumstatsdir}/EUR_1KG_chr${chr}_${random_position}_pruned.clumped --out ${lddir}/EUR_1KG_chr${chr}_${random_position}
		#/expanse/lustre/projects/ddp412/kakamatsu/plink --bfile ${randomgenes}/EUR_1KG_chr${chr}_${random_position} --r inter-chr --ld-window-r2 0 --keep-allele-order --extract ${sumstatsdir}/sumstats_EUR.csv --out ${lddir}/EUR_1KG_chr${chr}_${random_position}
	

		# AMR - LD created in Sim_geno.py
		#/expanse/lustre/projects/ddp412/kakamatsu/plink --bfile ${randomgenes}/AMR_1KG_chr${chr}_${random_position} --clump ${sumstatsdir}/sumstats_AMR.csv --clump-p1 1 --clump-r2 0.97 --out ${sumstatsdir}/AMR_1KG_chr${chr}_${random_position}_pruned
		#/expanse/lustre/projects/ddp412/kakamatsu/plink --bfile ${randomgenes}/AMR_1KG_chr${chr}_${random_position} --r2 inter-chr --ld-window-r2 0 --extract ${sumstatsdir}/AMR_1KG_chr${chr}_${random_position}_pruned.clumped --out ${lddir}/AMR_1KG_chr${chr}_${random_position}
		#/expanse/lustre/projects/ddp412/kakamatsu/plink --bfile ${randomgenes}/AMR_1KG_chr${chr}_${random_position} --r inter-chr --ld-window-r2 0 --keep-allele-order --extract ${sumstatsdir}/sumstats_AMR.csv --out ${lddir}/AMR_1KG_chr${chr}_${random_position}


		# run susie on summary statistics 
		Rscript ../../../run_susie_testnewld.R --sumstats_file ${sumstatsdir}/sumstats_EUR.csv --ld ${lddir}/EUR_LD.npz --ss 500 --out ${sumstatsdir}/sumstats_EUR_susie.csv
		Rscript ../../../run_susie_testnewld.R --sumstats_file ${sumstatsdir}/sumstats_AMR.csv --ld ${lddir}/AMR_LD.npz --ss 500 --out ${sumstatsdir}/sumstats_AMR_susie.csv

		
		# simulate AFR gene models with lasso 
		# use MAGEPRO 
		#python3 Sim_Model_multi.py <path to plink files> <sample size> <population> <simulated genotypes> <sumstats> <populations> <number of causal snps> <index of causal> <simulation number> <heritability of gene> <temporary directory> <output directory> <threads>
		python3 Sim_Model_corr.py ${randomgenes}/AFR_1KG_chr${chr}_${random_position} $numafr AFR ${simgeno}/simulated_genotypes_AFR.csv ${sumstatsdir}/sumstats_EUR_susie.csv,${sumstatsdir}/sumstats_AMR_susie.csv EUR,AMR $num_causal $causal_indices $i $heritability $tempdir/effects.txt $tempdir ${out} $threads

	done
done
