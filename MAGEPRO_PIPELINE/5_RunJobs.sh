#!/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH -t 10:00:00
#SBATCH -J MAGEPRO
#SBATCH -A csd832
#SBATCH -o ../working_err/MAGEPRO.%j.%N.out
#SBATCH -e ../working_err/MAGEPRO.%j.%N.err
#SBATCH --export=ALL
#SBATCH --constraint="lustre"
#--- EDIT ABOVE TO SUIT YOUR HPC CLUSTER

source ~/.bashrc
conda activate r_env
module load cpu/0.15.4 
module load parallel/20200822 

#--- read command line arguments
batch=$1
gefile=$2
scratch=$3 
intermed=$4
weights=$5
plink_exec=$6
gcta=$7
sumstats_dir=$8
sumstats=$9
models=${10}
ss=${11}
resid=${12}
hsq_p=${13}
lassohsq=${14}
hsq_set=${15}
crossval=${16}
verbose=${17}
noclean=${18}
save_hsq=${19}
ldref_pt=${20}
prune_r2=${21}
threshold_p=${22}
ldref_PRSCSx=${23}
dir_PRSCSx=${24}
phi_shrinkage_PRSCSx=${25}
pops=${26}
impact_path=${27}
ldref_dir=${28}
ldrefs=${29}
out_susie=${30}
skip_susie=${31}
n_threads=${32}


export plinkdir=$plinkdir
export plink_exec=$plink_exec
export wd=$wd
export ind=$ind
export geneids=$geneids
export gefile=$gefile
export colind2=$colind2
export tmpdir=$tmpdir
export weights=$weights
export gcta=$gcta
export sumstats_dir=$sumstats_dir
export sumstats=$sumstats
export models=$models
export ss=$ss
export resid=$resid
export hsq_p=$hsq_p
export lassohsq=$lassohsq
export hsq_set=$hsq_set
export crossval=$crossval
export verbose=$verbose
export noclean=$noclean
export save_hsq=$save_hsq
export ldref_pt=$ldref_pt
export prune_r2=$prune_r2
export threshold_p=$threshold_p
export ldref_PRSCSx=$ldref_PRSCSx
export dir_PRSCSx=$dir_PRSCSx
export phi_shrinkage_PRSCSx=$phi_shrinkage_PRSCSx
export pops=$pops
export impact_path=$impact_path
export ldref_dir=$ldref_dir
export ldrefs=$ldrefs
export out_susie=$out_susie
export skip_susie=$skip_susie
export intermed=$intermed


#--- create directory for temporary/working files 
#--- NOTE: create "tmp" and "wd" where your system can write/delete files efficiently (EDIT TO SUIT YOUR HPC CLUSTER)
#tmpdir=$scratch/tmp
tmpdir=/scratch/$USER/job_$SLURM_JOBID/tmp
mkdir $tmpdir
#wd=$scratch/wd
wd=/scratch/$USER/job_$SLURM_JOBID/wd
mkdir $wd

#--- define file paths 
plinkdir=$scratch/plink_gene
batchfile=$intermed/Genes_Assigned.txt 
geneids=$intermed/All_Genes_Expressed.txt 
ind=$intermed/All_Individuals.txt #all individuals with both genotype and ge data 
samples=$intermed/Sample_IDs.txt 

#--- get column numbers (in ge data) of individuals we are interested in
genes=$(awk '$2 == '${batch}' {print $1}' $batchfile) #all genes with the batch number passed in to the script 
#grabbing just the individuals we want
alldonors=$(zcat $gefile | head -n 1)
colind=$(echo $alldonors | sed 's| |\n|g' | nl | grep -f $samples | awk 'BEGIN {ORS=","} {print $1}') 
colind2=${colind%,} #remove last comma - at this point we have a comma delimited list of column numbers of individuals we want to extract from the ge data 

process_gene() {
    gene=$1
    filecheck=$plinkdir/$gene.bed
    if test -f "$filecheck"
        then
            $plink_exec --bfile $plinkdir/$gene --allow-no-sex --make-bed --keep $ind --indiv-sort f $ind --out $wd/$gene
            rm $wd/$gene.log 
            chr=$(cat ${wd}/${gene}.bim | head -n 1 | awk '{print $1}')
            rowid=$(cat $geneids | nl | grep $gene | awk '{print $1 + 1}') #+1 for col header in ge file - row number of the gene in the ge file 
            ge_donors=$(zcat $gefile | head -n $rowid | tail -n 1 | cut -f $colind2)  #gene expression from the individuals of interest
            paste --delimiters=' ' <(cut -d' ' -f1-5 $wd/$gene.fam) <(echo $ge_donors | sed 's/ /\n/g') > $wd/$gene.mod.fam #modifying fam file with ge data 
            mv $wd/$gene.mod.fam $wd/$gene.fam 
            TMP=$tmpdir/${gene}
            OUT=$weights/${gene}
            cmd="time Rscript MAGEPRO.R --gene $gene --bfile $wd/$gene --covar $intermed/Covar_All.txt --tmp $TMP --out $OUT --PATH_gcta $gcta --PATH_plink ${plink_exec} --sumstats_dir $sumstats_dir --sumstats $sumstats --models $models --ss $ss --resid $resid --hsq_p $hsq_p --lassohsq $lassohsq --hsq_set $hsq_set --crossval $crossval --verbose $verbose --noclean $noclean --save_hsq $save_hsq --ldref_pt $ldref_pt --prune_r2 $prune_r2 --threshold_p $threshold_p --ldref_PRSCSx $ldref_PRSCSx --dir_PRSCSx $dir_PRSCSx --phi_shrinkage_PRSCSx $phi_shrinkage_PRSCSx --pops $pops --impact_path ${impact_path}${chr}.txt --ldref_dir ${ldref_dir} --ldrefs ${ldrefs} --out_susie ${out_susie} --skip_susie ${skip_susie}"
            echo $cmd
            $cmd
            rm $wd/$gene.* 
    fi
}

export -f process_gene

echo Utilizing $n_threads threads

echo $genes

echo $genes | tr ' ' '\n' | parallel -j $n_threads process_gene
