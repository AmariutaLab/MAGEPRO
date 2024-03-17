#!/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH -t 03:00:00
#SBATCH -J MAGEPRO_impact_scale_weighted
#SBATCH -A ddp412
#SBATCH -o ../working_err/MAGEPRO_impact_scale_weighted.%j.%N.out
#SBATCH -e ../working_err/MAGEPRO_impact_scale_weighted.%j.%N.err
#SBATCH --export=ALL
#SBATCH --constraint="lustre"
#--- EDIT ABOVE TO SUIT YOUR HPC CLUSTER

source ~/.bashrc
conda activate r_env

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
cell_meta=${12}
resid=${13}
hsq_p=${14}
lassohsq=${15}
hsq_set=${16}
crossval=${17}
verbose=${18}
noclean=${19}
save_hsq=${20}
IMPACT=${21}

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

for gene in $genes
do
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
    #Rscript MAGEPRO_IMPACT.R --gene $gene --bfile $wd/$gene --covar $intermed/Covar_All.txt --tmp $TMP --out $OUT --PATH_gcta $gcta --PATH_plink ${plink_exec} --sumstats_dir $sumstats_dir --sumstats $sumstats --models $models --ss $ss --cell_meta $cell_meta --resid $resid --hsq_p $hsq_p --lassohsq $lassohsq --hsq_set $hsq_set --crossval $crossval --verbose $verbose --noclean $noclean --save_hsq $save_hsq --IMPACT ${IMPACT}${chr}.txt
    Rscript MAGEPRO_IMPACT_scale_nosplit_enet_weighted.R --gene $gene --bfile $wd/$gene --covar $intermed/Covar_All.txt --tmp $TMP --out $OUT --PATH_gcta $gcta --PATH_plink ${plink_exec} --sumstats_dir $sumstats_dir --sumstats $sumstats --models $models --ss $ss --cell_meta $cell_meta --resid $resid --hsq_p $hsq_p --lassohsq $lassohsq --hsq_set $hsq_set --crossval $crossval --verbose $verbose --noclean $noclean --save_hsq $save_hsq --IMPACT ${IMPACT}${chr}.txt
    rm $wd/$gene.* 
fi

done




