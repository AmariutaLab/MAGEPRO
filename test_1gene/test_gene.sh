#scratch=/expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GEUVADIS_EUR_LCL/scratch_filtered_QC
#scratch=/expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GTExAFR_WHOLEBLOOD_MAGEPRO/scratch
#scratch=/expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GENOA_EA_LCL/scratch
scratch=/expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GENOA_AA_LCL/scratch
tmpdir=$scratch/tmp
wd=$scratch/wd
mkdir $tmpdir
mkdir $wd
plinkdir=$scratch/plink_gene
plink_exec=../../plink
#intermed=/expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GEUVADIS_EUR_LCL/intermediate_filtered_QC
#intermed=/expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GTExAFR_WHOLEBLOOD_MAGEPRO/intermediate
#intermed=/expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GENOA_EA_LCL/intermediate
intermed=/expanse/lustre/projects/ddp412/kakamatsu/GENE_MODELS/GENOA_AA_LCL/intermediate
batchfile=$intermed/Genes_Assigned.txt 
geneids=$intermed/All_Genes_Expressed.txt 
ind=$intermed/All_Individuals.txt #all individuals with both genotype and ge data 
samples=$intermed/Sample_IDs.txt 
#ge=/expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GEUVADIS/GE/GEUVADIS_normalized_processed_ge_EUR.bed.gz
#ge=/expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GTEX_ge_covar/Whole_Blood/Whole_Blood.v8.normalized_expression.bed.gz
#ge=/expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GENOA/GE/GENOA_EA_ge_normalized.bed.gz
ge=/expanse/lustre/projects/ddp412/kakamatsu/GENOTYPES_GE_data/GENOA/GE/GENOA_AA_ge_normalized.bed.gz

#--- get column numbers (in ge data) of individuals we are interested in
gene=$1
#grabbing just the individuals we want
alldonors=$(zcat $ge | head -n 1)
colind=$(echo $alldonors | sed 's| |\n|g' | nl | grep -f $samples | awk 'BEGIN {ORS=","} {print $1}') 
colind2=${colind%,} #remove last comma - at this point we have a comma delimited list of column numbers of individuals we want to extract from the ge data 
$plink_exec --bfile $plinkdir/$gene --allow-no-sex --make-bed --keep $ind --indiv-sort f $ind --out $wd/$gene # reorder based on $ind
rm $wd/$gene.log 
chr=$(cat ${wd}/${gene}.bim | head -n 1 | awk '{print $1}')
rowid=$(cat $geneids | nl | grep $gene | awk '{print $1 + 1}') #+1 for col header in ge file - row number of the gene in the ge file 
ge_donors=$(zcat $ge | head -n $rowid | tail -n 1 | cut -f $colind2)  #gene expression from the individuals of interest
paste --delimiters=' ' <(cut -d' ' -f1-5 $wd/$gene.fam) <(echo $ge_donors | sed 's/ /\n/g') > $wd/$gene.mod.fam #modifying fam file with ge data 
mv $wd/$gene.mod.fam $wd/$gene.fam
TMP=$tmpdir/${gene}
OUT=${gene}
#Rscript ../MAGEPRO_bridge.R --gene $gene --bfile $wd/$gene --covar $intermed/Covar_All.txt --tmp $TMP --out $OUT --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --PATH_plink ${plink_exec} --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_datasets_allinfo --sumstats eqtlgen --models SINGLE,META,MAGEPRO --ss 31684 --hsq_p 1 --verbose 1
#Rscript ../MAGEPRO_bridge_meta_target.R --gene $gene --bfile $wd/$gene --covar $intermed/Covar_All.txt --tmp $TMP --out $OUT --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --PATH_plink ${plink_exec} --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_datasets_allinfo --sumstats eqtlgen,mesahis,mesaafr --models SINGLE,META,MAGEPRO --ss 31684,352,233 --hsq_p 1 --verbose 1
Rscript ../MAGEPRO_IMPACT_enet_important.R --gene $gene --bfile $wd/$gene --covar $intermed/Covar_All.txt --tmp $TMP --out $OUT --PATH_gcta /expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust --PATH_plink ${plink_exec} --sumstats_dir /expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_datasets_allinfo --sumstats eqtlgen,mesahis,mesaafr --models SINGLE,META,MAGEPRO --ss 31684,352,233 --hsq_p 1 --verbose 1 --IMPACT /expanse/lustre/projects/ddp412/kakamatsu/IMPACT_TRACKS/EUR_processed/means_LCL/IMPACT_707_LCL_HM3_EUR_chr${chr}.txt
rm $wd/$gene.* 
