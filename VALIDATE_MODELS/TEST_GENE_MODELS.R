# load packages
suppressMessages(library("optparse"))
suppressMessages(library('plink2R'))
suppressMessages(library('glmnet'))
suppressMessages(library('methods'))
suppressMessages(library('data.table'))


option_list = list(
  make_option("--gene", action="store", default=NA, type='character',
              help="ENSG ID, for ex ENSG00000013503.9"),
  make_option("--bfile", action="store", default=NA, type='character',
              help="Path to PLINK binary input file prefix (minus bed/bim/fam) [required]"),
  make_option("--out", action="store", default=NA, type='character',
              help="Path to output files [required]"),
  make_option("--genemodel", action="store", default=NA, type='character',
              help="Path to gene model to apply"),
  make_option("--tmp", action="store", default=NA, type='character',
              help="Path to temporary files [required]"),
  make_option("--pheno", action="store", default=NA, type='character',
              help="Path to molecular phenotype file (PLINK format) [optional, taken from bfile otherwise]"),
  make_option("--PATH_plink", action="store", default="plink", type='character',
              help="Path to plink executable [%default]"),
  make_option("--PATH_gcta", action="store", default="gcta_nr_robust", type='character',
              help="Path to plink executable [%default]"),
  make_option("--covar", action="store", default=NA, type='character',
              help="Path to quantitative covariates (PLINK format) [optional]"),
  make_option("--resid", action="store_true", default=FALSE,
              help="Also regress the covariates out of the genotypes [default: %default]"),              
  make_option("--hsq_p", action="store", default=0.01, type='double',
              help="Minimum heritability p-value for which to compute weights [default: %default]"),
  make_option("--hsq_set", action="store", default=NA, type='double',
              help="Skip heritability estimation and set hsq estimate to this value [optional]"),
  make_option("--verbose", action="store", default=1, type="integer",
              help="How much chatter to print: 0=nothing; 1=minimal; 2=all [default: %default]"),
  make_option("--noclean", action="store_true", default=FALSE,
              help="Do not delete any temporary files (for debugging) [default: %default]"),
  make_option("--save_hsq", action="store_true", default=FALSE,
              help="Save heritability results even if weights are not computed [default: %default]") 
)


# --- PREPARING EXTERNAL DATASET FILES AND PARSING OPTIONS
# parse command-line arguments
opt = parse_args(OptionParser(option_list=option_list))
na <- which(opt == "NA") 
opt[na] <- NA
print(opt)

# find gene name without version number
#name <- strsplit(opt$gene, ".", fixed = TRUE)[[1]][1]

# path to weights  
file.wgts <- paste0(opt$genemodel,"/", opt$gene, ".wgt.RDat")

if ( opt$verbose == 2 ) {
  SYS_PRINT = F
} else {
  SYS_PRINT = T
}


# --- CLEANUP
cleanup = function() {
	if ( ! opt$noclean ) {
		arg = paste("rm -f " , opt$tmp , "*", sep='')
		system(arg)
	}
}


# --- i/o CHECKS
files = paste(opt$bfile,c(".bed",".bim",".fam"),sep='')
if ( !is.na(opt$pheno) ) files = c(files,opt$pheno)
if ( !is.na(opt$covar) ) files = c(files,opt$covar)

for ( f in files ) {
	if ( !file.exists(f) ){
		cat( "ERROR: ", f , " input file does not exist\n" , sep='', file=stderr() )
		cleanup()
		q()
	}
}

if ( system( paste(opt$PATH_plink,"--help") , ignore.stdout=T,ignore.stderr=T ) != 0 ) {
	cat( "ERROR: plink could not be executed, set with --PATH_plink\n" , sep='', file=stderr() )
	cleanup()
	q()
}


if ( opt$verbose == 2 ) cat("I/O checks complete \n")

# --- PREPARE PHENO, LOAD COVAR

#fam file read - the modified fam file version (added ge data)  
fam = read.table(paste(opt$bfile,".fam",sep=''),as.is=T)

# Make/fetch the phenotype file
if ( !is.na(opt$pheno) ) {
	pheno.file = opt$pheno
	pheno = read.table(pheno.file,as.is=T)
	# Match up data
	m = match( paste(fam[,1],fam[,2]) , paste(pheno[,1],pheno[,2]) )
	m.keep = !is.na(m)
	fam = fam[m.keep,]
	m = m[m.keep]
	pheno = pheno[m,]
} else { #creating pheno file from fam files
	pheno.file = paste(opt$tmp,".pheno",sep='')
	pheno = fam[,c(1,2,6)]
	write.table(pheno,quote=F,row.names=F,col.names=F,file=pheno.file)
}

if ( opt$verbose == 2 ) cat("Made/Fetched phenotype \n")

# Load in the covariates if needed
if ( !is.na(opt$covar) ) { 
	covar = ( read.table(opt$covar,as.is=T,head=T) )
	if ( opt$verbose >= 1 ) cat( "Loaded",ncol(covar)-2,"covariates\n")
	# Match up data
	m = match( fam[,2] , covar[,2] ) # THIS ONLY MATCHES THE SECOND COLUMN OF FAM TO SAMPLE IDS, leaves out family IDs
	m.keep = !is.na(m)
	fam = fam[m.keep,]
	pheno = pheno[m.keep,]
	m = m[m.keep]
	covar = covar[m,] #reordering covariates to match fam file
	reg = summary(lm( pheno[,3] ~ as.matrix(covar[,3:ncol(covar)]) ))  
	if ( opt$verbose >= 1 ) cat( reg$r.sq , "variance in phenotype explained by covariates\n" )
	pheno[,3] = scale(reg$resid) #regressing out covar to single out genetic effect
	raw.pheno.file = pheno.file
	pheno.file = paste(pheno.file,".resid",sep='')
	write.table(pheno,quote=F,row.names=F,col.names=F,file=pheno.file) #newly scaled pheno file 
}


if ( opt$verbose == 2 ) cat("Covariates loaded and regressed out of phenotype \n")

geno.file = opt$tmp
# recode to the intersection of samples and new phenotype
arg = paste( opt$PATH_plink ," --allow-no-sex --bfile ",opt$bfile," --pheno ",pheno.file," --keep ",pheno.file," --make-bed --out ",geno.file,sep='')
system(arg , ignore.stdout=SYS_PRINT,ignore.stderr=SYS_PRINT)

# --- LOAD AND PREP GENOTYPES 
genos = read_plink(geno.file,impute="avg")
mafs = apply(genos$bed,2,mean)/2  
sds = apply(genos$bed,2,sd)  
# important : genotypes are standardized and scaled here:
genos$bed = scale(genos$bed)
pheno = genos$fam[,c(1,2,6)]
pheno[,3] = scale(pheno[,3])

# check if any genotypes are NA
nasnps = apply( is.na(genos$bed) , 2 , sum )
if ( sum(nasnps) != 0 ) {
	cat( "WARNING :",sum(nasnps != 0),"SNPs could not be scaled and were zeroed out, make sure all SNPs are polymorphic\n" , file=stderr())
	genos$bed[,nasnps != 0] = 0
}

# regress covariates out of the genotypes as well (this is more accurate but slower)
if ( !is.na(opt$covar) && opt$resid ) {
	if ( opt$verbose >= 1 ) cat("Regressing covariates out of the genotypes\n")
	for ( i in 1:ncol(genos$bed) ) {
		genos$bed[,i] = summary(lm( genos$bed[,i] ~ as.matrix(covar[,3:ncol(covar)]) ))$resid
	}
	genos$bed = scale(genos$bed)
	genos$bed[is.nan(genos$bed)] <- 0 # when there is no variation at a snp, scale() sets genotypes of the snp to NaN. set those to 0
}

N.tot = nrow(genos$bed)
if ( opt$verbose >= 1 ) cat(nrow(pheno),"phenotyped samples, ",nrow(genos$bed),"genotyped samples, ",ncol(genos$bed)," markers\n")

# --- PROCESS GENE MODEL

datasets_process <- function(wgt, snp_ref){
	# wgt = vector of snp-gene weights to process 
	# snp_ref = bim file for population where wgt came from 
	# snp_geno = bim file for population to which we want to apply wgt 
	
	for (k in 1:length(wgt)){
		match=which(genos$bim$V2 == snp_ref$V2[k])
		if (! identical(match, integer(0))){
			if (!is.na(snp_ref$V5[k]) && !is.na(snp_ref$V6[k])){
				if (as.character(genos$bim$V5[match]) != as.character(snp_ref$V5[k]) || as.character(genos$bim$V6[match]) != as.character(snp_ref$V6[k])) {  
					if (as.character(genos$bim$V5[match]) == as.character(snp_ref$V6[k]) && as.character(genos$bim$V6[match]) == as.character(snp_ref$V5[k])){
						wgt[k] <- wgt[k] * -1
					}else{
						wgt[k] <- 0
					}
				}
			} else {
				wgt[k] <- 0
			}
		}
	}
	m <- match(genos$bim[,2], snp_ref[,2])
	wgt <- wgt[m]
	w <- which(is.na(wgt))
	if(length(w) > 0){wgt[w] <- 0}
	return (wgt)
}

#load in gene models
if (file.exists(file.wgts)){
	load(file.wgts)
	#colnames(wgt.matrix) <- c("LASSO","META", "PT", "SuSiE", "PRSCSx", "MAGEPRO_fullsumstats", "MAGEPRO") # for one iteration where "P+T" name was producing weird errors due to the "+" symbol
	weights <- c()
	for (c in 1:ncol(wgt.matrix)){
		assign(paste0("pred.wgt.", colnames(wgt.matrix)[c]), datasets_process(wgt.matrix[,c], snps))
		weights <- append(weights, paste0("pred.wgt.", colnames(wgt.matrix)[c]))
	}		
}else{
	cat("gene model does not exist. skipping...\n", file.wgts, "\n")
	cleanup()
	q()	
}


cv.all = pheno #note, cv.all is the gene expression phenotype for all 80 afr individuals for whole blood for ex 
n = nrow(cv.all)
cv.calls = matrix(NA,nrow=n,ncol=ncol(wgt.matrix)) 

# --- compute predictions
for (i in 1:length(weights)){
	cv.calls[ , i ] = genos$bed %*% eval(parse(text = weights[i]))
}


# --- compute rsq + P-value for each model
cv.performance = matrix(NA,nrow=2,ncol=ncol(wgt.matrix))  
rownames(cv.performance) = c("rsq","pval") 
colnames(cv.performance) = colnames(wgt.matrix) 

for ( mod in 1:ncol(cv.calls) ) { 
	if ( !is.na(sd(cv.calls[,mod])) && sd(cv.calls[,mod]) != 0 ) {   
		reg = summary(lm( cv.all[,3] ~ cv.calls[,mod] )) #r^2 between predicted and actual 
		cv.performance[ 1, mod ] = reg$adj.r.sq
		cv.performance[ 2, mod ] = reg$coef[2,4]
	} else {
		cv.performance[ 1, mod ] = NA
		cv.performance[ 2, mod ] = NA
	}
}
if ( opt$verbose >= 1 ) write.table(cv.performance,quote=F,sep='\t')

# --- HERITABILITY ANALYSIS (this can be used for downstream analysis... ex. should we analyze genes that are significantly heritable in both cohorts?)
if ( is.na(opt$hsq_set) ) {
	if ( opt$verbose >= 1 ) cat("### ESTIMATING HERITABILITY\n")

	# 1. generate GRM
	arg = paste( opt$PATH_plink," --allow-no-sex --bfile ",opt$tmp," --make-grm-bin --out ",opt$tmp,sep='')
	system(arg , ignore.stdout=SYS_PRINT,ignore.stderr=SYS_PRINT)

	# 2. estimate heritability
	if ( !is.na(opt$covar) ) {
		arg = paste( opt$PATH_gcta ," --grm ",opt$tmp," --pheno ",raw.pheno.file," --qcovar ",opt$covar," --out ",opt$tmp," --reml --reml-no-constrain --reml-lrt 1",sep='')
	} else {
		arg = paste( opt$PATH_gcta ," --grm ",opt$tmp," --pheno ",pheno.file," --out ",opt$tmp," --reml --reml-no-constrain --reml-lrt 1",sep='')
	}
	system(arg , ignore.stdout=SYS_PRINT,ignore.stderr=SYS_PRINT)

	# 3. evaluate LRT and V(G)/Vp
	#V(G)/Vp = proportion of genotypic to phenotypic variation
	#LRT - likelihood ratio test - compare goodness of fit 
	if ( !file.exists( paste(opt$tmp,".hsq",sep='') ) ) {
		if ( opt$verbose >= 1 ) cat(opt$tmp,"does not exist, GCTA could not converge, forcing h2 to fixed value\n",file=stderr()) 
		hsq = NA
		hsq.pv = NA
	}else{
		hsq.file = read.table(file=paste(opt$tmp,".hsq",sep=''),as.is=T,fill=T)
		hsq = as.numeric(unlist(hsq.file[hsq.file[,1] == "V(G)/Vp",2:3]))
		hsq.pv = as.numeric(unlist(hsq.file[hsq.file[,1] == "Pval",2]))
		if ( hsq[1] > 1 ){hsq = 1} #should not happen but gcta may overestimate at very low sample sizes
		if ( opt$verbose >= 1 ) cat("Heritability (se):",hsq,"LRT P-value:",hsq.pv,'\n')
		if ( opt$save_hsq ) cat( opt$out , hsq , hsq.pv , '\n' , file=paste(opt$out,".hsq",sep='') )

		# 4. stop if insufficient
		if (!is.na(hsq.pv)){
			if ( hsq.pv > opt$hsq_p ) { 
				cat(opt$tmp," : heritability ",hsq[1],"; LRT P-value ",hsq.pv," : skipping gene\n",sep='',file=stderr())
				cleanup()
				q()
			}
		}
	}
} else {
	if ( opt$verbose >= 1 ) cat("### SKIPPING HERITABILITY ESTIMATE\n")
	hsq = opt$hsq_set
	hsq.pv = NA
}


snps = genos$bim

save( snps, cv.performance, hsq, hsq.pv, file = paste( opt$out , ".wgt.RDat" , sep='' ) )

# --- CLEAN-UP
if ( opt$verbose >= 1 ) cat("Cleaning up\n")
cleanup()
