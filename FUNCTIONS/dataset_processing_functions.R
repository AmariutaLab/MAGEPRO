suppressMessages(library("optparse"))
suppressMessages(library('plink2R'))
suppressMessages(library('glmnet'))
suppressMessages(library('data.table'))
suppressMessages(library('dplyr'))
suppressMessages(library('susieR'))

# --- LOAD SUMSTATS AND FIX FLIPPED ALLELES

allele.qc = function(a1,a2,ref1,ref2) {
  a1 = toupper(a1)
  a2 = toupper(a2)
  ref1 = toupper(ref1)
  ref2 = toupper(ref2)
  
  ref = ref1
  flip = ref
  flip[ref == "A"] = "T"
  flip[ref == "T"] = "A"
  flip[ref == "G"] = "C"
  flip[ref == "C"] = "G"
  flip1 = flip
  
  ref = ref2
  flip = ref
  flip[ref == "A"] = "T"
  flip[ref == "T"] = "A"
  flip[ref == "G"] = "C"
  flip[ref == "C"] = "G"
  flip2 = flip;
  
  snp = list()
  snp[["keep"]] = !((a1=="A" & a2=="T") | (a1=="T" & a2=="A") | (a1=="C" & a2=="G") | (a1=="G" & a2=="C")) # ambiguous snps
  snp[["keep"]][ a1 != "A" & a1 != "T" & a1 != "G" & a1 != "C" ] = F # non ATGC
  snp[["keep"]][ a2 != "A" & a2 != "T" & a2 != "G" & a2 != "C" ] = F # non ATGC
  snp[["keep"]][ (a1==ref1 & a2!=ref2) | (a1==ref2 & a2!=ref1) | (a1==flip1 & a2!=flip2) | (a1==flip2 & a2!=flip1) ] = F # 3 alleles
  snp[["flip"]] = (a1 == ref2 & a2 == ref1) | (a1 == flip2 & a2 == flip1) # flips
  
  return(snp)
}

load_flip_dataset <- function(bim, dataset, file, datasets_loaded, select, susie = FALSE){
	# PURPOSE: read summary statistics, flip signs according to effect allele 
	# bim = bim file of snps
	# dataset = name of sumstat dataset
	# file = file path to sumstat (make sure it is properly formatted)
	# datasets_loaded = current list of loaded datasets
	# select = columns numbers to extract from file 
	# susie = true/false: using susie posterior weights? 
	# RETURN: updated list of loaded datasets, including the one processed here

	# read external summary statistics table 
	table <- fread(file, select = select) 
	# match by snps 
	m <- match(bim[,2], table[[1]])
	table <- table[m,]
	w <- which(is.na(table[[1]]))
	if(length(w) > 0){
		table[w,] <- 0
	}
	# determine effect sizes to flip, as well as those to remove (zero out)
	qc <- allele.qc(table[[2]], table[[3]], bim[,5], bim[,6])
	table[!qc$keep, 4] <- 0
	table[qc$flip, 4] <- table[qc$flip, 4] * -1
	if (susie){
		table[!qc$keep, 7] <- 0
		table[qc$flip, 7] <- table[qc$flip, 7] * -1
	}
	
	assign(paste0("loaded.", dataset), table, envir = .GlobalEnv)
	datasets_loaded <- append(datasets_loaded, paste0("loaded.", dataset))
	return(datasets_loaded)
}

# --- PROCESS-SUMSTATS
datasets_process_fullsumstats <- function(dataset, loaded, wgts){
	# PURPOSE: process effect sizes from full summary statistics
	# dataset = name of sumstat dataset
	# loaded = dataframe containing summary statistics from this dataset 
		# SNP(rsid) A1(allele1) A2(allele2) b(effect size) p(p-value) pip(pip from SuSiE) cs(credible set membersip from SuSiE) 
	# wgts = current list of the names of the variables containing these wgts
	# RETURN: updated list of wgts, including the one that was processed in this function
	assign(paste0("pred.wgt.", dataset), loaded[[4]], envir = .GlobalEnv)
	if(sum(which(eval(parse(text = paste0("pred.wgt.", dataset))) != 0)) > 0){
		wgts <- append(wgts, paste0("pred.wgt.", dataset))
		return (wgts)
	}else {
		return (wgts)
	}
}

# --- PROCESS-SUMSTATS-USE-SUSIE
datasets_process_susie <- function(dataset, loaded, wgts){
	# PURPOSE: process sumstats using SuSiE results: ONLY keep nonzero effect sizes for high PIP SNPs. 
	# dataset = name of sumstat dataset
	# loaded = dataframe containing summary statistics from this dataset
		# SNP(rsid) A1(allele1) A2(allele2) b(effect size) p(p-value) pip(pip from SuSiE) b(coef from SuSiE) cs(credible set membersip from SuSiE)
	# wgts = current list of the names of the variables containing these wgts
	# RETURN: updated list of wgts including the one that was processed in this function

	# STRATEGY:
		# SuSiE coefficients for all snps
	assign(paste0("pred.wgtsusie.", dataset), loaded[[7]], envir = .GlobalEnv)
	if(sum(which(eval(parse(text = paste0("pred.wgtsusie.", dataset))) != 0)) > 0){
		wgts <- append(wgts, paste0("pred.wgtsusie.", dataset))
		return (wgts)
	}else {
		return (wgts)
	}
}

datasets_process_prscsx <- function(name, loaded, working_dir){
	# PURPOSE = write summary statistics files to prepare to run PRS-CSx
	# name = name of the dataset
	# loaded = dataframe containing summary statistics from this dataset
		# SNP(rsid) A1(allele1) A2(allele2) b(effect size) p(p-value) pip(pip from SuSiE) cs(credible set membersip from SuSiE)
	# working_dir = temporary directory to write intermediate files for running PRS-CSx
	df <- data.frame(loaded[,c(1,2,3,4,5)])
	colnames(df) <- c("SNP", "A1", "A2", "BETA", "P")
	f_name <- paste0(working_dir, name, ".txt")
	write.table(df, file = f_name, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
}
