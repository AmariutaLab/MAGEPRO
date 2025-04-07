suppressMessages(library("optparse"))
suppressMessages(library('plink2R'))
suppressMessages(library('glmnet'))
suppressMessages(library('data.table'))
suppressMessages(library('dplyr'))
suppressMessages(library('susieR'))

# --- PREDICTION MODELS
# Elastic Net
weights.enet = function( genos , pheno , alpha=0.5 ) {
	eff.wgt = matrix( 0 , ncol=1 , nrow=ncol(genos) )
	# remove monomorphics
	sds = apply( genos  , 2 , sd )
	keep = sds != 0 & !is.na(sds)
	enet = cv.glmnet( x=genos[,keep] , y=pheno , alpha=alpha , nfold=5 , intercept=T , standardize=F )
	eff.wgt[ keep ] = coef( enet , s = "lambda.min")[2:(sum(keep)+1)]
	return( eff.wgt )
}

# PLINK: LASSO
weights.lasso = function( input , hsq , snp , out=NA ) {
	# PURPOSE: perform LASSO regression using plink --lasso flag
	# input = plink file (phenotype in fam file)
	# hsq = heritability estimate used for calibrating regression
	# snp = vector of snps used in the gene model
	# RETURN: vector of effect sizes for each snp
	if ( is.na(out) ) out = paste(input,".LASSO",sep='')
	arg = paste( opt$PATH_plink , " --allow-no-sex --bfile " , input , " --lasso " , hsq , " --out " , out , sep='' )
	system( arg , ignore.stdout=SYS_PRINT,ignore.stderr=SYS_PRINT )
	if ( !file.exists(paste(out,".lasso",sep='')) ) {
	cat( paste(out,".lasso",sep='') , " LASSO output did not exist\n" )
	eff.wgt = rep(NA,length(snp))
	} else {
	eff = read.table( paste(out,".lasso",sep=''),head=T,as.is=T)
	eff.wgt = rep(0,length(snp))
	m = match( snp , eff$SNP )
	m.keep = !is.na(m)	
	m = m[m.keep]
	eff.wgt[m.keep] = eff$EFFECT[m]
	}
	return( eff.wgt )
}

# Marginal Z-scores (used for top1)
weights.marginal = function( genos , pheno , beta=F ) {
	# PURPOSE: compute marginal effect size estimates 
	# genos = plink genotype bed
	# pheno = phenotype as a matrix (1 column)
	# RETURN: vector of effect sizes for each snp
	if ( beta ) eff.wgt = t( genos ) %*% (pheno) / ( nrow(pheno) - 1)
	else eff.wgt = t( genos ) %*% (pheno) / sqrt( nrow(pheno) - 1 )
	return( eff.wgt )
}

weights.meta = function(target_model, target_ss, wgts, h, total_ss) {
	# PURPOSE: ss-weighted meta-analysis of effect sizes from datasets and target
	# target_model = vector of target population effect sizes
	# target_ss = sample size used to compute the target_model
	# wgts = vector of the names (string) of external dataset vectors to use 
	# h = hashmap that maps the dataset name to sample size 
	# total_ss = total sample size of all datasets and target_ss

	pred.wgt.meta <- target_model * (target_ss/total_ss)
	for (w in wgts){
		dataset <- strsplit(w, split="[.]")[[1]][3]
		size <- h[[dataset]]
		pred.wgt.meta = pred.wgt.meta + (eval(parse(text=w)) * (size/total_ss))
	}
	if (length(pred.wgt.meta) == 1){
		pred.wgt.meta <- t(pred.wgt.meta) 
	}
	return (pred.wgt.meta)
}


tune.pt = function(geno_bed, geno_bim, pheno_df, r2_choices, p_choices, cross, ldref, tmp){
	# PURPOSE: perform 5-fold cross validation on individuals in geno-bed and pheno to fine the best r2 and p parameter out of given choices 
	# geno_bed = genotype matrix (row = people, col = snps)
	# geno_bim = bim file for target population (snps correspond to geno_bed)
	# pheno_df = phenotype data for individuals in geno_bed (dataframe with phenotype in third column)
	# r2_choices = vector of r2 thresholds to search over 
	# p_choices = vector of p value thresholds to search over 
	# cross = number of cross validation folds
	# ldref = ld reference file for pruning and thresholding 
	# tmp = path/name for temporary files	
	# RETURN: c(r2_best, p_best): a vector holding the best r2 and p value threshold 
	# 1. create a matrix of size |r2_choices| x |p_choices|
		# holds the r-squared of the model created using i_th r2 and j_th p 
	performance = matrix(NA, nrow=length(r2_choices), ncol=length(p_choices))
	rownames(performance) <- r2_choices
	colnames(performance) <- p_choices
	# 2. define indices of cross validation 
	all_phenos = pheno_df
	sample_size = nrow(all_phenos)
	index_sample = sample(sample_size) #sample randomly 
	all_phenos = all_phenos[index_sample,]
	folds = cut(seq(1,sample_size),breaks=cross,labels=FALSE)
	# 3. for every fold
	calls = matrix(NA,nrow=sample_size,ncol=(length(r2_choices)*length(p_choices))) # reset calls for every r2/p pair 
	for ( fold in 1:cross ) {
		fold_indx = which(folds==fold,arr.ind=TRUE)
		train_phenos = all_phenos[-fold_indx,]
		train_phenos[,3] = scale(train_phenos[,3]) 
		geno_bed_train = geno_bed[index_sample[-fold_indx], , drop = FALSE]	
		# 4. compute betas in training fold
		sumstats <- eqtl_assoc(geno_bed_train, geno_bim, train_phenos[,3])	
		# 5. for every r2, p pair 
		colnum = 1
		for (i in 1:length(r2_choices)){
			for (j in 1:length(p_choices)){
				r2_current = r2_choices[i]
				p_current = p_choices[j]
				# P+T
	       			current_pred = prune_thresh(sumstats, geno_bim, r2_current, p_current, ldref, tmp)
				# predict 
				if (sum(is.na(current_pred)) == nrow(geno_bim)){
                        		calls[fold_indx, colnum] = NA
                		}else{
                        		calls[fold_indx, colnum] <- geno_bed[index_sample[fold_indx], , drop = FALSE] %*% current_pred
				}
				colnum = colnum + 1
			}
		}
	}
	# 6. compute r-squared of predictions and store in matrix
	colnum = 1
	for (i in 1:length(r2_choices)){
		for (j in 1:length(p_choices)){
			if ( !is.na(sd(calls[,colnum])) && sd(calls[,colnum]) != 0 ) {
				summ = summary(lm( all_phenos[,3] ~ calls[,colnum] )) #r^2 between predicted and actual
				performance[i,j] = summ$adj.r.sq
                        }else{
                                performance[i,j] = NA
                        }
			colnum = colnum + 1
		}
	}
	# 7. return the r2 and p that results in the maximum r-squared
	max_index <- which(performance == max(performance), arr.ind=TRUE)
	if (nrow(max_index) > 0){
		row_max <- max_index[1,1]
		col_max <- max_index[1,2]
		best_r2_p <- c(r2_choices[row_max], p_choices[col_max])
	}else{
		best_r2_p <- c(0.2,0.5) # if none of the tuned parameters result in any clumps, just try 0.2 and 0.5
	}
	return(best_r2_p)
}

eqtl_assoc = function(geno_bed, geno_bim, pheno) {
	# PURPOSE: compute effect sizes of eQTL
	# geno_bed = genotype matrix (row = people, col = snps)
	# geno_bim = bim file for target population
	# pheno = phenotype of interest across people
	
	Betas <- c()
        Pvals <- c()
        SNPs <- c()
	for(col in 1:ncol(geno_bed)){
                SNPs <- append(SNPs, colnames(geno_bed)[col])
                snp <- geno_bed[,col]
                model <- lm(pheno ~ snp)
                b <- coef(model)[2]
                if (is.na(b)){
                        b = 0
                }
                Betas <- append(Betas, b)
                if ( nrow(summary(model)$coefficients) > 1 ){
                        Pvals <- append(Pvals, summary(model)$coefficients[2, "Pr(>|t|)"])
                }else{
                        Pvals <- append(Pvals, 1)
                }
        }
        sumstats_pt <- data.frame(SNPs, Betas, Pvals)
	return (sumstats_pt)
}

prune_thresh = function(sumstats_pt, geno_bim, r2, p, ldref, tmp) {
	# PURPOSE: run pruning+thresholding on dataframe with rsid, betas, pval
	# sumstats_pt = dataframe with SNPs, Betas, Pvals as columns
	# geno_bim = bim file for target population
	# r2 = LD pruning threshold 
	# p = p-value threshold 
	# ldref = path/file to plink ldref file for P+T
	# tmp = path/name for temporary file
	
	#write dataframe, run plink --clump to P+T, write output to temp directory
        sumstats_file = paste(tmp,"_sumstats.txt",sep='')
        write.table(sumstats_pt, file = sumstats_file, quote = F, row.names = F, col.names = T, sep = '\t')
        arg = paste( opt$PATH_plink ," --allow-no-sex --bfile ", ldref, geno_bim[1,1], " --clump-p1 ", p ," --clump-r2 ", r2 ," --clump-snp-field SNPs --clump-field Pvals --clump ", tmp, "_sumstats.txt"," --out ", sumstats_file,sep='')
        system(arg, ignore.stdout=SYS_PRINT,ignore.stderr=SYS_PRINT)	

	#read in result file, create wgt matrix and predict
        if (file.exists(paste0(sumstats_file, ".clumped"))){
        	clumped <- fread(file = paste0(sumstats_file, ".clumped"), header = T)
		clumps <- clumped[[3]]
		sumstats_pt <- sumstats_pt[which(sumstats_pt$SNPs %in% clumps),]
		m_sumstats <- match(geno_bim[,2], sumstats_pt[[1]])
                sumstats_pt <- sumstats_pt[m_sumstats,]
		w_sumstats <- which(is.na(sumstats_pt[[1]]))
                if(length(w_sumstats) > 0){
                        sumstats_pt[w_sumstats,2] <- 0
                }
                pred.wgt.PT_sumstats = sumstats_pt[[2]]
        }else{
                pred.wgt.PT_sumstats = rep(NA, times = nrow(geno_bim))
        }
        if(length(pred.wgt.PT_sumstats) == 1){
               pred.wgt.PT_sumstats <- t(pred.wgt.PT_sumstats)
        }
	system(paste0("rm -rf ", sumstats_file, ".clumped"))
	return (pred.wgt.PT_sumstats)

}

weights.pt = function(geno_bed, geno_bim, pheno, r2, p, ldref, tmp) {
	# PURPOSE: compute effect sizes of eQTL and run pruning+thresholding
	# geno_bed = genotype matrix (row = people, col = snps)
	# geno_bim = bim file for target population
	# pheno = phenotype of interest across people
	# r2 = LD pruning threshold 
	# p = p-value threshold 
	# ldref = path/file to plink ldref file for P+T
	# tmp = path/name for temporary file
	sumstats_pt <- eqtl_assoc(geno_bed, geno_bim, pheno)
	pred.wgt.PT_sumstats <- prune_thresh(sumstats_pt, geno_bim, r2, p, ldref, tmp)
	return (pred.wgt.PT_sumstats)
}

weights.susie = function(geno, pheno){
	# PURPOSE: fit the sum of single effects model and return coefficients for prediction
	# geno = nxp genotype matrix 
	# pheno = n-length vector of phenotype 
	# RETURN: p-length vector of effect sizes

	fitted <- susie(geno, pheno)
	coefs <- coef(fitted)[-1]
	pred.wgt.susie <- coefs
	return(pred.wgt.susie)

}


weights.susie_impact = function(geno, pheno, impact_file){
	# PURPOSE: fit the sum of single effects model and return coefficients for prediction
	# geno = nxp genotype matrix 
	# pheno = n-length vector of phenotype 
	# impact_file = path to file with impact scores for every SNP
	# RETURN: p-length vector of effect sizes
	# use IMPACT scores as priors for SuSiE
	
	snps <- colnames(geno)

	# read in impact scores, take snps in geno 
	impact <- fread(impact_file)
	impact_ordered <- impact %>% 
		filter(SNP %in% snps) %>%
		arrange(match(SNP, snps))

	fitted <- susie(geno, pheno, prior_weights = impact_ordered$means)
	coefs <- coef(fitted)[-1]
	pred.wgt.susie <- coefs
	return(pred.wgt.susie)

}


shrinkage.prscsx = function(exec_dir, ldref, working_dir, shrinkage, input, ss, pp, bim) {
	# PURPOSE: run PRS-CSx shrinkage and retrieve list of weights
	# exec_dir: directory where PRScsx.py executable is 
	# ldref: directory where ldref for PRS-CSx is (see PRS-CSx github for preparation)
	# working_dir: directory where external summary statistics files for PRS-CSx and snps list is written 
	# input: comma separated sequence of path/file of summary statistics for PRS-CSx
	# ss: comma separated sequence of sample sizes of summary statistics 
	# pp: comma separated sequence of ancestries of summary statistics
	# bim: bim file of target population 
	
	write.table(bim, file = paste0(working_dir, "snps.bim"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
	arg = paste0("python ", exec_dir, "/PRScsx.py --ref_dir=", ldref, " --bim_prefix=", working_dir, "snps", " --sst_file=", input," --n_gwas=", ss, " --pop=", pp, " --chrom=", bim[1,1]," --phi=", shrinkage," --out_dir=", working_dir, " --out_name=results")
	if ( opt$verbose >= 1 ) print(arg)
	system(arg)
	
	wgts <- c()
	unique_pp <- unique(strsplit(pp, split = ",")[[1]])
	for (n in unique_pp){
		wgts <- weights_process_prscsx(n, paste0(working_dir,"results_", n, "_pst_eff_a1_b0.5_phi", sub(".00e", "e", sprintf("%.2e", shrinkage)) ,"_chr", bim[1,1], ".txt"), wgts, bim)
	}
	return (wgts)

}

weights_process_prscsx <- function(dataset, file, wgts, bim){
        if (file.size(file) != 0L){
                table <- fread(file)
                m <- match(bim[,2], table[[2]])
                table <- table[m,]
                w <- which(is.na(table[[2]]))
                if(length(w) > 0){
                        table[w,6] <- 0
                }
                assign(paste0("pred.wgtprscsx.", dataset), table[[6]], envir = .GlobalEnv)
                if(sum(which(eval(parse(text = paste0("pred.wgtprscsx.", dataset))) != 0)) > 0){
                        wgts <- append(wgts, paste0("pred.wgtprscsx.", dataset))
                        return (wgts)
                }else {
                        return (wgts)
                }
        }else{
                return (wgts)
        }
}

weights.prscsx = function(wgts, geno, pheno, target_model = NA){
	# PURPOSE: compute optimal linear combination of post-shrinkage weights from prscsx
	# wgts: list of strings, name of variables holding post-shrinkage weights 
	# geno: genotype matrix of target cohort 
	# pheno: vector of phenotypes of target cohort, for individuals in geno
	# target_model: name of variable holding gene model for target population (if provided, it is added as one of the features in the linear combination)
	ext <- length(wgts)
	eq = list()
	if (!is.na(target_model)){
		eq <- append(eq, paste0("geno %*% ", target_model))
	}
        for (w in wgts){
                if (length(eval(parse(text = w))) == 1){
                       assign(w, t(eval(parse(text = w))))
                }
                eq <- append(eq, paste0("geno %*% ", w))
        }
	eq <- paste(eq, collapse="+")
        eq <- paste0("pheno ~ ", eq)
	y <- lm(eval(parse(text = eq)))
	coef <- coef(y)
	coef <- ifelse(is.na(coef), 0, coef)
	w_eq = list()
        if (!is.na(target_model)){
		w_eq <- append(w_eq, paste0("coef[", 2,"]*",target_model))
		for (nums in 3:(ext+2)){
                	w_eq <- append(w_eq, paste0("coef[", nums,"]*",wgts[nums-2]))
        	}
	}else{
		for (nums in 2:(ext+1)){
                	w_eq <- append(w_eq, paste0("coef[", nums,"]*",wgts[nums-1]))
        	}
	}
        w_eq <- paste(w_eq, collapse="+")
        pred.wgt.prs_csx <- eval(parse(text=w_eq))
        if (length(pred.wgt.prs_csx) == 1){
                pred.wgt.prs_csx <- t(pred.wgt.prs_csx)
        }
	return(pred.wgt.prs_csx)
}

weights.magepro_marquezluna = function(basemodel, wgts, geno, pheno_df, geno_bim, tmp, lasso_h2, save_alphas) {
	# PURPOSE: compute MAGEPRO weights, use MarquezLuna approach to compute Betas and alphas in the same training fold
	# basemodel = target population prediction model, a vector of effect sizes (this base model will be one of the features in the regression)
	# wgts = list of name of variables holding magepro_processed weights 
	# geno = genotype matrix for training mixing weights 
	# pheno_df = data frame with gene expression values for individuals in geno in column 3 (also for training mixing weights)
	# geno_bim = bim file of target population
	# tmp = directory/name for temporary files
	# lasso_h2 = heritability of gene, used for plink --lasso
	# save_alphas = T/F, save alpha coefficients to 'cf_total'?
	ext <- length(wgts)
	#1a. format glmnet input
	eq <- matrix(0, nrow = nrow(geno), ncol = ext+1)
	# 10-fold cv to compute betas and "predict" on all people
	all_phenos <- pheno_df
	sample_size <- nrow(all_phenos)
	index_sample <- sample(sample_size) #sample randomly 
	all_phenos <- all_phenos[index_sample,]
	folds <- cut(seq(1,sample_size),breaks=10,labels=FALSE)
	for ( fold in 1:10 ) {
		fold_indx = which(folds==fold,arr.ind=TRUE)
		train_phenos = all_phenos[-fold_indx,]
		train_phenos[,3] = scale(train_phenos[,3]) 
		geno_bed_train = geno[index_sample[-fold_indx], , drop = FALSE]	
		train.file = paste(tmp,".cv_nested",sep='')
		write.table( train_phenos , quote=F , row.names=F , col.names=F , file=paste(train.file,".keep",sep=''))
		arg = paste(opt$PATH_plink ," --allow-no-sex --bfile ",tmp," --keep ",train.file,".keep --out ",train.file," --make-bed",sep='')
		system(arg , ignore.stdout=SYS_PRINT,ignore.stderr=SYS_PRINT)
		pred.wgt_current = weights.lasso( train.file , lasso_h2 , snp=geno_bim[,2] )
		if ( sum(is.na(pred.wgt_current)) == nrow(geno_bim)) {
			pred.wgt_current = weights.marginal( geno_bed_train , as.matrix(train_phenos[,3,drop=F]) , beta=T )
			pred.wgt_current[ - which.max( pred.wgt_current^2 ) ] = 0
		}
		if (length(pred.wgt_current) == 1){
			pred.wgt_current <- t(pred.wgt_current) # 1 snp in the cis window -> transpose for "matrix" mult
		}
		if (sum(is.na(pred.wgt_current)) == nrow(geno_bim)){
                        eq[index_sample[fold_indx], 1] = NA
                }else{
			eq[index_sample[fold_indx], 1] <- geno[index_sample[fold_indx], , drop = FALSE] %*% pred.wgt_current
		}
	}
	for (c in 1:length(wgts)){
		eq[,(c+1)] <- geno %*%  eval(parse(text = wgts[c])) 
	}	
	#1b. when geno %*% wgt -> all 0 (snps at nonzero wgt have no variaion among people) -> remove sumstat
	zero_cols <- colSums(eq == 0) == nrow(eq) 
	if (any(zero_cols)) {
  		eq <- eq[, !zero_cols]
  		ext <- ext - sum(zero_cols)
		wgts <- wgts[-(which(zero_cols) - 1)]
	}
	#2. run ridge regression to find optimal coefficients for each dataset
	y <- cv.glmnet(x = eq , y = pheno_df[,3], alpha = 0, nfold = 5, intercept = T, standardize = T)
	cf <- coef(y, s = "lambda.min")[2:(ext+2)]
	predtext <- "cf[1]*basemodel"
	for (i in 2:(length(cf))){
		predtext <- paste0(predtext, " + cf[", i, "]*", wgts[(i-1)])
	}
	pred.wgt.magepro <- eval(parse(text = predtext))
	if(length(pred.wgt.magepro) == 1){
		pred.wgt.magepro <- t(pred.wgt.magepro)
	}
	if(save_alphas) cf_total <<- cf
	return(pred.wgt.magepro)
}

weights.magepro = function(basemodel, wgts, geno, pheno, save_alphas) {
	# PURPOSE: compute MAGEPRO weights
	# basemodel = target population prediction model, a vector of effect sizes (this base model will be one of the features in the regression)
	# wgts = list of name of variables holding magepro_processed weights 
	# geno = genotype matrix for training mixing weights 
	# pheno = vector of gene expression values for individuals in geno (also for training mixing weights)
	# save_alphas = T/F, save alpha coefficients to 'cf_total'?
	ext <- length(wgts)
	#1. format glmnet input
	eq <- matrix(0, nrow = nrow(geno), ncol = ext+1)
	eq[,1] <- geno %*% basemodel
	
	for (c in 1:length(wgts)){
		eq[,(c+1)] <- geno %*%  eval(parse(text = wgts[c])) 
	}	
	#2. run ridge regression to find optimal coefficients for each dataset
	y <- cv.glmnet(x = eq , y = pheno, alpha = 0, nfold = 5, intercept = T, standardize = T)
	cf = coef(y, s = "lambda.min")[2:(ext+2)]
	predtext <- "cf[1]*basemodel"
	for (i in 2:(length(cf))){
		predtext <- paste0(predtext, " + cf[", i, "]*", wgts[(i-1)])
	}
	pred.wgt.magepro <- eval(parse(text = predtext))
	if(length(pred.wgt.magepro) == 1){
		pred.wgt.magepro <- t(pred.wgt.magepro)
	}
	if(save_alphas) cf_total <<- cf
	return(pred.wgt.magepro)
}


weights.magepro_localx = function(basemodel, wgts, geno, pheno, pcs, save_alphas) {
	# PURPOSE: compute MAGEPRO weights with local ancestry interactions
	# basemodel = target population prediction model, a vector of effect sizes (this base model will be one of the features in the regression)
	# wgts = list of name of variables holding magepro_processed weights 
	# geno = genotype matrix for training mixing weights 
	# pheno = vector of gene expression values for individuals in geno (also for training mixing weights)
	# pcs = local genotype pcs to allow interactions with wgts
	# save_alphas = T/F, save alpha coefficients to 'cf_total'?
	ext <- length(wgts) # number of external summary statistics 
	numpcs <- ncol(pcs) # number of local genotype pcs 
	#1. format glmnet input
	eq <- matrix(0, nrow = nrow(geno), ncol = ext+1+(ext*numpcs))
	eq[,1] <- geno %*% basemodel
	for (c in 1:ext){
		eq[,(c+1)] <- geno %*%  eval(parse(text = wgts[c])) 
	}
	for (c in 1:ext){
		for (l in 1:numpcs){
			nextindex = (((c-1)*numpcs)+1+ext+l)
			eq[,nextindex] <- ( geno %*%  eval(parse(text = wgts[c])) ) * pcs[,l] # interaction
		}
	}
	#2. run ridge regression to find optimal coefficients for each dataset
	y <- cv.glmnet(x = eq , y = pheno, alpha = 0, nfold = 5, intercept = T, standardize = T)
	cf = coef(y, s = "lambda.min")[2:(ext+(ext*numpcs)+2)]
	predtext <- "cf[1]*basemodel"
	for (i in 2:(ext+1)){
		predtext <- paste0(predtext, " + cf[", i, "]*", wgts[(i-1)])
	}
	interaction_effects <- cf[(ext+2):length(cf)] # coefficients for interactions
	interaction_effect_dict <- list()
	for (c in 1:ext){
		for (l in 1:numpcs){
			interaction_effect_dict[[paste0(wgts[c], as.character(l))]] <- interaction_effects[((c-1)*numpcs)+l]
		}
	}
	pred.wgt.magepro <- eval(parse(text = predtext)) # linearly combined base + external datasets 
	if(length(pred.wgt.magepro) == 1){
		pred.wgt.magepro <- t(pred.wgt.magepro)
	}
	if(save_alphas) cf_total <<- cf
	return(list(pred.wgt.magepro, interaction_effect_dict))
}

