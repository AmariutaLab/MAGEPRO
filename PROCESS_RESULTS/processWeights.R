library(data.table)
library(dplyr)

# sbatch runProcess.sh /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GENOA_target/mesaafr /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GENOA_target/mesaafr/out /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GENOA_target/mesaafr/intermediate/Genes_Assigned.txt MAGEPRO
args <- commandArgs(trailingOnly = TRUE)
output <- args[1] #path to output directory
weightsdir <- args[2] #path to directory where R data files are stored 
genes_assign <- args[3] #path to genes assigned file
models <- strsplit(args[4], split = ",")[[1]] #comma-separated list of models used (SINGLE,META,MAGEPRO)

num_models <- length(models)
genes <- read.table(file = genes_assign, header = F)$V1 #all genes in analysis
r2_h2 <- matrix(0, length(genes), 2*num_models+8)

# /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GEUVADIS_target/geuvadis_total/ 
# /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GEUVADIS_target/geuvadis_total/out
# /expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/GEUVADIS_target/eqtlgen/intermediate/Genes_Assigned.txt
# MAGEPRO


for (j in 1:length(genes)){

	file <- paste0(weightsdir,"/",genes[j],".wgt.RDat")
	if(file.exists(file)){
		if(file.info(file)$size > 0){
			load(file)
			#if (length(cv.performance[1,]) != num_models){ # this line is necessary for only this iteration of the analysis because I forgot to delete old files in the same weights directory 
			#next
			#}
			r2_h2[j,1] <- genes[j]
			r2_h2[j,2:(num_models+1)] <- cv.performance[1,]
			r2_h2[j,((num_models+2): (2*num_models+1) )] <- cv.performance[2,]
			r2_h2[j, 2*num_models+2] <- hsq[1]
			r2_h2[j, 2*num_models+3] <- hsq[2]
			r2_h2[j, 2*num_models+4] <- hsq.pv
			if (sum(is.na(wgtmagepro)) == 0){
				r2_h2[j, 2*num_models+5] <- paste(wgtmagepro, collapse = ",")
				r2_h2[j, 2*num_models+6] <- paste(cf_total, collapse = ",")
			}else{
				r2_h2[j, 2*num_models+5] <- NA
				r2_h2[j, 2*num_models+6] <-NA
			}
			r2_h2[j, 2*num_models+7] <- var_cov
			r2_h2[j, 2*num_models+8] <- avg_cor 
		}
	}
}
h <- which(r2_h2[,1] == 0)
if(length(h) > 0){r2_h2 <- r2_h2[-h,]}
newcolnamesr2h2 <- c("gene", paste0(models, "_r2"), paste0(models, "_pv"), "hsq", "hsq_se", "hsq.pv", "datasets", "alphas", "var_covariates", "avg_cor")
colnames(r2_h2) <- newcolnamesr2h2
write.table(r2_h2, file = paste0(output, "/MAGEPRO_results.txt"), row.names = F, col.names = T, sep = "\t", quote = F)
