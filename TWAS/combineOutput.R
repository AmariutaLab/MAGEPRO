#combine TWAS results across all chromosomes into one file

library(data.table)
library(dplyr)

args <- commandArgs(trailingOnly=TRUE)
pheno = args[1]
gwas = args[2]
anc = args[3]
dataset = args[4]

models <- c("ALL", "MAGEPRO", "PRSCSx", "LASSO", "SuSiE")

pathbase <- paste0("output_", gwas, "_", anc, "_", dataset, "_")

for (model in models){
df <- fread(paste0(pathbase, model, "/", pheno, ".1.dat"), header = T)
for (chr in 2:22){
	table <- fread(paste0(pathbase, model, "/", pheno, "." , chr, ".dat"), header = T)
	df <- rbind(df, table)
}
system(paste0("rm -rf ", pathbase, model, "/", pheno))
system(paste0("mkdir ", pathbase, model, "/", pheno))
write.table(df, file=paste0(pathbase, model, "/", pheno, "/", pheno, "_", dataset, "_", model, ".combined.txt"), row.names = F, col.names = T, sep = "\t", quote = F)
command = paste0("scp kakamatsu@login.expanse.sdsc.edu:/expanse/lustre/projects/ddp412/kakamatsu/fusion_multipop/", pathbase, model, "/", pheno, "/", pheno, "_", dataset, "_", model, ".combined.txt", " C:/Users/kaiak/OneDrive/Desktop/AMARIUTALAB/PAPER_MAGEPRO_data/TWAS/", gwas, "/", anc, "/")
cat(command, file = "scp_commands.txt", sep = "\n", append = TRUE)
}
