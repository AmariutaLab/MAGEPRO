suppressMessages(library("optparse"))
suppressMessages(library("data.table"))
suppressMessages(library("dplyr"))
suppressMessages(library("susieR"))
suppressMessages(library('reticulate'))
np <- import("numpy")

# --- PARSE COMMAND LINE ARGUMENTS 

option_list = list(
    make_option("--sumstats_file", action="store", default=NA, type='character',
              help="Path to summary statistics"),
    make_option("--ld", action="store", default=NA, type='character',
              help="Path to ld file .npz"),
    make_option("--ss", action="store", default=NA, type='double',
              help="Sample-size of dataset"),
    make_option("--out", action="store", default=NA, type='character',
              help="Path to susie output")
)

opt = parse_args(OptionParser(option_list=option_list))

# read files 
df <- fread(opt$sumstats_file, header=TRUE)
#ld_stats <- fread(opt$ld_plink, header=TRUE, sep=" ")

# get unique SNPs that are both in ld matrix and gene file
#df <- df[df[[1]] %in% (union(ld_stats$SNP_A, ld_stats$SNP_B))]
snp_list <- df[[1]]

# build the ld matrix
#ld_matrix <- matrix(NA, nrow = length(snp_list), ncol = length(snp_list), dimnames = list(snp_list, snp_list))
#index_A <- match(ld_stats$SNP_A, snp_list)
#index_B <- match(ld_stats$SNP_B, snp_list)
#ld_matrix[cbind(index_A, index_B)] <- ld_stats$R
#ld_matrix[cbind(index_B, index_A)] <- ld_stats$R
#diag(ld_matrix) <- 1

npz <- np$load(opt$ld)
LD_array <- npz$f[["LD"]]
LD_r <- as.matrix(LD_array)
dimnames(LD_r) <- list(snp_list, snp_list)

print(nrow(df))
print(nrow(LD_r))
quit()

# run susie
res <- susie_rss(bhat=df[[4]], shat=df[[5]], R=LD_r, n=opt$ss)

# create credible set column
cs <- rep(-1, length(df[[5]]))
for(cname in names(res$sets$cs)) {
  number <- as.integer(gsub("L", "", cname))
  indices <- res$sets$cs[[cname]]
  cs[indices] <- number
}

df$pip <- res$pip
df$posterior <- coef(res)[-1]
df$cs <- cs
df <- df[order(df$cs, decreasing = TRUE), ]

colnames(df) = c("SNP", "A1", "A2", "BETA", "SE", "P", "PIP", "POSTERIOR", "CS")
fwrite(x=df, file=opt$out, sep="\t", col.names=TRUE, row.names=FALSE, quote=FALSE)

