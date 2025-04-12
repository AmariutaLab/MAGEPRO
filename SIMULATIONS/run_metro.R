suppressMessages(library("optparse"))
suppressMessages(library("data.table"))
suppressMessages(library("dplyr"))
suppressMessages(library("METRO"))
cat("Running run_metro.r\n")

# NOTE THAT BY DEFAULT WE USE THE TARGET EQTL COHORT LD AS THE GWAS LD
arg_parser <- function() {
    option_list <- list(
        make_option(c("-e", "--eqtl_files"), type = "character",
                    help = "Comma-separated string of marginal eQTL summary statistics file paths", metavar = "eQTLsumstats"),
        make_option(c("-p", "--pop_names"), type="character",
                    help="Names of populations", metavar='PopNames'),
        make_option(c("-l", "--ld_matrices"), type = "character",
                    help = "Comma-separated string of LD matrix file paths", metavar = "eQTLlds"),
        make_option(c("-s", "--ns_of_people"), type = "character",
                    help = "Comma-separated string of samples sizes for each eQTL cohort", metavar="eQTLns"),
        make_option(c("-g", "--gwas_file"), type = "character",
                    help = "Path to GWAS summmary statistics", metavar="GWASsumstat"),
        make_option(c("-n", "--ngwas_of_people"), type = "integer",
                    help = "Integer indicating the number of individuals in GWAS summary statistics", metavar="GWASn"),
        make_option(c("-o", "--output_folder"), type="character", help="output folder name", metavar="OUTPUTFOLDER")
    )

    opt_parser <- OptionParser(option_list = option_list)
    opt <- parse_args(opt_parser)

    if (is.null(opt$eqtl_files)) {
        stop("eQTL summary statistics (-e, --eqtl_files) are required", call. = FALSE)
    }
    if (is.null(opt$ld_matrices)) {
        stop("LD Matrices (-l, --ld_matrices) are required", call. = FALSE)
    }
    if (is.null(opt$ns_of_people)) {
        stop("Number of people/sample sizes in eQTL cohorts (-s, --ns_of_people) are required", call. = FALSE)
    }
    if (is.null(opt$gwas_file)) {
        stop("GWAS summary statistics file (-g, --gwas_file) is required", call. = FALSE)
    }
    if (is.null(opt$ngwas_of_people)) {
        stop("GWAS sample size (-n, --ngwas_of_people) is required", call. = FALSE)
    }
    if (is.null(opt$output_folder)) {
        stop("Output folder (-o, --output_folder) is required", call. = FALSE)
    }
    return(opt)
}
# load("PLTP_GEUVADIS.RData")
# ls()
opt <- arg_parser()

eQTL_paths = strsplit(opt$eqtl_files, split = ",")[[1]]
eQTL_LD_paths = strsplit(opt$ld_matrices, split = ",")[[1]]
eQTL_samplesizes = lapply(strsplit(opt$ns_of_people, split = ","), as.numeric)[[1]]
pop_names = strsplit(opt$pop_names, split=",")[[1]]
cat("Current METRO gene is ", opt$output_folder, "\n")
# read eQTL files
eQTL_dfs = list()
for (path in eQTL_paths){
    df <- fread(path, header=TRUE, dec=".")
    colnames(df) = c("Gene", "SNP", "A1", "A2", "BETA", "SE", "P")
    if (length(eQTL_dfs) > 0){
        df <- df[match(eQTL_dfs[[length(eQTL_dfs)]]$SNP, df$SNP), ] # match snp order
        if (!all(df$SNP == eQTL_dfs[[length(eQTL_dfs)]]$SNP)){
            sys.exit('eQTL files contain different snps')
        }
    }
    eQTL_dfs = append(eQTL_dfs, list(df))
}




snp_list = eQTL_dfs[[1]]$SNP
# print(eQTL_dfs)
# read LD files
eQTLLDs = list()
for (i in seq_along(eQTL_LD_paths)){
    path <- eQTL_LD_paths[i]
    pop <- pop_names[i]
    ld_df <- fread(path, header=TRUE, dec=".")
    # build matrix
    ld_matrix <- matrix(NA, nrow = length(snp_list), ncol = length(snp_list), dimnames = list(snp_list, snp_list))
    index_A <- match(ld_df$SNP_A, snp_list)
    index_B <- match(ld_df$SNP_B, snp_list)

    ld_matrix[cbind(index_A, index_B)] <- ld_df$R
    ld_matrix[cbind(index_B, index_A)] <- ld_df$R
    diag(ld_matrix) <- 1
    eQTLLDs[[pop]] <- ld_matrix
}

# create matrix of eQTL marginal z scores 
eQTLzscores <- matrix(NA, nrow = length(snp_list), ncol = length(eQTL_dfs))
for (i in 1:length(eQTL_dfs)){
    eQTLzscores[,i] = eQTL_dfs[[i]]$BETA / eQTL_dfs[[i]]$SE
}
colnames(eQTLzscores) <- pop_names

# read in gwas file 
gwas_df <- fread(opt$gwas_file, header = TRUE, dec = ".")
GWASzscores <- gwas_df$beta / gwas_df$se
names(GWASzscores) <- snp_list


# metro 
METRORes2 <- METRO2SumStat(eQTLzscores,
                           eQTLLDs,
                           GWASzscores,
                           eQTLLDs[[1]],
                           eQTL_samplesizes,
                           opt$ngwas_of_people,
                           hthre = 0.001,
                           verbose = T)

if (anyNA(eQTLzscores)) {
  cat("There are NA values in eQTLzscores\n")
} else {
  cat("No NA values in eQTLzscores\n")
}

if (anyNA(eQTLLDs)) {
  cat("There are NA values in eQTLLDs\n")
} else {
  cat("No NA values in eQTLLDs\n")
}

if (anyNA(GWASzscores)) {
  cat("There are NA values in GWASzscores\n")
} else {
  cat("No NA values in GWASzscores\n")
}

if (anyNA(eQTLLDs[[1]])) {
  cat("There are NA values in eQTLLDs[[1]]\n")
} else {
  cat("No NA values in eQTLLDs[[1]]\n")
}


# METRORes2 <- METRO2SumStat(eQTLzscores,
#                            eQTLLDMatrix,
#                            GWASzscores, 
#                            GWASLDMatrix,
#                            ns,
#                            n, verbose = T)
metro_statistic = METRORes2$alpha # gene effect on GWAS outcome
metro_p = METRORes2$pvalueLRT
metro_lrtstat = METRORes2$LRTStat
metro_betas = METRORes2$beta
metro_df = METRORes2$df
str(METRORes2)
write.table(data.frame(a = c(metro_statistic), p = c(metro_p), lrt = (metro_lrtstat), df=(metro_df)), file = paste0(opt$output_folder, "_stats.txt"), quote = F, row.names = F, col.names = T, sep = '\t')
write.table(data.frame(betas = metro_betas), file = paste0(opt$output_folder, "_betas.txt"), quote = F, row.names = F, col.names = T, sep = '\t')