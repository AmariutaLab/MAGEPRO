suppressMessages(library("optparse"))
suppressMessages(library("susieR"))
suppressMessages(library("data.table"))

suppressMessages({library("Rfast")})
suppressMessages(library("dplyr"))
suppressMessages(library("ggplot2"))
cat("Running run_susie.r\n")
arg_parser <- function() {
    option_list <- list(
        make_option(c("-c", "--current_gene"), type = "character",
                    help = "Identifier of the current gene", metavar = "CURRENTGENE"),
        make_option(c("-l", "--ld_matrix"), type = "character",
                    help = "Path to the LD matrix file", metavar = "LDMATRIX"),
        make_option(c("-n", "--n_of_people"), type = "integer",
                    help = "Number of people/sample size", metavar="NSUSIE"),
        make_option(c("-o", "--output_folder"), type="character", help="output folder name", metavar="OUTPUTFOLDER"),
        make_option(c("-b", "--bim_file"), type="character", help="Path to file containing ld reference bim file", metavar="BIMFILE")
    )

    opt_parser <- OptionParser(option_list = option_list)
    opt <- parse_args(opt_parser)

    if (is.null(opt$current_gene)) {
        stop("Current gene csv (-c, --current_gene) is required", call. = FALSE)
    }
    if (is.null(opt$ld_matrix)) {
        stop("LD Matrix (-l, --ld_matrix) is required", call. = FALSE)
    }
    if (is.null(opt$n_of_people)) {
        stop("Number of people/sample size (-n, --n_of_people) is required", call. = FALSE)
    }
    if (is.null(opt$output_folder)) {
        stop("Output folder (-o, --output_folder) is required", call. = FALSE)
    }
    if (is.null(opt$bim_file)) {
        stop("bim file path (-b, --bim_file) is required", call. = FALSE)
    }
    return(opt)
}

opt <- arg_parser()

# read files
df <- fread(opt$c, header=TRUE, dec=".")       # read in current gene
colnames(df) = c("SNP", "A1", "A2", "BETA", "SE", "P")
ld_stats <- fread(opt$l, header=TRUE, dec=".")    # read in ld_stats

# filter out in gene that are not present in LD
df <- df[df$SNP %in% (union(ld_stats$SNP_A, ld_stats$SNP_B))]
snp_list <- df$SNP

############################################################
# checking for allele flips or ambiguous allele pairs

bimdf <- fread(opt$b, header=FALSE)
colnames(bimdf) = c("CHR", "SNP", "CM", "POS", "A1", "A2")
bimdf <- bimdf[bimdf[[2]] %in% snp_list]
bimdf <- bimdf %>% arrange(match(bimdf[[2]], snp_list))

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
  snp[["keep"]][ a1 != "A" & a1 != "T" & a1 != "G" & a1 != "C" ] = FALSE # non ATGC
  snp[["keep"]][ a2 != "A" & a2 != "T" & a2 != "G" & a2 != "C" ] = FALSE # non ATGC
  snp[["keep"]][ (a1==ref1 & a2!=ref2) | (a1==ref2 & a2!=ref1) | (a1==flip1 & a2!=flip2) | (a1==flip2 & a2!=flip1) ] = FALSE # 3 alleles
  snp[["flip"]] = (a1 == ref2 & a2 == ref1) | (a1 == flip2 & a2 == flip1) # flips
  
  return(snp)
}

qc_results <- allele.qc(df$A1, df$A2, bimdf[[5]], bimdf[[6]])

df[qc_results$flip, "BETA"] <- df[qc_results$flip, "BETA"] * -1

temp <- df[qc_results$flip, "A2"]
df[qc_results$flip, "A2"] <- df[qc_results$flip, "A1"]
df[qc_results$flip, "A1"] <- temp

df <- df[qc_results$keep, ]

rownames(df) <- NULL
snp_list <- df$SNP


ld_stats <- ld_stats[ld_stats$SNP_A %in% snp_list] # removed snps from corr matrix that are removed by QC
ld_stats <- ld_stats[ld_stats$SNP_B %in% snp_list]
############################################################


############################################################
# build matrix
ld_matrix <- matrix(NA, nrow = length(snp_list), ncol = length(snp_list), dimnames = list(snp_list, snp_list))
index_A <- match(ld_stats$SNP_A, snp_list)
index_B <- match(ld_stats$SNP_B, snp_list)

ld_matrix[cbind(index_A, index_B)] <- ld_stats$R
ld_matrix[cbind(index_B, index_A)] <- ld_stats$R
diag(ld_matrix) <- 1
############################################################

gene <- sub("\\.txt$", "", opt$c)
cat("Running IBSS-ss algorithm through susie_rss...\n")
tryCatch({
    withCallingHandlers({
        res <- susie_rss(bhat = df$BETA, shat = df$SE, R = ld_matrix, n = opt$n, max_iter = 100)
    }, warning = function(w) {
        if (grepl("IBSS algorithm did not converge", w$message)) {
            cat("WARNING in run_susie.R:131: IBSS algorithm did not converge in 100 iterations. ", "\n")
            quit(status=1)
        } else {
            cat(w$message, "\n")
        }
    })
}, error = function(e) {
    cat("In susie_rss:\n", e$message, "\n")
    quit(status=1)
})

pips <- res$pip

# create credible set column
cs <- rep(-1, length(df[[5]]))
for(cname in names(res$sets$cs)) {
  number <- as.integer(gsub("L", "", cname))
  indices <- res$sets$cs[[cname]]
  cs[indices] <- number
}

# writing the table
df$pip <- pips
betas <- coef(res)[-1]  # coef.susie – extract regression
df$posterior <- betas
df$posterior_sd <- susie_get_posterior_sd(res)
df$cs <- cs

df <- df[order(df$cs, decreasing = TRUE), ]
cat("Saving susie output to", opt$o, "\n")
colnames(df) = c("SNP", "A1", "A2", "BETA", "SE", "P", "PIP", "POSTERIOR", "POSTERIOR_SD", "CS")
fwrite(x=df, file=opt$o, sep=" ", col.names=TRUE, row.names=FALSE, quote=FALSE)