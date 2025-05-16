suppressMessages(library("optparse"))
suppressMessages(library("susieR"))
suppressMessages(library("data.table"))

suppressMessages({library("Rfast")})
suppressMessages(library("dplyr"))
suppressMessages(library("ggplot2"))
suppressMessages(library('reticulate'))
np <- import("numpy")
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
    return(opt)
}

opt <- arg_parser()
print(opt)

# read files
df <- fread(opt$c, header=TRUE, dec=".")       # read in current gene
colnames(df) = c("SNP", "A1", "A2", "BETA", "SE", "P")
snp_list <- df$SNP

rownames(df) <- NULL

############################################################
# build matrix
npz <- np$load(opt$l)
LD_array <- npz$f[["LD"]]

LD_r <- as.matrix(LD_array)
dimnames(LD_r) <- list(snp_list, snp_list)

gene <- sub("\\.txt$", "", opt$c)
cat("Running IBSS-ss algorithm through susie_rss...\n")
tryCatch({
    withCallingHandlers({
        res <- susie_rss(bhat = df$BETA, shat = df$SE, R = LD_r, n = opt$n, max_iter = 100)
    }, warning = function(w) {
        if (grepl("IBSS algorithm did not converge", w$message)) {
            cat("WARNING in run_susie.R:66: IBSS algorithm did not converge in 100 iterations. ", "\n")
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
