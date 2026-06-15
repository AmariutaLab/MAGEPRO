expected_header <- c("Gene", "SNP", "A1", "A2", "BETA", "SE", "P")
suppressMessages(library('tools'))
suppressMessages(library('data.table'))
suppressMessages(library('parallel'))


cohort_fine_mapping <- function(cohort_map, sumstats_dir, tmp, ldref_dir, out, gene, PATH_plink, verbose) {
	# PURPOSE: run fine_mapping on each on a gene in specified and available cohorts
	# PREREQ: cohort_map – map containing information about sample_size and 
	#         ldref file for the cohort
	# Every other parameter description can be found in MAGEPRO.R option_list.
	# RETURN: hashmap which maps dataset name to TRUE/FALSE, indicating if 
	# susie was ran successfully for that dataset

	results <- lapply(names(cohort_map), function(cohort) {
		process_cohort(cohort, cohort_map, sumstats_dir, tmp, ldref_dir, out, gene, PATH_plink, verbose)
	})

	susie_result_status <- list()
	for (i in seq_along(results)) {
		print( paste( names(cohort_map)[i] , results[[i]]$status) )
		cohort <- names(cohort_map)[i]
		susie_result_status[[cohort]] <- results[[i]]$status
		if (results[[i]]$status) {
			assign(paste0("file.", cohort), results[[i]]$path, envir = .GlobalEnv)
		}
	}
	return(susie_result_status)
}


process_cohort <- function(cohort, cohort_map, sumstats_dir, tmp, ldref_dir, out, gene, PATH_plink, verbose) {
	print(paste0("current cohort: ", cohort))
	cohort_data <- cohort_map[[cohort]]
	cohort_path <- file.path(sumstats_dir, cohort)
	cohort_ld_directory <- file.path(tmp, paste0(cohort, "ld"))
	cohort_ldref_path <- file.path(ldref_dir, cohort_data$ldref)
	out_cohort_path <- file.path(out, cohort)

	# Create directories for output and ld_matrix_path
	dir.create(cohort_ld_directory, recursive = TRUE, showWarnings = FALSE)
	dir.create(out_cohort_path, recursive = TRUE, showWarnings = FALSE)

	gene_txt <- paste0(gene, '.txt')
	check <- file.path(cohort_path, gene_txt)
	if (!file.exists(check)) {
		if (verbose == 2) {
			cat("skipping sumstat", cohort, "for this gene since the file was not found in ", check, "\n")
		}
		return(list(status = FALSE, path = NULL))
	}

	path_to_fine_mapping_output <- gene_fine_mapping(gene_txt, cohort, cohort_data, cohort_path, cohort_ld_directory, cohort_ldref_path, PATH_plink, out_cohort_path, verbose)
	if (path_to_fine_mapping_output == "Error") {
		return(list(status = FALSE, path = NULL))
	}

	cat("successfully fine mapped ", gene, " for ", cohort, "\n")
	return(list(status = TRUE, path = path_to_fine_mapping_output))
}




gene_fine_mapping <- function(gene_txt, cohort, cohort_data, cohort_path, cohort_ld_directory, cohort_ldref_path, plink, out, verbose) {
	# PURPOSE: create correlation matrix for each gene for given cohort, then perform fine-mapping with susie_rss in get_pips.R
	# PREREQs: 
	# 	cohort – current cohort string;  
	#	cohort_data – vector containing information about sample size and ldref
	# 	cohort_path – file path containing information about curent cohort
	# 	cohort_ld_directory – path to temporary folder storing correlation matrix information
	#   cohort_ldref_path – path to ld reference file
	# Every other parameter description can be found in MAGEPRO.R option_list.
	# RETURN: file path to the gene output
	file_path <- file.path(cohort_path, gene_txt)
	if ( ! identical( colnames(fread(file_path, nrows = 0))[1:7] , expected_header) ) {
		cat( "WARNING: rewriting summary statistics file with correct headers. Dropping extra columns. Make sure statistics are in the correct order specified in README \n" , sep='', file=stderr() )
		temp_df <- fread(file_path, select = ( c(1:length(expected_header)) ) )
		colnames(temp_df) <- expected_header
		write.table(temp_df, file = file_path, sep = ' ', quote = F, col.names = T, row.names = F)
	}

	# --- REVIEWS3 negative control, replace betas with randomly sampled sumstats 
	#df_sumstats <- fread(file_path, select = ( c(1:7)) )
	#df_sumstats[,5] <- rnorm(nrow(df_sumstats), mean = 0, sd = 1) # BETA
	#df_sumstats[,6] <- 1 # SE
	#print(df_sumstats)
	#gene_txt <- paste0('random_', gene_txt)
	#cohort_path <- out
	#file_path = file.path(cohort_path, gene_txt)
	#print(file_path)
	#write.table(df_sumstats, file = file_path, sep = ' ', quote = F, col.names = T, row.names = F)
	# --- COMMENT OUT AFTER

	gene <- file_path_sans_ext(gene_txt)
	to_return <- file.path(out, gene_txt)
	
	is_verbose <- ifelse(verbose >= 2, "", " > /dev/null ")
	get_ld_cmd <- paste0(
		plink,
		" --bfile ", cohort_ldref_path,
		" --r inter-chr --ld-window-r2 0 --keep-allele-order",
		" --extract ", file_path,
		" --out ", file.path(cohort_ld_directory, gene),
		is_verbose
	)

	exit_status_plink <- system(get_ld_cmd, wait = TRUE, intern=FALSE)

	if (exit_status_plink != 0) {
		to_return <- "Error"
		cat("Error in computing pairwise SNP correlation in plink for ", gene, " in ", cohort, ". Moving to next cohort...\n", file=stderr())
		return(to_return)
	}

	rcommand <- paste0("Rscript FUNCTIONS/get_pips.R -g ", cohort_path, "/",
	" -c ", gene_txt, 
	" -l ", file.path(cohort_ld_directory, paste0(gene, ".ld")),
	" -n ", cohort_data$sample_size,
	" -o ", out,
	" -b ", paste0(cohort_ldref_path, ".bim"),
	' -v ', cohort_data$in_sample
	)

	exit_status_susie <- system(rcommand, wait = TRUE, intern=FALSE)
	if (exit_status_susie != 0) {
		to_return <- "Error"
		cat("Error in running SuSiE in R for ", gene, " in ", cohort, ". Moving to next cohort...\n", file=stderr())
		return(to_return)
	}


	return(to_return)
}
