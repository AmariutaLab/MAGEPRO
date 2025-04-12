# This script is used to simulate TWAS with MAGEPRO
# A lot of code for this simulation was taken from:
#   https://github.com/mancusolab/twas_sim?tab=readme-ov-file



### NOTES TO STEPHEN: 
# I changed build_susie_sumstats to save the marginal eQTL summary statistics and LD correlation matrices, which is necessary for METRO
# I wrote the scripts to be able to run METRO. I tested everything up to actually running the METRO function because I could not get the environment to work. 
    # The inputs are formatted as METRO expects it so everything should run in theory
# TO-DO: 
    # could you double check line 707 to make sure the LD matrix used is the target population LD matrix? It seems to me like LD there is whatever comes last in the populations list? 
    # line 732, when we get the betas from METRO, I want to calculate the R-squared between true African gene expression and the ones predicted by METRO betas. However, this isn't really a cross validation because these betas are from a linear combination of marginal betas computed in each cohort... any thoughts?

from pandas_plink import read_plink
import numpy as np
import scipy.linalg as linalg
from scipy.stats import invgamma
from scipy.stats import norm
import pandas as pd
import subprocess
import os
import argparse
import gzip
import sys
import statsmodels.api as sm
from numpy.linalg import multi_dot as mdot
from scipy.special import ndtr
from magepro_simulations_functions import (magepro_cv, top1_cv,
                                           fit_sparse_regularized_lm,
                                           load_process_sumstats,
                                           correlated_effects_cholesky,
                                           create_betas)

NUMTARGET_LIST = [300]
WINDOW = 500000

NCAUSAL_ARR = [10] # number of causal SNPs for eQTL file

# EQTL_H2_ARR = [0.01, 0.05, 0.1] # True heritability value
# PHEN_VAR_GENE_COMPONENT = [0, 0.0001, 0.001, 0.01]
# CORRELATIONS_GE = [0.2, 0.8]
# GWAS_SAMPLE_SIZE = [20000, 100000, 600000]

EQTL_H2_ARR = [0.01, 0.05, 0.1] # True heritability value
PHEN_VAR_GENE_COMPONENT = [0, 0.0001, 0.001, 0.01]
CORRELATIONS_GE = [0.2, 0.8, 1.0]

GWAS_SAMPLE_SIZE = [20000, 600000]

DEFAULT_NUM_OF_PEOPLE_EQTL = 500
SEP = "\t"
OUTPUT_COLUMNS = f"""GENE{SEP}EQTL_H2{SEP}H2GE{SEP}CORRELATION{SEP}NUM_GWAS{SEP}HSQ{SEP}HSSQ_PV{SEP}MAGEPRO_R2{SEP}MAGEPRO_Z{SEP}MAGEPRO_PVAL{SEP}LASSO_R2{SEP}LASSO_Z{SEP}LASSO_PVAL{SEP}METRO_ALPHA{SEP}METRO_PVAL{SEP}METRO_LRT{SEP}METRO_DF{SEP}MAGEPRO_LASSO_CORR{SEP}MAGEPRO_METRO_CORR{SEP}realized_h\n"""

def get_ld(prefix):
    # return cholesky L
    bim, fam, G = read_plink(prefix, verbose=False)
    G = G.T # Transpose to have SNPs as columns

    # estimate LD for population from PLINK data
    n_samples, n_snps = G.shape
    mafs = (np.mean(G, axis=0) / 2).compute()

    # Center and standardize G
    G -= mafs * 2
    std_devs = np.std(G, axis=0)
    std_devs[std_devs == 0] = 1
    G /= std_devs

    # Regularize the LD matrix to make it Positive Semi Definite
    LD = np.dot(G.T, G) / n_samples
    regularization = 0.1
    LD += np.eye(n_snps) * regularization

    # re-adjust to get proper correlation matrix
    LD = LD / (1 + regularization)

    # compute cholesky decomp for faster sampling/simulation
    L = linalg.cholesky(LD, lower=True)

    return (bim, L)


def sim_geno(L, n): 
    """
    Sample genotypes from an MVN approximation.

    :param L: numpy.ndarray lower cholesky factor of the p x p LD matrix for the population
    :param n: int the number of genotypes to sample

    :return: numpy.ndarray n x p centered/scaled genotype matrix
    """
    p, p = L.shape
    Z = L.dot(np.random.normal(size=(n, p)).T).T #pxn
    Z -= np.mean(Z, axis=0)
    Z /= np.std(Z, axis=0) 
    return Z


def sim_beta(L, ncausal, index_output, eqtl_h2, rescale=True):
    """
    Sample qtl effects under a specified architecture.

    :param L: numpy.ndarray lower cholesky factor of the p x p LD matrix for the population
    :param ncausal: int number of causal SNPs
    :index_output: int index of causal SNP
    :param eqtl_h2: float the heritability of gene expression
    :param rescale: bool whether to rescale effects such that var(b V b) = h2 (default=True)

    :return: numpy.ndarray of causal effects
    """

    n_snps = L.shape[0]
    if eqtl_h2 != 0 and ncausal > 0:
        # choose how many eQTLs
        n_qtls = ncausal
        c_qtls = index_output
        b_qtls = np.zeros(int(n_snps))

        # sample effects from normal prior
        b_qtls[c_qtls] = np.random.normal(
            loc=0, scale=np.sqrt(eqtl_h2 / n_qtls), size=n_qtls
        )
        if rescale:
            s2g = compute_s2g(L, b_qtls)
            b_qtls *= np.sqrt(eqtl_h2 / s2g)
    else:
        b_qtls = np.zeros(int(n_snps))
    return b_qtls


def sim_trait(g, h2g):
    """
    Simulate a complex trait as a function of latent genetic value and env noise.

    :param g: numpy.ndarray of latent genetic values
    :param h2g: float the heritability of the trait in the population

    :return: (numpy.ndarray, float) simulated phenotype, sd of Y
    """
    n = len(g)
    if h2g > 0:
        s2g = np.var(g, ddof=1)
        s2e = s2g * ( (1.0 / h2g ) - 1 )
        e = np.random.normal(0, np.sqrt(s2e), n)
        y = g + e #appears to add noise to the genetic values, adding the same about of noise if the seed is the same 
    else:
        e = np.random.normal(0, 1, n)
        y = e
    # standardize
    y -= np.mean(y)
    y_std = np.std(y)
    y /= y_std

    y_std = y_std.item()
    return y, y_std


def compute_s2g(L, beta):
    """
    Compute genetic variance given betas and LD cholesky factor

    s2g := beta' V beta = beta' L L' beta

    :param L: numpy.ndarray lower cholesky factor of the p x p LD matrix for the population
    :param beta: numpy.ndarray genotype effects

    :return: float s2g
    """
    Ltb = np.dot(L.T, beta)
    s2g = np.dot(Ltb.T, Ltb)

    return s2g


def pick_rand_gene(plink_path, geno_prefix_path_list, random_genes_folder):
    '''
        :plink_path: path to plink binary
        :geno_prefix_path_list: prefix path of the genotype file (no bed, bim, fam)
        :random_genes_folder: folder where random genes are stored
        :return: random gene output_list, chromosome, random causal SNP
    '''
    while True:
        # choose random chromosome from 1 to 22
        chr = np.random.randint(1, 23)
        print(f"Chose chromosome {chr}")
        file = f"{geno_prefix_path_list[0]}{chr}.bim"
        # Count the number of lines in the bim file
        result = subprocess.run(['wc', '-l', file], stdout=subprocess.PIPE, text=True)
        num_variants = int(result.stdout.split()[0])  # Extract the number of lines

        # choose random line and print the content of it
        line = np.random.randint(1, num_variants + 1)
        result = subprocess.run(['sed', '-n', f"{line}p", file],
                                stdout=subprocess.PIPE, text=True)
        random_variant = result.stdout.strip()

        # in the chosen line, extract 4th column, which is position
        random_position = int(random_variant.strip().split()[3])
        start_pos = max(0, random_position - WINDOW)
        end_pos = random_position + WINDOW

        print(f"Start Position: {start_pos}")
        print(f"End Position: {end_pos}")

        # we run plink on all input cohorts
        output_list = []
        for geno_prefix_path in geno_prefix_path_list:
            geno_prefix = os.path.basename(geno_prefix_path)
            output_list.append(f'{random_genes_folder}/{geno_prefix}{chr}_{random_position}')
            # run plink to create a bed
            result = subprocess.run([plink_path, 
                                    '--bfile', f'{geno_prefix_path}{chr}', 
                                    '--chr', str(chr),
                                    '--from-bp', str(start_pos),
                                    '--to-bp', str(end_pos),
                                    '--make-bed',
                                    '--out', output_list[-1]],
                                    stdout=subprocess.PIPE, text=True)
        
        # here we check if no SNPs were found in at least 1 gene
        flag = True
        for output in output_list:
            if not os.path.exists(f'{output}.bim'):
                flag = False
                print(f"No SNPs found in the region for {output}.bim. Rerunning...")
                subprocess.run(f'rm -rf {random_genes_folder}/*', shell=True)
                break
        if flag == False:
            continue
        
        # here we check if enough SNPs were found in all cohorts
        vars_random_gene_list = []
        for output in output_list:
            result = subprocess.run(['wc', '-l', f'{output}.bim'],
                                    stdout=subprocess.PIPE, text=True)
            vars_random_gene_list.append(int(result.stdout.split()[0]))
        min_snps = min(vars_random_gene_list)
        if min_snps < 50:
                print('Not enough snps in chosen gene. Rerunning...')
                subprocess.run(f'rm -rf {random_genes_folder}/*', shell=True)
                continue

        print('Successfully picked a random gene')
        return output_list, str(chr), min_snps


def get_gcta_hsq(geno, gexpr, output_path, gcta):
    '''
        Calculates heritability estimates using GCTA
        :geno: numpy.ndarray n x p centered/scaled genotype matrix
        :gexpr: (numpy.ndarray, float) simulated phenotype, sd of Y
        :output_path: path to output file
        :gcta: path to gcta binary file

        :returns: hsq, hsq_se, hsq_p - all floats
    '''
    nindv, nsnp = geno.shape
    grm = np.dot(geno, geno.T) / nsnp
    write_pheno(gexpr, output_path)
    write_grm(grm, nsnp, output_path)
    run_gcta(gcta, output_path)
    clean_up(output_path)
    hsq_f = f'{output_path}.hsq'
    if os.path.exists(hsq_f):
        hsq_out = pd.read_table(hsq_f)
        hsq = hsq_out['Variance'][0]
        hsq_se = hsq_out['SE'][0]
        hsq_p = hsq_out['Variance'][8]
        command = f'rm {hsq_f}'
        subprocess.run(command.split())
        return hsq, hsq_se, hsq_p
    else:
        return 0, 10000, 1


def build_susie_sumstats(bim, geno, gexpr, sumstats_path, ld_path, gene,
                        gene_pref, plink_path, cNUM_PEOPLE):
    '''
        Builds SuSie Summary Statistics file

        :bim: bim file from plink random gene input
        :geno: numpy.ndarray n x p centered/scaled genotype matrix
        :gexpr: (numpy.ndarray, float) simulated phenotype, sd of Y
        :sumstats_path: path to where to save sumstats
        :ld_path: path to where to save ld files
        :gene: gene name from @pick_rand_gene
        :gene_pref: gene prefix with plink input

        :returns: filepath to fine mapped summary statistics, filepath to marginal summary statistics, filepath to snp-correlation 

    '''
    betas = []
    pvals = []
    std_errs = []
    for i in range(geno.shape[1]):
        x = sm.add_constant(geno[:, i].reshape(-1,1))
        mod = sm.OLS(gexpr, x)
        results = mod.fit()
        betas.append(results.params[1])
        pvals.append(results.pvalues[1])
        std_errs.append(results.bse[1])
    sumstats = pd.DataFrame({
        'Gene': os.path.basename(gene),
        'SNP': bim['snp'],
        'A1': bim['a0'],
        'A2': bim['a1'],
        'BETA': betas,
        'SE': std_errs,
        'P': pvals
    })
    filename_marginal = os.path.join(sumstats_path, gene + "marginal.txt")
    sumstats.to_csv(filename_marginal, sep="\t", index=False, header=True)
    ld_path_gene = os.path.join(ld_path, gene)

    subprocess.run([plink_path, '--bfile', gene_pref, '--r', 'inter-chr',
                    '--ld-window-r2', '0', '--keep-allele-order', '--extract',
                    filename_marginal, '--out', ld_path_gene])

    filename_susie = os.path.join(sumstats_path, gene + ".txt")
    subprocess.run(['Rscript', 'run_susie_twas.R',
                    '-c', filename_marginal,
                    '-l', ld_path_gene + ".ld",
                    '-n', str(cNUM_PEOPLE),
                    '-b', gene_pref + ".bim",
                    '-o', filename_susie])
    return filename_susie, filename_marginal, ld_path_gene + ".ld"


def clean_up(out):
    command = f'rm {out}.grm.gz'
    subprocess.run(command.split())
    command = f'rm {out}.grm.id'
    subprocess.run(command.split())
    command = f'rm {out}.phen'
    subprocess.run(command.split())


def write_pheno(phen, out):
    out_file = open(out+'.phen', 'w')
    nindv = phen.shape[0]
    for i in range(nindv):
        out_file.write('{}\t{}\t{}\n'.format(i, i, phen[i]))
    out_file.close()


def write_grm(grm, nsnp, out):
    out_id_file = open(out+'.grm.id', 'w')
    out_file = gzip.open(out+'.grm.gz', 'w')
    nindv = grm.shape[0]
    for i in range(nindv): 
        for j in range(0, i+1):
            out_file.write('{}\t{}\t{}\t{}\n'.format(i+1, j+1, nsnp,
                grm[i,j]).encode('utf-8'))
        out_id_file.write('{}\t{}\n'.format(i, i))
    out_file.close()
    out_id_file.close()


def run_gcta(gcta, out):
    command = '{} --grm-gz {} --pheno {} --reml --reml-no-constrain --out {}'\
        .format(gcta, out, out+'.phen', out)
    subprocess.run(command.split())
    command = 'rm {}.log'.format(out)
    subprocess.run(command.split())


def sim_gwas(L, ngwas, beta, h2ge):
    """
    Simulate a GWAS using `ngwas` individuals such that genetics explain `h2ge` of phenotype.
        This function approximates genotypes under an LD structure using an MVN model. Generating
        genotype for `ngwas` individuals takes `O(np^2)` time.

    :param L: numpy.ndarray lower cholesky factor of the p x p LD matrix for the population
    :param ngwas: int the number of GWAS genotypes to sample
    :param b_qtls: numpy.ndarray latent eQTL effects for the causal gene
    :param h2ge: float the amount of phenotypic variance explained by genetic component of gene expression

    :return: (pandas.DataFrame) estimated GWAS beta and standard error
    """
    Z_gwas = sim_geno(L, ngwas)

    # h2ge should only reflect that due to genetics
    g = np.dot(Z_gwas, beta)
    y = sim_trait(g, h2ge)[0]

    gwas = regress(Z_gwas, y)
    return gwas


def sim_gwasfast(L, ngwas, beta, h2ge):
    """
    Simulate a GWAS using `ngwas` individuals such that genetics explain `h2ge` of phenotype.
        This function differs from `sim_gwas` in that it samples GWAS summary statistics directly
        Using an MVN approximation, rather than generating genotype, phenotype, and performing
        a marginal regression for each simulated variant. Runtime should be `O(p^2)` where
        `p` is number of variants.

    :param L: numpy.ndarray lower cholesky factor of the p x p LD matrix for the population
    :param ngwas: int the number of GWAS genotypes to sample
    :param b_qtls: numpy.ndarray latent eQTL effects for the causal gene
    :param h2ge: float the amount of phenotypic variance explained by genetic component of gene expression

    :return: (pandas.DataFrame) estimated GWAS beta and standard error

    """
    n_snps = L.shape[0]

    s2g = compute_s2g(L, beta)

    if h2ge > 0:
        s2e = s2g * ((1.0 / h2ge) - 1)
    else:
        s2e = 1.0  # var[y]; could be diff from 1, but here we assume 1

    dof = ngwas - 1
    tau2 = s2e / ngwas
    se_gwas = np.sqrt(invgamma.rvs(a=0.5 * dof, scale=0.5 * dof * tau2, size=n_snps))
    DL = se_gwas[:, np.newaxis] * L

    beta_adj = mdot([DL, L.T, (beta / se_gwas)])  # D @ L @ Lt @ inv(D) @ beta

    # b_gwas ~ N(D @ L @ L.t inv(D) @ beta, D @ L @ Lt @ D), but fast
    b_gwas = beta_adj + np.dot(DL, np.random.normal(size=(n_snps,)))
    Z = b_gwas / se_gwas
    pvals = 2 * norm.sf(abs(Z))
    gwas = pd.DataFrame({"beta": b_gwas, "se": se_gwas, "pval": pvals})

    return gwas



def regress(Z, pheno):
    """
    Perform a marginal linear regression for each snp on the phenotype.

    :param Z: numpy.ndarray n x p genotype matrix to regress over
    :param pheno: numpy.ndarray phenotype vector

    :return: pandas.DataFrame containing estimated beta and standard error
    """
    betas = []
    ses = []
    pvals = []
    
    for snp in Z.T:
        X = sm.add_constant(snp)
        model = sm.OLS(pheno, X)
        results = model.fit()
        inter, beta = results.params
        se_inter, se_beta = results.bse
        inter_pvalue, pvalues = results.pvalues

        betas.append(beta)
        ses.append(se_beta)
        pvals.append(pvalues)

    gwas = pd.DataFrame({"beta": betas, "se": ses, "pval": pvals})
    return gwas


def sim_eqtl(Z_eqtl, gexpr, tmp_path): 
    """
    Simulate an eQTL study.
    :param Z_eqtl: simulated genotypes
    :param tmp_path: name of temporary path
    :return:  (numpy.ndarray x 6) Arrays of best lasso penalty, eqtl effect sizes, heritability, cross validation accuracies, simulated ge
    """
    penalty, coef, r2 = fit_sparse_regularized_lm(Z_eqtl, gexpr, 'lasso')
    if r2>0:
        h2g, hsq_se, hsq_p = get_gcta_hsq(Z_eqtl, gexpr, tmp_path, "/expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust")
    else:
        h2g=0
        hsq_se=10
        hsq_p=1
    return (penalty, coef, h2g, hsq_p, r2)


def run_magepro(target_gene, 
                Z_eqtl,
                b_qtls,
                gexpr,
                tmp_path,
                pop,
                num_people,
                sumstats_files_external,
                pops_external):
    """
        target_gene - plink formated file for gene from target cohort
        Z_eqtl - simulated genotypes for target gene
        b_qtls - simulated latent effect sizes
        gexpr -  simulated gene expression. Used to fit_sparse_regularized_lm
        tmp_path - path to where temporary file will be stored
        pop - target population
        num_people - number of people in target population
        sumstats_files_external - paths to summary statistics files
        pops_external - names of external populations

        Returns
        -------
        heritability value, heritability p-value, magepro betas, magepro_pval, magepro r2, lasso betas, lasso_pval, lasso r2
    """
    bim, fam, G_all = read_plink(target_gene)
    bim = np.array(bim)

    b_qtls = np.reshape(b_qtls.T, (bim.shape[0], 1))
    best_penalty, coef, h2g, hsq_p, r2all = sim_eqtl(Z_eqtl, gexpr, tmp_path)

    if np.all(coef == 0):
        print("using top1 backup")
        r2all, r2_top1, coef = top1_cv(num_people, Z_eqtl, gexpr)

    # num_causal_susie = 0
    sumstats_weights = {}
    for i in range(0, len(sumstats_files_external)):
        varname = pops_external[i] + 'weights'
        weights, pips = load_process_sumstats(sumstats_files_external[i], bim, sep=" ")
        # if SuSiE didn't converge, we can't use that as input for RIDGE
        if weights[0] == float('-inf'):
            print("Warning: Fine-mapping didn't converge for",
                  sumstats_files_external[i])
            continue
        sumstats_weights[varname] = weights
        # num_causal_susie += len([index for index in CAUSAL if pips[index] >= 0.95])
    magepro_r2, magepro_coef = magepro_cv(num_people, Z_eqtl, gexpr, sumstats_weights, best_penalty, m="lasso")

    # #coef = afr lasso model gene model 
    # lasso_causal_nonzero = len([index for index in CAUSAL if coef[index] != 0])
    # #magepro_coef = magepro gene model 
    # magepro_causal_nonzero = len([index for index in CAUSAL if magepro_coef[index] != 0])

    return (h2g, hsq_p, coef, r2all, magepro_coef, magepro_r2)

def run_metro(eqtl_paths, pop_names, eqtl_corr_paths,
              eqtl_samplesizes, gwas_file, ngwas, out):
    """
    Compute METRO test statistics. 

    :param eqtl_paths: Comma separated string of marginal eQTL summmary statistics paths
    :param eqtl_corr_paths: Comma separated string of snp-snp correlation paths for each eQTL cohort
    :param eqtl_samplesizes: Comma separated string of sample sizes for each eQTL cohort
    :param gwas_file: Path to gwas file 
    :param ngwas: gwas sample size as string
    :param out: output file path

    :return: (float, float, np.array) the METRO test statistics, p-value, gene model betas
    """

    subprocess.run(['Rscript', 'run_metro.R',
                    '-e', eqtl_paths,
                    '-l', eqtl_corr_paths,
                    '-p', pop_names,
                    '-s', eqtl_samplesizes,
                    '-g', gwas_file,
                    '-n', ngwas,
                    '-o', out])

    output = ['NaN', 'NaN', 'NaN', 'NaN', 'NaN']

    if os.path.exists(out + "_stats.txt"):
        stats = pd.read_csv(out + "_stats.txt", sep = '\t')
        output[0], output[1], output[2], output[3] = stats['a'].values[0], stats['p'].values[0], stats['lrt'].values[0], stats['df'].values[0]
    if os.path.exists(out + "_betas.txt"):
        betas = pd.read_csv(out + "_betas.txt", sep = '\t')
        output[4] = betas['betas'].values

    return output

def compute_twas(gwas, coef, LD):
    """
    Compute the TWAS test statistics.

    :param gwas: pandas.DataFrame containing estimated GWAS beta and standard error
    :param coef: numpy.ndarray LASSO eQTL coefficients
    :param LD:  numpy.ndarray p x p LD matrix

    :return: (float, float) the TWAS test statistics and p-value
    """
    # compute Z scores
    Z = gwas.beta.values / gwas.se.values

    # score and variance
    score = np.dot(coef, Z)
    within_var = mdot([coef, LD, coef])

    if within_var > 0:
        z_twas = score / np.sqrt(within_var)
        p_twas = 2 * (1 - ndtr(np.abs(z_twas)))
    else:
        # on underpowered/low-h2g genes LASSO can set all weights to 0 and effectively break the variance estimate
        z_twas = 0
        p_twas = 1

    return z_twas, p_twas


def arg_parser():
    parser = argparse.ArgumentParser(
        prog='Build simulation for TWAS.',
        description='''This brief command line creates a simulation for TWAS for
        MAGEPRO. At the top of the file you can specify the constants for which
        to simulate.'''
    )
    parser.add_argument("--plink_path", required=True,
                        type=str, help="Path to plink binary file")
    parser.add_argument('--geno_prefix_list', required=True,
                        type=str,
                        help="Path to geno file list, coma separated")
    parser.add_argument('--pops_list', required=True, type=str,
                        help="Coma Separated list of populations. Target population first")
    parser.add_argument('--tmpdir', required=True,
                        type=str, help="Path to temporary directory")
    parser.add_argument('--out', required=True,
                        type=str, help="Path to output file")

    return parser.parse_args()


def create_directory(directory):
    try:
        os.mkdir(directory)
        print(f"Directory '{directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    args = arg_parser()
    GENO_PREFIX_LIST = args.geno_prefix_list.split(',')
    POPS_LIST = args.pops_list.split(',')

    rgenes = os.path.join(args.tmpdir, 'rgenes')
    create_directory(rgenes)

    if not os.path.exists(args.out) or os.path.getsize(args.out) == 0:
        # If the file doesn't exist or is empty, write the header
        with open(args.out, "a") as f:
            f.write(OUTPUT_COLUMNS)

    sumstats_path_base = os.path.join(args.tmpdir, 'sumstats')
    create_directory(sumstats_path_base)
    ld_path_base = os.path.join(args.tmpdir, 'ld')
    create_directory(ld_path_base)
    metro_path_base = os.path.join(args.tmpdir, 'metro')
    create_directory(metro_path_base)

    for curr_pop in POPS_LIST:
        sumstats_path = os.path.join(sumstats_path_base, curr_pop)
        print(f'Sumstats_path for {curr_pop}', sumstats_path)
        create_directory(sumstats_path)

        ld_path_directory = os.path.join(ld_path_base, curr_pop)
        print(f'LD_path_directory {curr_pop}', ld_path_directory)
        create_directory(ld_path_directory)
    
    for i in range(5):
        print("Picking random gene. Number:", i)
        # pick random gene available in all populations
        random_gene_list, chr, min_num_snps = pick_rand_gene(
                                              args.plink_path,
                                              GENO_PREFIX_LIST, 
                                              rgenes)
        for NUMTARGET in NUMTARGET_LIST:
            for NCAUSAL in NCAUSAL_ARR:
                if NCAUSAL > min_num_snps:
                    print("Error: Too many Causal SNPs specified. Change the number of minimum SNPs in pick_rand_gene function")
                    return 
                index_causals = np.random.choice([i for i in range(0, min_num_snps)], 
                                                 NCAUSAL, replace=False)
                for H2GE in PHEN_VAR_GENE_COMPONENT:
                    for EQTL_H2 in EQTL_H2_ARR:
                        for CORRELATION in CORRELATIONS_GE:
                            simulated_genotypes = dict()
                            simulated_betas = dict()
                            simulated_gexprs = dict()
                            simulated_ld = dict()
                            sumstats_files_external = []
                            marginal_sumstats_files = [] # for METRO, need marginal eQTL z scores per snp for each population
                            eqtl_snp_correlation_files = [] # for METRO, need snp correlation matrices for each population
                            eqtl_samplesizes = [] # for METRO, need sample sizes for each summary statistic 

                            effects, realized_h = correlated_effects_cholesky(EQTL_H2,
                                                                EQTL_H2,
                                                                EQTL_H2,
                                                                corr=CORRELATION,
                                                                nums=NCAUSAL)
                            realized_h = ','.join(f"{x:.2e}" for x in realized_h.flatten())
                            # effects_df is a DataFrame, where columns are
                            # populations, rows represent SNP weights for causal 
                            # SNPs
                            effects_df = pd.DataFrame(effects)
                            effects_df.columns = POPS_LIST

                            ### We use this for loop to create summary statistics
                            for i, (rand_gene, geno_prefix, curr_pop) in enumerate(zip(random_gene_list, GENO_PREFIX_LIST, POPS_LIST)):
                                # 1. Simulate LD, Simulate Genotypes
                                print("Simlate LD and Genotypes")
                                bim, L = get_ld(rand_gene)
                                CURR_NUM_PEOPLE = DEFAULT_NUM_OF_PEOPLE_EQTL

                                # target population is always first
                                if i == 0:
                                    CURR_NUM_PEOPLE = NUMTARGET

                                Z = sim_geno(L, CURR_NUM_PEOPLE)
                                # 2. Simulate Beta, Simulate Gene Expression
                                print("Simulate Latent Betas and Gene Expression")
                                beta = create_betas(effects=effects_df[curr_pop].values,
                                                num_causal=NCAUSAL,
                                                num_snps=L.shape[0],
                                                causal_index=index_causals)
                                gexpr = sim_trait(np.dot(Z, beta), EQTL_H2)[0]
                                # 3. Build SuSiE summary statistics
                                curr_gene = f'{os.path.basename(rand_gene)}_{NCAUSAL}_{EQTL_H2:.5f}_{H2GE:.5f}_{CORRELATION}_{CURR_NUM_PEOPLE}'
                                print(f"Build SuSiE summary statistics for {curr_gene}")
                                sumstats_path = os.path.join(sumstats_path_base, curr_pop)
                                ld_path = os.path.join(sumstats_path_base, curr_pop)

                                simulated_genotypes[curr_pop] = Z
                                simulated_betas[curr_pop] = beta
                                simulated_gexprs[curr_pop] = gexpr
                                simulated_ld[curr_pop] = L

                                # we run build_susie_sumstats for both target and external population 
                                # for target, we do not need to fine-mapping results, but the marginal eqtl z scores and snp correlation matrices are inputs to metro
                                sumstat_output_file, marginal_sumstats_file, snp_corr_plink_file = build_susie_sumstats(
                                    bim=bim,
                                    geno=Z,
                                    gexpr=gexpr,
                                    sumstats_path=sumstats_path,
                                    ld_path=ld_path,
                                    gene=curr_gene,
                                    gene_pref=geno_prefix + chr,
                                    plink_path=args.plink_path,
                                    cNUM_PEOPLE=CURR_NUM_PEOPLE
                                )
                                if i != 0: # for target population, we do not need the fine-mapping results 
                                    sumstats_files_external.append(sumstat_output_file)

                                marginal_sumstats_files.append(marginal_sumstats_file)
                                eqtl_snp_correlation_files.append(snp_corr_plink_file)
                                eqtl_samplesizes.append(CURR_NUM_PEOPLE)

                            # 4. Run MAGEPRO; target population is first
                            curr_gene = f'{os.path.basename(random_gene_list[0])}_{NCAUSAL}_{EQTL_H2:.5f}_{H2GE:.5f}_{CORRELATION}_{NUMTARGET}'
                            gcta_output_path = os.path.join(args.tmpdir, curr_gene)

                            print("Run MAGEPRO for", curr_gene)
                            GCTA_h2, h2_pval, lasso_coef, lasso_r2, magepro_coef, magepro_r2 = run_magepro(
                                target_gene=random_gene_list[0],
                                Z_eqtl=simulated_genotypes[POPS_LIST[0]],
                                b_qtls=simulated_betas[POPS_LIST[0]],
                                gexpr=simulated_gexprs[POPS_LIST[0]],
                                tmp_path=gcta_output_path,
                                pop=POPS_LIST[0],
                                num_people=NUMTARGET,
                                sumstats_files_external=sumstats_files_external,
                                pops_external=POPS_LIST[1::]  # here we just slice
                            )
                            if GCTA_h2 == float('-inf'):
                                print("SuSiE summary statistics were not calculated properly for this gene.")
                                print("Debug information:")
                                print("Current genes:", random_gene_list)
                                print("Summary statistics:", sumstats_files_external)
                                continue

                            for NUM_GWAS in GWAS_SAMPLE_SIZE:
                                # 5. Perform TWAS
                                print("Perform TWAS. GWAS NUM:", NUM_GWAS)
                                SIGN = np.random.choice([-1, 1])
                                alpha = np.sqrt(H2GE / EQTL_H2) * SIGN
                                if NUM_GWAS < 100000:
                                    gwas = sim_gwas(simulated_ld[POPS_LIST[0]],
                                                    NUM_GWAS,
                                                    simulated_betas[POPS_LIST[0]] * alpha,
                                                    H2GE)
                                else:
                                    gwas = sim_gwasfast(simulated_ld[POPS_LIST[0]],
                                                    NUM_GWAS,
                                                    simulated_betas[POPS_LIST[0]] * alpha,
                                                    H2GE)

                                LD_test = np.dot(simulated_ld[POPS_LIST[0]],
                                                simulated_ld[POPS_LIST[0]].T)
                                z_twas_lasso, p_twas_lasso = compute_twas(
                                                                        gwas,
                                                                        lasso_coef,
                                                                        LD_test)

                                z_twas_magepro, p_twas_magepro = compute_twas(
                                                                        gwas,
                                                                        magepro_coef,
                                                                        LD_test)

                                # 6. Run METRO
                                print('Preparing inputs for METRO')
                                # compute marginal z scores
                                curr_gene = f'{os.path.basename(random_gene_list[0])}_{NCAUSAL}_{EQTL_H2:.5f}_{H2GE:.5f}_{CORRELATION}_{NUMTARGET}_{NUM_GWAS}'
                                eqtl_marginal_sumstats_paths = ",".join(marginal_sumstats_files)
                                eqtl_snp_corr_paths = ",".join(eqtl_snp_correlation_files)
                                eqtl_samplesizes_joined = ",".join([str(ss) for ss in eqtl_samplesizes])
                                gwas_file = os.path.join(metro_path_base, curr_gene + '_gwas.txt')
                                gwas.to_csv(gwas_file, header = True, sep = '\t', index = False)
                                # in the run_metro.R we add "_betas" or "_stat"
                                metro_output_path = os.path.join(metro_path_base, curr_gene)
                                print('Running METRO')
                                
                                metro_alpha, metro_p, metro_lrt, metro_df, metro_betas = run_metro(
                                    eqtl_paths=eqtl_marginal_sumstats_paths,
                                    pop_names=args.pops_list,
                                    eqtl_corr_paths=eqtl_snp_corr_paths,
                                    eqtl_samplesizes=eqtl_samplesizes_joined,
                                    gwas_file=gwas_file,
                                    ngwas=str(NUM_GWAS),
                                    out=metro_output_path)
                                if len(metro_betas) == 3 and metro_betas == 'NaN':
                                    magepro_metro_corr = 'NaN'
                                    print("Metro beta is NaN. Can't calculate correlation coefficient.")
                                else:
                                    magepro_metro_corr = np.corrcoef(magepro_coef, metro_betas)[0, 1]
                                
                                magepro_lasso_corr = np.corrcoef(magepro_coef, lasso_coef)[0, 1]
                                print('Writing row for:', curr_gene)
                                with open(args.out, "a") as f:
                                        f.write(f"""{curr_gene}{SEP}{EQTL_H2}{SEP}{H2GE}{SEP}{CORRELATION}{SEP}{NUM_GWAS}{SEP}{GCTA_h2}{SEP}{h2_pval}{SEP}{magepro_r2}{SEP}{z_twas_magepro}{SEP}{p_twas_magepro}{SEP}{lasso_r2}{SEP}{z_twas_lasso}{SEP}{p_twas_lasso}{SEP}{metro_alpha}{SEP}{metro_p}{SEP}{metro_lrt}{SEP}{metro_df}{SEP}{magepro_lasso_corr}{SEP}{magepro_metro_corr}{SEP}{realized_h}\n""")
        # clean up
        # command = f'rm -rf {rgenes}'
        # subprocess.run(command.split())
        # command = f'rm -rf {sumstats_path_base}'
        # subprocess.run(command.split())
        # command = f'rm -rf {ld_path_base}'
        # subprocess.run(command.split())


if __name__ == "__main__":
    main()
