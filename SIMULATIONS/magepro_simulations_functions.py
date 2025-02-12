#!/usr/bin/env python
import argparse as ap
import sys
import numpy as np
import pandas as pd
import scipy.linalg as linalg
from numpy.linalg import multi_dot as mdot
from pandas_plink import read_plink
from scipy import stats
from sklearn import linear_model as lm
import random
import warnings
import scipy
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import subprocess, os
import gzip
import os.path  
from os import path  
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
import os

mvn = stats.multivariate_normal

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

# --- SIMULATED GENE EXPRESSION
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

# --- FIT GENE MODEL 
def fit_sparse_regularized_lm(Z, y, m):
    """
    Infer eqtl coefficients using LASSO regression for simeQTL function above. Uses the PLINK-style coordinate descent algorithm
    that is bootstrapped by the current h2g estimate.
    :param Z:  numpy.ndarray n x p genotype matrix
    :param y: numpy.ndarray gene expression for n individuals
    :param m: str indicating type of model ['enet', 'lasso']
    :return: (numpy.ndarray, float, float) tuple of the LASSO coefficients, the r-squared score, and log-likelihood
    """
    n, p = Z.shape
    alphas=np.logspace(-2,1.2,50,base=10)
    kf=KFold(n_splits=5)

    if m not in ['enet', 'lasso']:
        print("Invalid model type passed to param m")
        sys.exit(1)
    
    #lm() r2 extraction
    lm_r2_allpenalty=[] 
    
    for penalty in alphas:
       
        #every fold, predict expression
        predicted_expressions = np.zeros(n)

        CV_r2=[]
        for train_index, test_index in kf.split(Z):
            X_train, X_test = Z[train_index], Z[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train_std=(y_train-np.mean(y_train))/np.std(y_train)
            y_test_std=(y_test-np.mean(y_test))/np.std(y_test)
            if m == 'lasso':
                model = lm.Lasso(fit_intercept=True)
            else: # enet 
                model = lm.ElasticNet(fit_intercept=True)
            model.set_params(alpha=penalty,tol=1e-4,max_iter=1000)
            model.fit(X_train,y_train_std)

            #predict on testing 
            predicted_expressions[test_index] = np.dot(X_test, model.coef_)

        #evaluate r2 using lm
        lreg = LinearRegression().fit(np.array(predicted_expressions).reshape(-1, 1), y)
        r2_cv_penalty = lreg.score(np.array(predicted_expressions).reshape(-1, 1), y)
        lm_r2_allpenalty.append(r2_cv_penalty)

    besti=np.argmax(lm_r2_allpenalty)
    bestalpha=alphas[besti]
    if m == 'lasso':
        model = lm.Lasso(fit_intercept=True)
    else: # enet 
        model = lm.ElasticNet(fit_intercept=True)
    model.set_params(alpha=bestalpha,tol=1e-4,max_iter=1000)
    model.fit(Z, y)
    coef = model.coef_
    return bestalpha, coef, lm_r2_allpenalty[besti]


# --- SIMULATE EQTL 
def sim_eqtl(Z_qtl, nqtl, b_qtls, eqtl_h2, temp, thread_n): 
    """
    Simulate an eQTL study using `nqtl` individuals.
    :param Z_qtl: simulated genotypes
    :param nqtl: int the number of eQTL-panel genotypes to sample
    :param b_qtls: numpy.ndarray latent eQTL effects for the causal gene: SNPs x genes
    :param eqtl_h2: list of expression variance explained by linear model of SNPs for each gene (not used) rather min(1,b@LD_qtl[np.ix_(nearSNPsi,nearSNPsi)]@b)
    :param temp: name of temporary directory
    :return:  (numpy.ndarray x 6) Arrays of best lasso penalty, eqtl effect sizes, heritability, cross validation accuracies, simulated ge
    """
    allqtlshape = b_qtls.shape
    n, p = [float(x) for x in Z_qtl.shape] 

    tempdir = temp + "/temp"
    b = b_qtls[:,0]
    nonzeroindex = np.where(b != 0)[0]
    gexpr = sim_trait(np.dot(Z_qtl,b), eqtl_h2)[0]
    penalty, coef, r2= fit_sparse_regularized_lm(Z_qtl, gexpr, 'lasso')
    if r2>0:
        h2g,hsq_se,hsq_p = get_hsq(Z_qtl,gexpr,tempdir,"/expanse/lustre/projects/ddp412/kakamatsu/fusion_twas-master/gcta_nr_robust", thread_n)
    else:
        h2g=0
        hsq_se=10
        hsq_p=1
    return (penalty, coef,h2g,hsq_p,r2,gexpr)

# -- SIMULATE CAUSAL EFFECT SIZES
def sim_effect_sizes(h2g, num_causal, num_snps, causal_index):
    mean = 0
    variance = h2g/num_causal
    effect_sizes = np.zeros(num_snps)
    if isinstance(causal_index, list):
        for index in causal_index:
            effect_sizes[index] = np.random.normal(mean, np.sqrt(variance))
    else:
        effect_sizes[causal_index] = np.random.normal(mean, np.sqrt(variance))
    return effect_sizes

def sim_effect_sizes_only(h2g, num_causal):
    mean = 0
    variance = h2g/num_causal
    effect_sizes = np.zeros(num_causal)
    if num_causal >= 2:
        for index in range(num_causal):
            effect_sizes[index] = np.random.normal(mean, np.sqrt(variance))
    else:
        effect_sizes[0] = np.random.normal(mean, np.sqrt(variance))
    return effect_sizes

def create_betas(effects, num_causal, num_snps, causal_index):
    # effects = list of causal effect sizes 
    # num_causal = number of causal variants 
    # num_snps = number of snps 
    # causal_index = list containing indices of causal variants (same order as effects)
    betas = np.zeros(num_snps)
    if num_causal >= 2:
        for index in range(num_causal):
            betas[causal_index[index]] = effects[index]
    else:
        betas[causal_index[0]] = effects[0]
    return betas

def random_correlated_effect(effect1, heritability, correlation):
    # effect1 = first effect for which you want another effect that is correlated to it
    # heritability = preset PER-SNP heritability value (variance of the distribution of effect size)
    # correlation = desired correlation of the two effect sizes 
    effect_temp = np.random.normal(0, np.sqrt(heritability))
    effect2 = (correlation*effect1) + np.sqrt(1-(correlation**2))*effect_temp # see supplements for more information
    return (effect2)

def correlated_effects_cholesky(h1, h2, h3, corr, nums):
    # corr = genetic correlation
    # nums = number of causal snps
    # Note if nums is large, it is more efficient to loop through nums and sample size (1, 2) rather than sample (nums, 2) at once
    sigma1 = np.sqrt(h1 / nums)
    sigma2 = np.sqrt(h2 / nums)
    sigma3 = np.sqrt(h3 / nums)

    cov = np.array([
        [sigma1**2, corr * sigma1 * sigma2, corr * sigma1 * sigma3],
        [corr * sigma1 * sigma2, sigma2**2, corr * sigma2 * sigma3],
        [corr * sigma1 * sigma3, corr * sigma2 * sigma3, sigma3**2]
    ])
    L = np.linalg.cholesky(cov)
    effects = np.random.normal(loc=0, scale=1, size=(nums, 3))
    samples = effects @ L.T
    return (samples)

def get_hsq(geno, gene_expr, out, gcta, thread_n):
    nindv, nsnp = geno.shape
    grm = np.dot(geno, geno.T) / nsnp
    write_pheno(gene_expr, out)
    write_grm(grm, nsnp, out)
    run_gcta(gcta, out, thread_n)
    clean_up(out)
    hsq_f = '{}.hsq'.format(out)
    if os.path.exists(hsq_f):
        hsq_out = pd.read_table(hsq_f)
        hsq = hsq_out['Variance'][0]
        hsq_se = hsq_out['SE'][0]
        hsq_p = hsq_out['Variance'][8]
        command = 'rm {}'.format(hsq_f)
        subprocess.run(command.split())
        return hsq, hsq_se, hsq_p
    else:
        return 0,10000,1
    return None

def run_gcta(gcta, out, thread_n):
    command = '{} --grm-gz {} --pheno {} --reml --reml-no-constrain --thread-num {} --out {}'\
        .format(gcta, out, out+'.phen', thread_n, out)
    subprocess.run(command.split())
    #command = 'rm {}.log'.format(out)
    #subprocess.run(command.split())
    command = 'rm {}.phen'.format(out)
    subprocess.run(command.split())

def write_pheno(phen, out):
    out_f = open(out+'.phen', 'w')
    nindv = phen.shape[0]
    for i in range(nindv):
        out_f.write('{}\t{}\t{}\n'.format(i, i, phen[i]))
    out_f.close()

def clean_up(out):
    command = 'rm {}.grm.gz'.format(out)
    subprocess.run(command.split())
    command = 'rm {}.grm.id'.format(out)
    subprocess.run(command.split())

def write_grm(grm, nsnp, out):
    out_id_f = open(out+'.grm.id', 'w')
    out_f = gzip.open(out+'.grm.gz', 'w')
    nindv = grm.shape[0]
    for i in range(nindv):
        for j in range(0, i+1):
            out_f.write('{}\t{}\t{}\t{}\n'.format(i+1, j+1, nsnp,
                grm[i,j]).encode('utf-8'))
        out_id_f.write('{}\t{}\n'.format(i, i))
    out_f.close()
    out_id_f.close()

#compute marginal weights per snp
def marginal_weights(ge, genotypes):
    betas = []
    pvals = []

    for i in range(genotypes.shape[1]):
        x = sm.add_constant(genotypes[:, i].reshape(-1,1))
        mod = sm.OLS(ge, x)
        results = mod.fit()
        betas.append(results.params[1])
        pvals.append(results.pvalues[1])

    return betas, pvals

#take ge, genotypes and marginal betas, create top1 gene model and find r2 between predicted and actual ge
def top1(ge, genotypes, betas):
    weights = np.array(betas.copy())
    #top 1 gene model takes the highest marginally weighted snp 
    i_max = np.argmax(weights**2)
    zero = (weights != weights[i_max])
    weights[zero] = 0
    #calculate r2
    predicted_expression = np.dot(genotypes, weights)
    r2 = r2_score(ge, predicted_expression)
    #return top1 gene model weights and r2
    return weights, r2

# FUSION framework for marginal weights - alternative to marginal_weights(ge, genotypes)
def weights_marginal_FUSION(genos, pheno, beta=True):
    n_samples = len(pheno)
    
    if beta:
        eff_wgt = np.dot(genos.T, pheno) / (n_samples - 1)
    else:
        eff_wgt = np.dot(genos.T, pheno) / np.sqrt(n_samples - 1)
    
    return eff_wgt

def allele_qc(a1, a2, ref1, ref2):
    # Convert to uppercase
    a1 = a1.str.upper()
    a2 = a2.str.upper()
    ref1 = ref1.str.upper()
    ref2 = ref2.str.upper()
    
    # Function to flip alleles
    def flip_allele(ref):
        flip = ref.copy()
        flip[ref == "A"] = "T"
        flip[ref == "T"] = "A"
        flip[ref == "G"] = "C"
        flip[ref == "C"] = "G"
        return flip
    
    flip1 = flip_allele(ref1)
    flip2 = flip_allele(ref2)
    
    keep = ~( ((a1 == "A") & (a2 == "T")) | ((a1 == "T") & (a2 == "A")) |
             ((a1 == "C") & (a2 == "G")) | ((a1 == "G") & (a2 == "C")) ) # DON'T KEEP AMBIGUOUS SNPS
    keep &= (a1.isin(["A", "T", "G", "C"]) & a2.isin(["A", "T", "G", "C"])) # ONLY KEEP ATGC SNPs
    keep &= ~(((a1 == ref1) & (a2 != ref2)) | ((a1 == ref2) & (a2 != ref1)) |
              ((a1 == flip1) & (a2 != flip2)) | ((a1 == flip2) & (a2 != flip1))) # DON'T KEEP 3 ALLELES SNPs
    
    flip = ((a1 == ref2) & (a2 == ref1)) | ((a1 == flip2) & (a2 == flip1)) # FLIPPED ALLELES
    
    return {"keep": keep, "flip": flip}

def load_process_sumstats(file_sumstats, bim_df, sep="\t"):
    # LOAD AND PROCESS SUMMARY STATISTICS (SUSIE POSTERIOR) FOR MAGEPRO 

    # --- LOAD IN SUM STATS
    sumstats = pd.read_csv(file_sumstats, sep=sep)
    # If susie couldn't perform fine-mapping, we return.
    if 'POSTERIOR' not in sumstats.columns and 'PIP' not in sumstats.columns:
        return np.array(float('-inf')), np.array(float('-inf'))

    # --- get SNPs in common between bim and sumstats
    snps_bim = pd.DataFrame(bim_df[:, 1], columns=['SNP'])
    merged = sumstats.merge(snps_bim, on='SNP', how = 'inner')
    merged = merged.set_index('SNP')
    merged = merged.reindex(bim_df[:, 1])
    merged = merged.fillna(0)
    sumstats = merged.reset_index()


    # --- FLIP SIGNS, SUBSET SUM STATS TO SNPS IN COMMON
    # match snps, flip signs? - unfortunately the snps present in one population isn't always present in the other
    qc = allele_qc(pd.Series(sumstats['A1']), pd.Series(sumstats['A2']), pd.Series(bim_df[:,4]), pd.Series(bim_df[:,5]))

    sumstats.loc[qc['flip'], 'BETA'] = sumstats.loc[qc['flip'], 'BETA']  * -1
    sumstats.loc[qc['flip'], 'POSTERIOR'] = sumstats.loc[qc['flip'], 'POSTERIOR'] * -1
    sumstats.loc[~qc['keep'],'BETA'] = 0
    sumstats.loc[~qc['keep'], 'POSTERIOR'] = 0

    wgt = np.array(sumstats['POSTERIOR'])
    pips = np.array(sumstats['PIP'])

    return wgt, pips

def top1_cv(N, genotypes, phenotype):
    #r2all also has to be reassigned - have to perform 5-fold cross validation again using top1 model
    predicted_expressions = np.zeros(N)
    kf_top1=KFold(n_splits=5)
    for train_i_top1, test_i_top1 in kf_top1.split(genotypes):
        X_train_top1, X_test_top1 = genotypes[train_i_top1], genotypes[test_i_top1]
        y_train_top1, y_test_top1 = phenotype[train_i_top1], phenotype[test_i_top1] 
        y_train_std_top1=(y_train_top1-np.mean(y_train_top1))/np.std(y_train_top1)
        y_test_std_top1=(y_test_top1-np.mean(y_test_top1))/np.std(y_test_top1)
        wgts = weights_marginal_FUSION(X_train_top1, y_train_std_top1, beta = True)
        top1_wgts, r2_top1 = top1(y_test_std_top1, X_test_top1, wgts)
        top1_wgts_train, r2_top1_train = top1(y_train_std_top1, X_train_top1, wgts)
        predicted_expressions[test_i_top1] = np.dot(X_test_top1, top1_wgts)
    lreg = LinearRegression().fit(np.array(predicted_expressions).reshape(-1, 1), phenotype)
    r2_top1_cv = lreg.score(np.array(predicted_expressions).reshape(-1, 1), phenotype) # cross validation r2
    #full model 
    wgts_full = weights_marginal_FUSION(genotypes, phenotype, beta = True)
    coef_top1_full, r2_top1_training = top1(phenotype, genotypes, wgts_full) # top1 gene models weights, total training r2 

    return r2_top1_cv, r2_top1_training, coef_top1_full


def magepro_cv(N, geno_magepro, pheno_magepro, susie_weights, best_penalty, m):
    # N = sample size 
    # geno_magepro = genotypes
    # pheno_magepro = gene expression or phenotype data 
    # susie_weights = dictionary of external datasets to use (susie posterior weights)
        # Ancestry : Array of weights
    # best_penalty = best alpha penalty from single pop model corresponding to m (from lasso or enet)
    # m = str indicating type of model ['enet', 'lasso']

    if m not in ['enet', 'lasso']:
        print("Invalid model type passed to param m")
        sys.exit(1)

    alphas_magepro=np.logspace(-2,1.2,50,base=10)
    kf_magepro=KFold(n_splits=5)
    #store r2 from lm for each penalty
    lm_r2_allpenalty=[] 
    for penalty_magepro in alphas_magepro:
        #everyfold record predicted expression on testing set
        predicted_expressions = np.zeros(N)
        for train_index_magepro, test_index_magepro in kf_magepro.split(geno_magepro):
            y_train_magepro, y_test_magepro = pheno_magepro[train_index_magepro], pheno_magepro[test_index_magepro]
            y_train_std_magepro=(y_train_magepro-np.mean(y_train_magepro))/np.std(y_train_magepro)
            y_test_std_magepro=(y_test_magepro-np.mean(y_test_magepro))/np.std(y_test_magepro)
            X_train_magepro, X_test_magepro = geno_magepro[train_index_magepro], geno_magepro[test_index_magepro]
            # afronly weights
            if m == 'lasso':
                model = lm.Lasso(fit_intercept=True)
            else:
                model = lm.ElasticNet(fit_intercept=True)
            model.set_params(alpha=best_penalty,tol=1e-4,max_iter=1000)
            model.fit(X_train_magepro, y_train_std_magepro)
            coef_magepro = model.coef_
            if np.all(coef_magepro == 0):
                wgts = weights_marginal_FUSION(X_train_magepro, y_train_magepro, beta = True)
                coef_magepro, r2_top1 = top1(y_test_std_magepro, X_test_magepro, wgts)
            #prepare for ridge regression to find optimal combination of AFR gene model and EUR sumstat
            X_train_magepro2 = np.dot(X_train_magepro, coef_magepro.reshape(-1, 1))
            X_test_magepro2 = np.dot(X_test_magepro, coef_magepro.reshape(-1, 1))
            for ancestry, weights in susie_weights.items():
                X_train_magepro2 = np.hstack((X_train_magepro2, np.dot(X_train_magepro, weights.reshape(-1, 1))))
                X_test_magepro2 = np.hstack((X_test_magepro2, np.dot(X_test_magepro, weights.reshape(-1, 1))))
            ridge = Ridge(alpha=penalty_magepro) #now finding best penalty for ridge regression 
            ridge.fit(X_train_magepro2,y_train_std_magepro)
            ridge_coef = ridge.coef_
            #predict on testing
            predicted_expressions[test_index_magepro] = np.dot(X_test_magepro2, ridge_coef)
        #record r2 from lm
        lreg = LinearRegression().fit(np.array(predicted_expressions).reshape(-1, 1), pheno_magepro)
        r2_cv_penalty = lreg.score(np.array(predicted_expressions).reshape(-1, 1), pheno_magepro)
        lm_r2_allpenalty.append(r2_cv_penalty) 
    besti_magepro=np.argmax(lm_r2_allpenalty)
    bestalpha_magepro=alphas_magepro[besti_magepro]
    magepro_r2 = lm_r2_allpenalty[besti_magepro] # returning this for cv r2

    # full model for training r2 and full coef
    if m == 'lasso':
        model = lm.Lasso(fit_intercept=True)
    else:
        model = lm.ElasticNet(fit_intercept=True)
    model.set_params(alpha=best_penalty,tol=1e-4,max_iter=1000)
    model.fit(geno_magepro, pheno_magepro)
    coef_magepro = model.coef_
    if np.all(coef_magepro == 0):
        wgts = weights_marginal_FUSION(geno_magepro, pheno_magepro, beta = True)
        coef_magepro, r2_top1 = top1(pheno_magepro, geno_magepro, wgts)
    X_magepro = np.dot(geno_magepro, coef_magepro.reshape(-1, 1))
    for ancestry, weights in susie_weights.items():
        X_magepro = np.hstack((X_magepro, np.dot(geno_magepro, weights.reshape(-1, 1))))
    ridge = Ridge(alpha=bestalpha_magepro)
    ridge.fit(X_magepro, pheno_magepro)
    wgts_sep = coef_magepro.reshape(-1, 1)
    for ancestry, weights in susie_weights.items():
        wgts_sep = np.hstack((wgts_sep, weights.reshape(-1, 1)))
    magepro_coef = np.dot(wgts_sep, ridge.coef_) #magepro coef

    return magepro_r2, magepro_coef


def PRSCSx_shrinkage(exec_dir, ldref_dir, working_dir, input, ss, pp, BIM, phi):
    # exec_dir = directory where PRSCSx executable is located 
    # ldref_dir = directory with ld reference files from PRSCSx github 
    # working_dir = directory to write intermediate files for PRSCSx
    # input = comma separated list of input sumstats 
    # ss = comma separated list of input sumstats sample sizes 
    # pp = comma separated list of input sumstats ancestries 
    # BIM = target population bim file 
    # phi = shrinkage parameter 

    # run PRSCSx tool
    BIM = pd.DataFrame(BIM)
    BIM.to_csv(os.path.join(working_dir, "snps.bim"), sep='\t', index=False, header=False, quoting=False)
    arg = f"python {exec_dir}/PRScsx.py --ref_dir={ldref_dir} --bim_prefix={working_dir}snps --sst_file={input} --n_gwas={ss} --pop={pp} --chrom={BIM.iloc[0, 0]} --phi={phi} --out_dir={working_dir} --out_name=results"
    subprocess.run(arg, shell=True, check=True)

    # collect PRSCSx results
    formatted_shrinkage = "{:.2e}".format(phi).replace(".00e", "e")
    BIM.set_index(1, inplace=True)
    prscsx_sumstats_weights = {}
    populations = pp.split(',')
    for i in range(0,len(populations)):
        varname = populations[i]+'weights_prscsx'
        file_path = os.path.join(working_dir, f"results_{populations[i]}_pst_eff_a1_b0.5_phi{formatted_shrinkage}_chr{BIM.iloc[0, 0]}.txt")
        table = pd.read_csv(file_path, sep='\t', header=None)
        table.set_index(1, inplace=True)
        table = table.reindex(BIM.index, fill_value=0)
        prscsx_sumstats_weights[varname] = np.array(table[5])

    return prscsx_sumstats_weights


def prscsx_cv(N, geno_prscsx, pheno_prscsx, prscsx_weights, best_penalty, m):
    # N = sample size 
    # geno_prscsx = genotypes
    # pheno_prscsx = gene expression or phenotype data 
    # prscsx_weights = dictionary of external datasets to use from prscsx shrinkage 
        # Ancestry : Array of weights
    # best_penalty = best alpha penalty from single pop model corresponding to m (from lasso or enet)
    # m = str indicating type of model ['enet', 'lasso']

    if m not in ['enet', 'lasso']:
        print("Invalid model type passed to param m")
        sys.exit(1)

    alphas_prscsx=np.logspace(-2,1.2,50,base=10)
    kf_prscsx=KFold(n_splits=5)
    #store r2 from lm for each penalty
    lm_r2_allpenalty=[] 
    for penalty_prscsx in alphas_prscsx:
        #everyfold record predicted expression on testing set
        predicted_expressions = np.zeros(N)
        for train_index_prscsx, test_index_prscsx in kf_prscsx.split(geno_prscsx):
            y_train_prscsx, y_test_prscsx = pheno_prscsx[train_index_prscsx], pheno_prscsx[test_index_prscsx]
            y_train_std_prscsx=(y_train_prscsx-np.mean(y_train_prscsx))/np.std(y_train_prscsx)
            y_test_std_prscsx=(y_test_prscsx-np.mean(y_test_prscsx))/np.std(y_test_prscsx)
            X_train_prscsx, X_test_prscsx = geno_prscsx[train_index_prscsx], geno_prscsx[test_index_prscsx]
            #afronly weights
            if m == 'lasso':
                model = lm.Lasso(fit_intercept=True)
            else:
                model = lm.ElasticNet(fit_intercept=True)
            model.set_params(alpha=best_penalty,tol=1e-4,max_iter=1000)
            model.fit(X_train_prscsx, y_train_std_prscsx)
            coef_prscsx = model.coef_
            if np.all(coef_prscsx == 0):
                wgts = weights_marginal_FUSION(X_train_prscsx, y_train_prscsx, beta = True)
                coef_prscsx, r2_top1 = top1(y_test_std_prscsx, X_test_prscsx, wgts)
            #prepare for linear regression to find optimal combination of AFR gene model and EUR sumstat
            X_train_prscsx2 = np.dot(X_train_prscsx, coef_prscsx.reshape(-1, 1))
            X_test_prscsx2 = np.dot(X_test_prscsx, coef_prscsx.reshape(-1, 1))
            for ancestry, weights in prscsx_weights.items():
                X_train_prscsx2 = np.hstack((X_train_prscsx2, np.dot(X_train_prscsx, weights.reshape(-1, 1))))
                X_test_prscsx2 = np.hstack((X_test_prscsx2, np.dot(X_test_prscsx, weights.reshape(-1, 1))))
            linear_regression = LinearRegression()
            linear_regression.fit(X_train_prscsx2, y_train_std_prscsx)
            linear_coef = linear_regression.coef_
            #predict on testing
            predicted_expressions[test_index_prscsx] = np.dot(X_test_prscsx2, linear_coef)
        #record r2 from lm
        lreg = LinearRegression().fit(np.array(predicted_expressions).reshape(-1, 1), pheno_prscsx)
        r2_cv_penalty = lreg.score(np.array(predicted_expressions).reshape(-1, 1), pheno_prscsx)
        lm_r2_allpenalty.append(r2_cv_penalty) 
    besti_prscsx=np.argmax(lm_r2_allpenalty)
    bestalpha_prscsx=alphas_prscsx[besti_prscsx]
    prscsx_r2 = lm_r2_allpenalty[besti_prscsx] # returning this for cv r2

    # full model for training r2 and full coef
    if m == 'lasso':
        model = lm.Lasso(fit_intercept=True)
    else:
        model = lm.ElasticNet(fit_intercept=True)
    model.set_params(alpha=best_penalty,tol=1e-4,max_iter=1000)
    model.fit(geno_prscsx, pheno_prscsx)
    coef_prscsx = model.coef_
    if np.all(coef_prscsx == 0):
        wgts = weights_marginal_FUSION(geno_prscsx, pheno_prscsx, beta = True)
        coef_prscsx, r2_top1 = top1(pheno_prscsx, geno_prscsx, wgts)
    X_prscsx = np.dot(geno_prscsx, coef_prscsx.reshape(-1, 1))
    for ancestry, weights in prscsx_weights.items():
        X_prscsx = np.hstack((X_prscsx, np.dot(geno_prscsx, weights.reshape(-1, 1))))
    linear_regression = LinearRegression()
    linear_regression.fit(X_prscsx, pheno_prscsx)
    linear_coef = linear_regression.coef_
    wgts_sep = coef_prscsx.reshape(-1, 1)
    for ancestry, weights in prscsx_weights.items():
        wgts_sep = np.hstack((wgts_sep, weights.reshape(-1, 1)))
    prscsx_coef = np.dot(wgts_sep, linear_coef)

    return prscsx_r2, prscsx_coef



