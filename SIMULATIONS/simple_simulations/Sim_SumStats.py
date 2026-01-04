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
import time
import subprocess, os
import gzip
import os.path  
from os import path  
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from magepro_simulations_functions import * # SEE HERE FOR ALL FUNCTION CALLS

mvn = stats.multivariate_normal

args = sys.argv
plink_file = args[1] # path to plink files of simulated gene
samplesizes = int(args[2]) # number of people to simulated genotypes for 
pop = args[3] #population
genotype_file = args[4] #path to simulated genotypes
set_num_causal = int(args[5]) #how many causal variants?
CAUSAL = [ int(index) for index in args[6].split(',') ] #indices of causal variants
sim = int(args[7]) #iteration of simulation
set_h2 = float(args[8]) #predetermined h2g
samplesize_target = int(args[9]) #sample size
out_sumstat = args[10] #output dir for sumstats
temp_dir = args[11] #dir for temporary files for gcta
out_results = args[12]
threads = args[13]

# --- READ IN SIMULATED GENE PLINK FILES
bim, fam, G_all = read_plink(plink_file, verbose=False)  
G_all = G_all.T
bim = np.array(bim)

# --- READ IN SIMULATED GENOTYPE
z_eqtl = np.array(pd.read_csv(genotype_file, sep = "\t", header = None))

# --- SIMULATE CAUSAL EFFECT SIZES 
#causal eqtl effect sizes; matrix dim = snps x 1 gene
#if our sim only has 1 gene, then we just need a simple vector of mostly zeros and 1 nonzero effect size if 1 causal variant. 
#draw from mean 0 var = 0.05 if 1 causal variant, draw from mean 0 var = h2g/#causal sps if more than 1 causal variant. 
betas = sim_effect_sizes(set_h2, set_num_causal, bim.shape[0], CAUSAL)
b_qtls = np.array(betas) 
b_qtls = np.reshape(b_qtls.T, (bim.shape[0], 1))

best_penalty, coef, h2g, hsq_p, r2all, gexpr = sim_eqtl(z_eqtl, samplesizes, b_qtls, float(set_h2), temp_dir, threads) #this runs gcta and lasso

#check heritability
print(("heritability " + pop + ": "))
print(h2g)


# --- RUN LINEAR REGRESSION PER SNP TO CREATE SUMSTATS
ge_regression = gexpr #already standardized

betas = []
pvals = []
std_errs = []

for i in range(z_eqtl.shape[1]):
    x = sm.add_constant(z_eqtl[:, i].reshape(-1,1))
    mod = sm.OLS(ge_regression, x)
    results = mod.fit()
    betas.append(results.params[1])
    pvals.append(results.pvalues[1])
    std_errs.append(results.bse[1])

# PRSCSx format
    # SNP A1 A2 BETA SE

sumstats = pd.DataFrame({'SNP':bim[:,1], 'A1': bim[:,4], 'A2': bim[:,5], 'BETA':betas, 'SE':std_errs, 'P': pvals})

filename = out_sumstat + "/sumstats_" + pop + ".csv"
sumstats.to_csv(filename, sep="\t", index=False, header = True)

lasso_causal_nonzero = len([index for index in CAUSAL if coef[index] != 0]) #number of nonzero causal variants 

#write results
filename = out_results + "/" + pop + "_h2_causal_" + str(samplesize_target) + "_h" + str(set_h2) + ".csv"
h2_causal = pd.DataFrame({'sim': sim, 'h2': h2g, 'lasso_casual': lasso_causal_nonzero, 'r2': r2all}, index=[0])
if sim == 1:
    h2_causal.to_csv(filename, sep="\t", index=False, header = True)
else:
    h2_causal.to_csv(filename, sep="\t", index=False, header = False, mode='a')
