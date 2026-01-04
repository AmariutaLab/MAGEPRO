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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from magepro_simulations_functions import * # SEE HERE FOR ALL FUNCTION CALLS

mvn = stats.multivariate_normal

args = sys.argv
plink_file = args[1] # path to plink files of simulated gene
samplesizes = int(args[2]) # number of people
pop = args[3] #population
genotype_file = args[4] #path to simulated genotypes
sumstats_file = args[5] #path to sumstats created in other poputation
sumstats_files = sumstats_file.split(',')
population_sumstat = args[6]
populations_sumstat = population_sumstat.split(',') # comma separated list of ancestries of sumstats
set_num_causal = int(args[7]) #number of causal snps
CAUSAL = [ int(index) for index in args[8].split(',') ] # indices of causal snps
sim = int(args[9]) #iteration of simulation
set_h2 = float(args[10]) #predetermiend h2g
temp_dir = args[11] #temporary directory for gcta
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
#make sure to pick a 1 mb region with enough snps (maybe > 30?) up to you! 
betas = sim_effect_sizes(set_h2, set_num_causal, bim.shape[0], CAUSAL)
beta_causal_list = [ betas[i] for i in CAUSAL ]
beta_causal = ','.join(map(str, beta_causal_list))
b_qtls = np.array(betas)
b_qtls = np.reshape(b_qtls.T, (bim.shape[0], 1))
#print(b_qtls.shape)

# --- SIMULATE LASSO GENE MODEL
best_penalty, coef, h2g, hsq_p, r2all, gexpr = sim_eqtl(z_eqtl, samplesizes, b_qtls, float(set_h2), temp_dir, threads) #this runs gcta and lasso 
#gexpr already standardized

# --- if the best lasso model with the best penalty gives coef of all 0, we have to use top1 to compute r2 afr
if np.all(coef == 0):
    print("using top1 backup")
    r2all, r2_top1, coef = top1_cv(samplesizes, z_eqtl, gexpr)

# --- PRINT HERITABILITY ESTIMATE FROM sim_eqtl() function 
print(("heritability " + pop + ": "))
print(h2g)

# --- PRSCSx SHRINKAGE 
executable_dir="/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/BENCHMARK/PRScsx"
ld_reference_dir="/expanse/lustre/projects/ddp412/kakamatsu/eQTLsummary/multipopGE/benchmark/LD_ref"
prscsx_working_dir=temp_dir+"/PRSCSx/"
prscsx_weights = PRSCSx_shrinkage(executable_dir, ld_reference_dir, prscsx_working_dir, sumstats_file, "500,500" , population_sumstat, bim, 1e-7)
prscsx_r2, prscsx_coef = prscsx_cv(samplesizes, z_eqtl, gexpr, prscsx_weights, best_penalty, 'lasso')

# --- PROCESS SUMMARY STATISTICS
num_causal_susie = 0
sumstats_weights = {}
for i in range(0,len(sumstats_files)):
    varname = populations_sumstat[i]+'weights'
    weights, pips = load_process_sumstats(sumstats_files[i], bim)
    sumstats_weights[varname] = weights
    num_causal_susie += len([index for index in CAUSAL if pips[index] >= 0.95])

# --- RUN CV MAGEPRO 
magepro_r2, magepro_coef = magepro_cv(samplesizes, z_eqtl, gexpr, sumstats_weights, best_penalty, 'lasso')

print("magepro: ")
print(magepro_r2)
print("prscsx: ")
print(prscsx_r2)
print("afronly: ")
print(r2all)

# --- POPULATE OUTPUTS
#coef = afr lasso model gene model 
lasso_causal_nonzero = len([index for index in CAUSAL if coef[index] != 0])
#magepro_coef = magepro gene model 
magepro_causal_nonzero = len([index for index in CAUSAL if magepro_coef[index] != 0])
#prscsx_coef = prscsx gene model
prscsx_causal_nonzero = len([index for index in CAUSAL if prscsx_coef[index] != 0])
#beta of causal snp from afr lasso
afr_B_causal_list = [ coef[i] for i in CAUSAL ]
afr_B_causal = ','.join(map(str, afr_B_causal_list))
#beta of causal snp from magepro
magepro_B_causal_list = [ magepro_coef[i] for i in CAUSAL ]
magepro_B_causal = ','.join(map(str, magepro_B_causal_list))
#beta of causal snp from prscsx
prscsx_B_causal_list = [ prscsx_coef[i] for i in CAUSAL ]
prscsx_B_causal = ','.join(map(str, prscsx_B_causal_list))
#actual beta
true_B_causal = beta_causal

#h2g = afr h2
#magepro_r2 = magepro cv r2 
#r2all = afronly magepro cv r2

filename = out_results + "/magepro_results_" + str(samplesizes) + "_h" + str(set_h2) + ".csv"

output = pd.DataFrame({'sim': sim, 'afr_h2': h2g, 'lasso_causal': lasso_causal_nonzero, 'magepro_causal': magepro_causal_nonzero, 'prscsx_causal': prscsx_causal_nonzero, 'afr_beta_causal': afr_B_causal, 'magepro_beta_causal': magepro_B_causal, 'prscsx_beta_causal': prscsx_B_causal, 'true_B_causal': true_B_causal, 'afr_r2': r2all, 'magepro_r2': magepro_r2, 'prscsx_r2': prscsx_r2, 'causal_susie': num_causal_susie}, index=[0])
if sim == 1:
    output.to_csv(filename, sep="\t", index=False, header = True)
else:
    output.to_csv(filename, sep="\t", index=False, header = False, mode='a')

