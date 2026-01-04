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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
#parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
#target_pop_betas = [ float(index) for index in args[11].split(',') ] 
betas_file = args[11] # file holding effect sizes for each ancestry
temp_dir = args[12] #temporary directory for gcta
out_results = args[13]
threads = args[14]

# --- READ IN SIMULATED GENE PLINK FILES
bim, fam, G_all = read_plink(plink_file, verbose=False)  
G_all = G_all.T
bim = np.array(bim)

# --- READ IN SIMULATED GENOTYPE
z_eqtl = np.array(pd.read_csv(genotype_file, sep = "\t", header = None))

# --- SIMULATE CAUSAL EFFECT SIZES 
#causal eqtl effect sizes; matrix dim = snps x 1 gene
df_betas = pd.read_csv(betas_file, sep = '\t')
target_pop_betas = df_betas[pop].values
betas = create_betas(target_pop_betas, set_num_causal, bim.shape[0], CAUSAL)
beta_causal = ','.join(map(str, target_pop_betas))

b_qtls = np.array(betas)
b_qtls = np.reshape(b_qtls.T, (bim.shape[0], 1))
#print(b_qtls.shape)

# --- SIMULATE LASSO GENE MODEL
best_penalty_lasso, coef, h2g, hsq_p, r2_lasso, gexpr = sim_eqtl(z_eqtl, samplesizes, b_qtls, float(set_h2), temp_dir, threads) #this runs gcta and lasso 
#gexpr already standardized

# --- FIT ENET GENE MODEL 
best_penalty_enet, coef_enet, r2_enet = fit_sparse_regularized_lm(z_eqtl, gexpr, 'enet')

# --- if the best lasso/enet model with the best penalty gives coef of all 0, we have to use top1 to compute r2 afr
if np.all(coef == 0):
    print("using top1 backup")
    r2_lasso, r2_top1, coef = top1_cv(samplesizes, z_eqtl, gexpr)
    r2_lasso = np.nan # added later to know when lasso or enet resulted in all 0 coefs, in which case top1 was used as the base model

if np.all(coef_enet == 0):
    print("using top1 backup")
    r2_enet, r2_top1_enet, coef_enet = top1_cv(samplesizes, z_eqtl, gexpr)
    r2_enet = np.nan

# --- PRINT HERITABILITY ESTIMATE FROM sim_eqtl() function 
print(("heritability " + pop + ": "))
print(h2g)

# --- PRSCSx SHRINKAGE 
executable_dir="/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/BENCHMARK/PRScsx"
ld_reference_dir="/expanse/lustre/projects/ddp412/kakamatsu/eQTLsummary/multipopGE/benchmark/LD_ref"
prscsx_working_dir=temp_dir+"/PRSCSx/"
prscsx_weights = PRSCSx_shrinkage(executable_dir, ld_reference_dir, prscsx_working_dir, sumstats_file, "500,500" , population_sumstat, bim, 1e-7)
prscsx_r2, prscsx_coef = prscsx_cv(samplesizes, z_eqtl, gexpr, prscsx_weights, best_penalty_lasso, 'lasso')
prscsx_enet_r2, prscsx_enet_coef = prscsx_cv(samplesizes, z_eqtl, gexpr, prscsx_weights, best_penalty_enet, 'enet')
#prscsx_r2, prscsx_coef = prscsx_cv_retune(samplesizes, z_eqtl, gexpr, prscsx_weights, 'lasso')
#prscsx_enet_r2, prscsx_enet_coef = prscsx_cv_retune(samplesizes, z_eqtl, gexpr, prscsx_weights, 'enet')

# --- PROCESS SUMMARY STATISTICS
num_causal_susie = 0
sumstats_weights = {}
for i in range(0,len(sumstats_files)):
    varname = populations_sumstat[i]+'weights'
    weights, pips = load_process_sumstats(sumstats_files[i], bim)
    sumstats_weights[varname] = weights
    num_causal_susie += len([index for index in CAUSAL if pips[index] >= 0.95])

# --- RUN CV MAGEPRO 
magepro_r2, magepro_coef = magepro_cv(samplesizes, z_eqtl, gexpr, sumstats_weights, best_penalty_lasso, 'lasso')
magepro_enet_r2, magepro_enet_coef = magepro_cv(samplesizes, z_eqtl, gexpr, sumstats_weights, best_penalty_enet, 'enet')
#magepro_r2, magepro_coef = magepro_cv_retune(samplesizes, z_eqtl, gexpr, sumstats_weights, 'lasso')
#magepro_enet_r2, magepro_enet_coef = magepro_cv_retune(samplesizes, z_eqtl, gexpr, sumstats_weights, 'enet')

print("magepro lasso: ")
print(magepro_r2)
print("magepro enet: ")
print(magepro_enet_r2)
print("prscsx lasso: ")
print(prscsx_r2)
print("prscsx enet: ")
print(prscsx_enet_r2)
print("afronly lasso: ")
print(r2_lasso)
print("afronly enet: ")
print(r2_enet)

# --- POPULATE OUTPUTS
#coef = afr lasso model gene model 
lasso_causal_nonzero = len([index for index in CAUSAL if coef[index] != 0])
enet_causal_nonzero = len([index for index in CAUSAL if coef_enet[index] != 0])
#magepro_coef = magepro gene model 
magepro_causal_nonzero = len([index for index in CAUSAL if magepro_coef[index] != 0])
magepro_enet_causal_nonzero = len([index for index in CAUSAL if magepro_enet_coef[index] != 0])
#prscsx_coef = prscsx gene model
prscsx_causal_nonzero = len([index for index in CAUSAL if prscsx_coef[index] != 0])
prscsx_enet_causal_nonzero = len([index for index in CAUSAL if prscsx_enet_coef[index] != 0])
#beta of causal snp from afr lasso
afr_B_causal_list = [ coef[i] for i in CAUSAL ]
afr_B_causal = ','.join(map(str, afr_B_causal_list))
afr_enet_B_causal_list = [ coef_enet[i] for i in CAUSAL ]
afr_enet_B_causal = ','.join(map(str, afr_enet_B_causal_list))
#beta of causal snp from magepro
magepro_B_causal_list = [ magepro_coef[i] for i in CAUSAL ]
magepro_B_causal = ','.join(map(str, magepro_B_causal_list))
magepro_enet_B_causal_list = [ magepro_enet_coef[i] for i in CAUSAL ]
magepro_enet_B_causal = ','.join(map(str, magepro_enet_B_causal_list))
#beta of causal snp from prscsx
prscsx_B_causal_list = [ prscsx_coef[i] for i in CAUSAL ]
prscsx_B_causal = ','.join(map(str, prscsx_B_causal_list))
prscsx_enet_B_causal_list = [ prscsx_enet_coef[i] for i in CAUSAL ]
prscsx_enet_B_causal = ','.join(map(str, prscsx_enet_B_causal_list))
#actual beta
true_B_causal = beta_causal

#h2g = afr h2
#magepro_r2 = magepro cv r2 
#r2_lasso = afronly magepro cv r2

filename = out_results + "/magepro_results_" + str(samplesizes) + "_h" + str(set_h2) + ".csv"

output = pd.DataFrame({'sim': sim, 'afr_h2': h2g, 'lasso_causal': lasso_causal_nonzero, 'enet_causal': enet_causal_nonzero, 'magepro_causal': magepro_causal_nonzero, 'magepro_enet_causal': magepro_enet_causal_nonzero, 'prscsx_causal': prscsx_causal_nonzero, 'prscsx_enet_causal': prscsx_enet_causal_nonzero, 'afr_beta_causal': afr_B_causal, 'afr_enet_beta_causal': afr_enet_B_causal, 'magepro_beta_causal': magepro_B_causal, 'magepro_enet_beta_causal': magepro_enet_B_causal, 'prscsx_beta_causal': prscsx_B_causal, 'prscsx_enet_beta_causal': prscsx_enet_B_causal, 'true_B_causal': true_B_causal, 'afr_r2': r2_lasso, 'afr_enet_r2': r2_enet, 'magepro_r2': magepro_r2, 'magepro_enet_r2': magepro_enet_r2, 'prscsx_r2': prscsx_r2, 'prscsx_enet_r2': prscsx_enet_r2, 'causal_susie': num_causal_susie}, index=[0])

if sim != 1 and not path.exists(filename):
    print("WARNING: first iteration crashed, check error files")

output.to_csv(
    filename,
    sep="\t",
    index=False,
    header=not path.exists(filename),
    mode='a'
)

#if sim == 1:
    #output.to_csv(filename, sep="\t", index=False, header = True)
#else:
    #output.to_csv(filename, sep="\t", index=False, header = False, mode='a')

