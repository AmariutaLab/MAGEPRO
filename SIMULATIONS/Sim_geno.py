import argparse as ap
import sys
import numpy as np
import pandas as pd
import scipy.linalg as linalg
from pandas_plink import read_plink
from scipy import stats
import random
import warnings
import scipy
import time
import subprocess, os
import gzip
import os.path  
from os import path  
from magepro_simulations_functions import * # SEE HERE FOR ALL FUNCTION CALLS

mvn = stats.multivariate_normal

args = sys.argv
file_name = args[1] # path to plink files of simulated gene
nums_people = args[2] # number of people to simulated genotypes for 
pop = args[3] # population group for naming
out = args[4] #output directory

# --- added 5/25 for testing
ldout = args[5]
# ---

# --- read in 1kg plink files
bim, fam, G = read_plink(file_name, verbose=False) 
G = G.T
bim = np.array(bim)

#np.random.seed(12345)    

# --- estimate LD for population from PLINK data
n, p = [float(x) for x in G.shape]
p_int = int(p)
mafs = np.mean(G, axis=0) / 2
G -= mafs * 2
std_devs = np.std(G, axis=0)
#std_devs[std_devs == 0] = 1 # when standard deviation at a snp is 0, set it to 1 to prevent NA values in LD matrix
G /= std_devs

# --- regularize so that LD is PSD (positive semi definite)
LD = (np.dot(G.T, G) / n) + np.eye(p_int) * 0.1 #this is the LD matrix!!!

# --- added 5/25 for testing
LD = LD / (1 + 0.1)
LD = (LD + LD.T) / 2 
# sometimes due to numerical precision susie will throw a warning that matrix is not symmetric
# I checked and it seems like it is symmetric up to precision 1e-6
# just in case we will take the average of itself and its transpose
# ---

# --- compute cholesky decomp for faster sampling/simulation
L = linalg.cholesky(LD, lower=True)

# --- added 5/25 for testing
pd.DataFrame(LD).to_csv(f'{ldout}/{pop}_LD.csv', index=False, header=False)
#np.savez(f'{ldout}/{pop}_LD.npz', LD=LD)
# ---

Z_qtl = pd.DataFrame(sim_geno(L,int(nums_people)))  
filename = out + "/simulated_genotypes_" + pop + ".csv"
Z_qtl.round(4).to_csv(filename, sep="\t", index=False, header = False)
