#!/usr/bin/env python
import argparse as ap
import sys
import numpy as np
import pandas as pd
from pandas_plink import read_plink
import random
import warnings

args = sys.argv
file_name = args[1] # path to plink files of simulated gene
CAUSAL = int(args[2])

bim, fam, G = read_plink(file_name, verbose=False) 
G = G.T
bim = np.array(bim)

np.random.seed(12345)
# estimate LD for population from PLINK data
n, p = [float(x) for x in G.shape]
p_int = int(p)
mafs = np.mean(G, axis=0) / 2
G -= mafs * 2
G /= np.std(G, axis=0)
# regularize so that LD is PSD (positive semi definite)
LD = np.dot(G.T, G) / n + np.eye(p_int) * 0.1 #this is the LD matrix!!!
high_LD = LD.compute()[CAUSAL, :] #LD with casaul snp
high_LD_snps = np.where( (high_LD > 0.8) & (np.arange(len(high_LD)) != CAUSAL) )[0] #indices of snps in high LD with causal, excluding itself
if len(high_LD_snps) > 0:
    high_LD_snp = high_LD_snps[np.argmax(high_LD[high_LD_snps])] #choose the index with the largest LD r2 value
else:
    sys.exit(1)

print(high_LD_snp)
