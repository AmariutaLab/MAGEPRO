#!/usr/bin/env python
import argparse as ap
import sys
import numpy as np
import pandas as pd
from pandas_plink import read_plink
import random
import warnings
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from magepro_simulations_functions import * # SEE HERE FOR ALL FUNCTION CALLS

args = sys.argv
set_num_causal = int(args[1]) #how many causal variants?
set_h2 = float(args[2]) # heritability of target
set_h2_2 = float(args[3]) # heritability of external 1
set_h2_3 = float(args[4]) # heritability of external 2
corr = float(args[5])
pops = [str(pop) for pop in args[6].split(',')] # comma separated strings of populations in order of set_h2 set_h2_2 set_h2_3
out = args[7]


effects = correlated_effects_cholesky(set_h2, set_h2_2, set_h2_3, corr, set_num_causal)

df = pd.DataFrame(effects)
df.columns = pops
df.to_csv(out, index = False, sep = '\t')
