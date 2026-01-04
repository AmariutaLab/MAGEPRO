#!/usr/bin/env python
import argparse as ap
import sys
import numpy as np
import pandas as pd
from pandas_plink import read_plink
import random
import warnings
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from magepro_simulations_functions import * # SEE HERE FOR ALL FUNCTION CALLS

args = sys.argv
set_num_causal = int(args[1]) #how many causal variants?
set_h2 = float(args[2])

effects = sim_effect_sizes_only(set_h2, set_num_causal)
effects = ','.join(map(str, effects))

print(effects)
