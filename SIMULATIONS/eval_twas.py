import pandas as pd 
import numpy as np

def calculate_power(df):
    threshold = 0.05/df.shape[0]
    df.loc[ df["METRO_PVAL"].isna() , "METRO_PVAL"] = 1
    power_lasso = np.mean( df["LASSO_PVAL"] < threshold )
    power_magepro = np.mean( df["MAGEPRO_PVAL"] < threshold )
    power_metro = np.mean( df["METRO_PVAL"] < threshold )
    return power_lasso, power_magepro, power_metro

path = "/expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/test_METRO/results/results_03_28.txt"

df = pd.read_csv(path, sep = '\t')

EQTL_H2_ARR = [0.01, 0.05, 0.1] # True heritability value
PHEN_VAR_GENE_COMPONENT = [0, 0.0001, 0.001, 0.01]
CORRELATIONS_GE = [0.2, 0.8]
GWAS_SAMPLE_SIZE = [20000, 100000, 600000]

columns = ["H2GE", "EQTL_H2", "CORRELATION", "NUM_GWAS", "LASSO_POWER", "MAGEPRO_POWER", "METRO_POWER"]
df_power = pd.DataFrame(columns=columns)

for H2GE in PHEN_VAR_GENE_COMPONENT:
    for EQTL_H2 in EQTL_H2_ARR:
        for CORRELATION in CORRELATIONS_GE:
            for NUM_GWAS in GWAS_SAMPLE_SIZE:
                boolmask = (df["H2GE"] == H2GE) & (df["EQTL_H2"] == EQTL_H2) & (df["CORRELATION"] == CORRELATION) & (df["NUM_GWAS"] == NUM_GWAS)
                df_current = df[boolmask]
                power_lasso, power_magepro, power_metro = calculate_power(df_current)
                new_row = pd.DataFrame([{"H2GE": H2GE, "EQTL_H2": EQTL_H2, "CORRELATION": CORRELATION, "NUM_GWAS": NUM_GWAS, "LASSO_POWER": power_lasso, "MAGEPRO_POWER": power_magepro, "METRO_POWER": power_metro}])
                df_power = pd.concat([df_power, new_row], ignore_index=True)

df_power.to_csv("/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/SIMULATIONS/twas_sim_results/twas_results_03_28.txt", sep = '\t', header = True, index = False)