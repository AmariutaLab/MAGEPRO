import pandas as pd 
import numpy as np
import sys

args = sys.argv[1:]
print(args) # first arg = T/F for filter METRO nonconvergence

def calculate_power(df_input, num_null):
   
    df = df_input.copy()
    threshold = 0.05/(df.shape[0] + num_null)
    print(df.shape[0])
    #df.loc[ df["METRO_PVAL"].isna() , "METRO_PVAL"] = 1
    power_lasso = np.mean( df["LASSO_PVAL"] < threshold )
    power_prscsx = np.mean( df["PRSCSx_PVAL"] < threshold )
    power_magepro = np.mean( df["MAGEPRO_PVAL"] < threshold )
    power_metro = np.mean( df["METRO_PVAL"] < threshold )
    return power_lasso, power_prscsx, power_magepro, power_metro

def compute_accuracy_mean_sem(r2, h2):
    r2 = np.array(r2, dtype=float)
    h2 = np.array(h2, dtype=float)

    # Mask for valid entries: finite r2 and h2
    valid_mask = np.isfinite(r2) & np.isfinite(h2)
    r2 = r2[valid_mask]
    h2 = h2[valid_mask]

    # Prevent negative values
    r2[r2 < 0] = 0
    h2[h2 < 0] = 0

    # Avoid division by zero; only compute valid indices
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = r2 / h2
        acc = acc[np.isfinite(acc)]  # remove NaN and Inf
        acc[acc > 1] = 1  # cap at 1

    mean_acc = np.mean(acc)
    sem_acc = np.std(acc, ddof=1) / np.sqrt(len(acc))

    return mean_acc, sem_acc, acc

def compute_r2_mean_sem(r2):
    r2 = np.asarray(r2)
    mean_r2 = np.nanmean(r2)
    sem_r2 = np.nanstd(r2, ddof=1) / np.sqrt(np.sum(~np.isnan(r2)))
    return mean_r2, sem_r2

def calculate_accuracy(df):
    
    #print(df[df['METRO_PVAL'].isna()])

    accuracy_lasso, sem_lasso, acc_lasso = compute_accuracy_mean_sem(df.LASSO_VAL_R2.values, df.TARGET_HSQ.values)
    accuracy_prscsx, sem_prscsx, acc_prscsx = compute_accuracy_mean_sem(df.PRSCSx_VAL_R2.values, df.TARGET_HSQ.values)
    accuracy_magepro, sem_magepro, acc_magepro = compute_accuracy_mean_sem(df.MAGEPRO_VAL_R2.values, df.TARGET_HSQ.values)
    accuracy_metro, sem_metro, acc_metro = compute_accuracy_mean_sem(df.METRO_VAL_R2.values, df.TARGET_HSQ.values)
    
    return accuracy_lasso, accuracy_prscsx, accuracy_magepro, accuracy_metro, sem_lasso, sem_prscsx, sem_magepro, sem_metro

def summarize_r2(df):

    r2_lasso, sem_r2_lasso = compute_r2_mean_sem(df.LASSO_VAL_R2.values)
    r2_prscsx, sem_r2_prscsx = compute_r2_mean_sem(df.PRSCSx_VAL_R2.values)
    r2_magepro, sem_r2_magepro = compute_r2_mean_sem(df.MAGEPRO_VAL_R2.values)
    r2_metro, sem_r2_metro = compute_r2_mean_sem(df.METRO_VAL_R2.values)

    return r2_lasso, r2_prscsx, r2_magepro, r2_metro, sem_r2_lasso, sem_r2_prscsx, sem_r2_magepro, sem_r2_metro

#path = "/expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/test_METRO/results/final_final_run_06_23.txt"
#path = "/expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/test_METRO/results/clean_final_final_run_06_23.txt"
path = "/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO_TWAS_SIMULATIONS/results/clean_validation_result_08_10.txt"

df = pd.read_csv(path, sep = '\t')

print(df.head())

# --- script from stephen to clean results - remove when METRO returns NaN 
#df_metro_noconverge = df[df['METRO_PVAL'].isna()]
#df_metro_noconverge_counts = df_metro_noconverge.groupby(['EQTL_H2', 'H2GE', 'CORRELATION', 'NUM_GWAS']).size().reset_index(name='count')
#df_metro_noconverge_counts.to_csv("/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/SIMULATIONS/twas_sim_results/twas_metro_noconverge_counts_06_27.txt", sep = '\t', header = True, index = False)


if args[0] == 'True' or args[0] == True:
    df = df[~df['METRO_PVAL'].isna()] # removes NaNs

df = df.sort_values(by=['GENE', 'EQTL_H2', 'H2GE', 'CORRELATION', 'NUM_GWAS']).reset_index(drop=True)
# ---

# --- calculate MAGEPRO LASSO PRSCSx accuracy and SEM 
#boolmask = (df["EQTL_H2"] == 0.01) & (df["CORRELATION"] == 0.8)
#df_interest = df[boolmask]
#accuracy_lasso, accuracy_prscsx, accuracy_magepro, accuracy_metro, sem_lasso, sem_prscsx, sem_magepro, sem_metro = calculate_accuracy(df_interest)
#print('lasso accuracy: ', accuracy_lasso)
#print('prscsx accuracy: ', accuracy_prscsx)
#print('magepro accuracy: ', accuracy_magepro)
#print(accuracy_metro)
# ---

EQTL_H2_ARR = [0.01, 0.05, 0.1] # True heritability value
PHEN_VAR_GENE_COMPONENT = [0, 0.0001, 0.001, 0.01]
CORRELATIONS_GE = [0.2, 0.8]
GWAS_SAMPLE_SIZE = [20000, 100000, 600000]

columns = ["NUM_GENES", "H2GE", "EQTL_H2", "CORRELATION", "NUM_GWAS", "LASSO_POWER", "PRSCSx_POWER", "MAGEPRO_POWER", "METRO_POWER", "LASSO_ACCURACY", "PRSCSx_ACCURACY", "MAGEPRO_ACCURACY", "METRO_ACCURACY", "LASSO_SEM", "PRSCSx_SEM", "MAGEPRO_SEM", "METRO_SEM", "LASSO_R2", "PRSCSx_R2", "MAGEPRO_R2", "METRO_R2", "LASSO_R2_SEM", "PRSCSx_R2_SEM", "MAGEPRO_R2_SEM", "METRO_R2_SEM"]
df_power = pd.DataFrame(columns=columns)

for H2GE in PHEN_VAR_GENE_COMPONENT:
    for EQTL_H2 in EQTL_H2_ARR:
        for CORRELATION in CORRELATIONS_GE:
            for NUM_GWAS in GWAS_SAMPLE_SIZE:
                boolmask = (df["H2GE"] == H2GE) & (df["EQTL_H2"] == EQTL_H2) & (df["CORRELATION"] == CORRELATION) & (df["NUM_GWAS"] == NUM_GWAS)
                df_current = df[boolmask]
                accuracy_lasso, accuracy_prscsx, accuracy_magepro, accuracy_metro, sem_lasso, sem_prscsx, sem_magepro, sem_metro = calculate_accuracy(df_current)
                r2_lasso, r2_prscsx, r2_magepro, r2_metro, sem_r2_lasso, sem_r2_prscsx, sem_r2_magepro, sem_r2_metro = summarize_r2(df_current)
                power_lasso, power_prscsx, power_magepro, power_metro = calculate_power(df_current, 1000)
                new_row = pd.DataFrame([{"NUM_GENES": df_current.shape[0], "H2GE": H2GE, "EQTL_H2": EQTL_H2, "CORRELATION": CORRELATION, "NUM_GWAS": NUM_GWAS, 
                    "LASSO_POWER": power_lasso, "PRSCSx_POWER": power_prscsx, "MAGEPRO_POWER": power_magepro, "METRO_POWER": power_metro, 
                    "LASSO_ACCURACY": accuracy_lasso, "PRSCSx_ACCURACY": accuracy_prscsx, "MAGEPRO_ACCURACY": accuracy_magepro, "METRO_ACCURACY": accuracy_metro, 
                    "LASSO_SEM": sem_lasso, "PRSCSx_SEM": sem_prscsx, "MAGEPRO_SEM": sem_magepro, "METRO_SEM": sem_metro, 
                    "LASSO_R2": r2_lasso, "PRSCSx_R2": r2_prscsx, "MAGEPRO_R2": r2_magepro, "METRO_R2": r2_metro,
                    "LASSO_R2_SEM": sem_r2_lasso, "PRSCSx_R2_SEM": sem_r2_prscsx, "MAGEPRO_R2_SEM": sem_r2_magepro, "METRO_R2_SEM": sem_r2_metro}])
                df_power = pd.concat([df_power, new_row], ignore_index=True)

df_power.to_csv("/expanse/lustre/projects/ddp412/kakamatsu/MAGEPRO/SIMULATIONS/twas_sim_results/twas_results_08_10.txt", sep = '\t', header = True, index = False)
