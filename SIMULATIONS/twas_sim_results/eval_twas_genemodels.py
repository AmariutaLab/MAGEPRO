import pandas as pd 
import numpy as np

path = "/expanse/lustre/projects/ddp412/sgolzari/MAGEPRO_REVIEWS/test_METRO/results/final_final_run_05_16.txt"

df = pd.read_csv(path, sep = '\t')


print("heritability = 0.01")
df_1_h2 = df[df['EQTL_H2'] == 0.01]
print(df_1_h2['LASSO_R2'].mean())
print(df_1_h2['PRSCSx_R2'].mean())
print(df_1_h2['MAGEPRO_R2'].mean())

print("heritability = 0.05")
df_5_h2 = df[df['EQTL_H2'] == 0.05]
print(df_5_h2['LASSO_R2'].mean())
print(df_5_h2['PRSCSx_R2'].mean())
print(df_5_h2['MAGEPRO_R2'].mean())

print("heritability = 0.1")
df_10_h2 = df[df['EQTL_H2'] == 0.1]
print(df_10_h2['LASSO_R2'].mean())
print(df_10_h2['PRSCSx_R2'].mean())
print(df_10_h2['MAGEPRO_R2'].mean())