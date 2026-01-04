two changes made for consistency checks: 

1. using LD matrix from Sim_geno.py instead of recomputing with plink
2. explicit MAF > 0.01 filter in each population, subset to SNPs in common  

I wanted to make sure simulation performances are not sensitive to these small changes 
