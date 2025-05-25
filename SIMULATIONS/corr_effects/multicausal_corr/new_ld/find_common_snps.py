import argparse
import pandas as pd

def read_bim_file(bim_file):
    """Read the .bim file as a pandas DataFrame and return the SNP column."""
    df = pd.read_csv(bim_file, delim_whitespace=True, header=None)
    return set(df[1])  # SNP IDs are in the second column (index 1)

def write_common_snps(common_snps, output_file):
    """Write the common SNPs to a file."""
    with open(output_file, 'w') as file:
        for snp in common_snps:
            file.write(f"{snp}\n")

def create_common_snps(eur_bim, afr_bim, amr_bim, output_file):
    """Create a list of common SNPs across three populations."""
    # Read SNP IDs from each population's .bim file
    eur_snps = read_bim_file(eur_bim)
    afr_snps = read_bim_file(afr_bim)
    amr_snps = read_bim_file(amr_bim)

    # Find the intersection of all three sets of SNPs
    common_snps = eur_snps & afr_snps & amr_snps
    
    # Write common SNPs to output file
    write_common_snps(common_snps, output_file)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Find common SNPs across three populations.")
    
    # Add arguments for the paths to the .bim files
    parser.add_argument('--eur_bim', type=str, required=True, help="Path to the EUR .bim file")
    parser.add_argument('--afr_bim', type=str, required=True, help="Path to the AFR .bim file")
    parser.add_argument('--amr_bim', type=str, required=True, help="Path to the AMR .bim file")
    parser.add_argument('--output_file', type=str, required=True, help="Output file for common SNPs")
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Step 1: Create the common SNPs file
    create_common_snps(args.eur_bim, args.afr_bim, args.amr_bim, args.output_file)

if __name__ == '__main__':
    main()

