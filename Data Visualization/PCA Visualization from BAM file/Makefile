WD = ~/work/PCA_Visualization


BFILE = ~/tutorials/GWASTutorial/01_Dataset/1KG.EAS.auto.snp.norm.nodup.split.rare002.common015.missing
THREAD = 2
PREFIX ?= out/plink_out
LD_REGION = high-LD-regions-hg19-GRCh37.txt

# PCA perperation: exclude SNPs in high-LD regions or HLA regions.
# need to get list of regions as .txt file
# high-LD-regions-hg38-GRCh38.txt hg38
# high-LD-regions-hg19-GRCh37.txt hg19

# extract SNPs based on high-LD regions
# For downstream analysis, we can exclude these SNPs using --exclude hild.set


hild:
	mkdir out
	plink \
    	--bfile ${BFILE} \
    	--make-set ${LD_REGION} \
    	--write-set \
    	--out ${PREFIX}.hild

# For LD-pruning and PCA, we usually only use variants with 
# MAF > 0.01 or MAF>0.05 ( --maf 0.01 or --maf 0.05) for robust estimation.
# LD-pruning, excluding high-LD and HLA regions
# --maf 0.01 : exlcude snps with maf<0.01
# --exclude : exclude snps in high-LD regions
# --indep-pairwise 50 5 0.2 : filters out all variants that are in linkage disequilibrium with each other.
# Remove related samples using king-cuttoff
filter: hild
	plink2 \
        --bfile ${BFILE} \
        --maf 0.01 \
        --threads ${THREAD} \
        --exclude ${PREFIX}.hild.set \
		--indep-pairwise 500 50 0.2 \
        --out ${PREFIX}
	plink2 \
        --bfile ${BFILE} \
        --extract ${PREFIX}.prune.in \
        --king-cutoff 0.0884 \
        --threads ${THREAD} \
        --out ${PREFIX}

# PCA after pruning and removing related samples
# note that approx flag is only recommended for analysis of >5000 samples.
# allele-wts 10 modifier requests an additional one-line-per-allele .eigenvec.allele file 
# with the first 10 PCs expressed as allele weights instead of sample weights.
# plink_results.eigenvec.allele file will be used to project onto all samples 
# along with an allele count plink_results.acount file
pca: filter
	plink2 \
        --bfile ${BFILE} \
        --keep ${PREFIX}.king.cutoff.in.id \
        --extract ${PREFIX}.prune.in \
        --freq counts \
        --threads ${THREAD} \
        --pca approx allele-wts 10 \
		--out ${PREFIX}

# Projection (related and unrelated samples)
# score ${outPrefix}.eigenvec.allele 2 5 sets the ID (2nd column) and A1 (5th column),
# score-col-nums 6-15 sets the first 10 PCs to be projected
# plink_results_projected.sscore - PCA results for all samples
	plink2 \
        --bfile ${BFILE} \
        --threads ${THREAD} \
        --read-freq ${PREFIX}.acount \
        --score ${PREFIX}.eigenvec.allele 2 6 header-read no-mean-imputation variance-standardize \
        --score-col-nums 7-16 \
		--out ${PREFIX}_projected


run: pca

clean:
	rm -rf out
	