#
# A variant calling Makefile made by @AmitElia. inspired by biostar handbook.
#


currTime ?= 1

WD = ~/work/GWAS_Example


BFILE = ~/tutorials/GWASTutorial/01_Dataset/1KG.EAS.auto.snp.norm.nodup.split.rare002.common015.missing

QC:
	mkdir -p out
	mkdir -p out/plink_out
# calculate sample missing rate (.imiss file), and SNP missing rate (.lmiss file)
	plink --bfile ${BFILE} --missing --out out/plink_out
	# python ${WD}/scripts/missing_histogram.py 

# calculate allele Frequencies
#Common variants : MAF>=0.05
#Low-frequency variants : 0.01<=MAF<0.05
#Rare variants : MAF<0.01
	plink --bfile ${BFILE} --freq --out out/plink_out
	# python ${WD}/scripts/allele_freq_histogram.py

# calculate Hardy-Weinberg Equilibrium (HWE)
# HWE is a test of whether the observed genotype 
# frequencies in a population are consistent with 
# the frequencies expected under Hardy-Weinberg equilibrium.
	plink --bfile ${BFILE} --hardy --out out/plink_out
	# python ${WD}/scripts/hwe_histogram.py

# Filtering
# --maf 0.01 : exlcude snps with maf<0.01
# --geno 0.02 :filters out all variants with missing rates exceeding 0.02
# --mind 0.02 :filters out all samples with missing rates exceeding 0.02
# --hwe 1e-6 : filters out all variants which have HWE p-value below the threshold.
# --indep-pairwise 50 5 0.2 : filters out all variants that are in linkage disequilibrium with each other.
	plink \
	    --bfile ${BFILE} \
	    --maf 0.01 \
    	--geno 0.02 \
    	--mind 0.02 \
    	--hwe 1e-6 \
    	--indep-pairwise 50 5 0.2 \
    	--out out/plink_out

# Inbreeding Coefficient F
# High F may indicate a relatively high level of inbreeding.
# Low F may suggest the sample DNA was contaminated.
	plink \
	    --bfile ${BFILE} \
	    --extract out/plink_out.prune.in \
	    --het \
	    --out out/plink_out









# Clean up all results.
clean:
	rm -rf out
	
	
# Inform make that these targets are not files.
.PHONY: QC clean

