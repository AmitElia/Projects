#
# A variant calling Makefile made by @AmitElia. inspired by biostar handbook.
#


currTime ?= 1
WD = ~/work/GWAS_Example

BFILE = ~/tutorials/GWASTutorial/01_Dataset/1KG.EAS.auto.snp.norm.nodup.split.rare002.common015.missing

QC:
# calculate sample missing rate (.imiss file), and SNP missing rate (.lmiss file)
	plink --bfile ${BFILE} --missing --out plink_out/missing_rate_log
	# python ${WD}/scripts/missing_histogram.py 
# calculate allele Frequencies
#Common variants : MAF>=0.05
#Low-frequency variants : 0.01<=MAF<0.05
#Rare variants : MAF<0.01
	plink --bfile ${BFILE} --freq --out plink_out/allele_freq_log
	# python ${WD}/scripts/allele_freq_histogram.py
	





# Clean up all results.
clean:
	rm -rf dat out
	
	
# Inform make that these targets are not files.
.PHONY: QC clean

