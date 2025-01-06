# GWAS Example and wrapper functions

**Description**
Genome-wide association studies typically consist of performing a regression test for each variant to test it for association with a phenotype. 

For instance, we would like to test if a particular SNP (with alleles "A" and "T" in the population) is associated with height. We would look at a bunch of people (current GWASs for height are up to 5 million people!), and record the SNP genotype (AA=0, AT=1, or TT=2) and height for each person. Then, we'll just do a linear regression to test if there is a linear relationship between $X$ and $Y$.
