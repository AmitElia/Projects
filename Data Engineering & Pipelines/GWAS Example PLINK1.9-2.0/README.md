# GWAS Example and wrapper functions

**Description**

Genome-wide association studies typically consist of performing a regression test for each variant to test it for association with a phenotype. 
For instance, we would like to test if a particular SNP (with alleles "A" and "T" in the population) is associated with height. We would look at a bunch of people (current GWASs for height are up to 5 million people!), and record the SNP genotype (AA=0, AT=1, or TT=2) and height for each person. Then, we'll just do a linear regression to test if there is a linear relationship between $X$ and $Y$.

Phenotypes are assumed based on a set of input genotypes and an effect size $\beta$ using the simple linear model:

$$y_i = \beta x_i + \epsilon_i$$

where:
* $y_i$ is the phenotype of person $i$
* $x_i$ is the genotype (0, 1, or 2) of person $i$ for the SNP of interest
* $\beta$ is the effect size, which gives a measure of how strongly the SNP is associated with the trait. We will require this to be between -1 and 1.
* $\epsilon_i$ is a noise term. You can think of it as the part of the trait explained by non-genetic factors, such as environment or measurement error.
