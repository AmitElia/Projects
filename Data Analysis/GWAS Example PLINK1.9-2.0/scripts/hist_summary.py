import pandas as pd
import matplotlib.pyplot as plt
import os.path

dfi = pd.read_csv(os.path.join('/home/aelia/work/GWAS_Example/out/plink_out.imiss.tsv'), sep='\t', header=0)
dfj = pd.read_csv(os.path.join('/home/aelia/work/GWAS_Example/out/plink_out.lmiss.tsv'), sep='\t', header=0)
maf = pd.read_csv(os.path.join('/home/aelia/work/GWAS_Example/out/plink_out.frq.tsv'), sep='\t', header=0)
hwe = pd.read_csv(os.path.join('/home/aelia/work/GWAS_Example/out/plink_out.hwe.tsv'), sep='\t', header=0)

fig = plt.figure()
fig.set_size_inches((16, 9))


ax = fig.add_subplot(211)
ax.hist(dfi['F_MISS'], bins=30, color='blue', alpha=0.5)
ax.set_title('sample missing rate')
ax = fig.add_subplot(212)
ax.hist(dfj['F_MISS'], bins=30, color='red', alpha=0.5)
ax.set_title('SNP missing rate')
ax = fig.add_subplot(221)
ax.hist(maf['MAF'], bins=30, color='green', alpha=0.5)
ax.set_title('Minor Allele Frequency')
ax = fig.add_subplot(222)
ax.hist(hwe['P'], bins=30, color='purple', alpha=0.5)
ax.set_title('Hardy-Weinberg equilibrium exact test')
fig.savefig(os.path.join('/home/aelia/work/GWAS_Example/out/histogram_Summary.png'))
