import pandas as pd
import matplotlib.pyplot as plt

dfi = pd.read_csv('~/work/GWAS_Example/plink_out/missing_rate_log.imiss',sep='\t', header=0)
dfj = pd.read_csv('~/work/GWAS_Example/plink_out/missing_rate_log.lmiss',sep='\t', header=0)

dfi
fig = plt.figure()
fig.set_size_inches((8, 3))

ax = fig.add_subplot(121)
ax.hist(dfi['F_MISS'], bins=30, color='blue', alpha=0.5)
ax.set_title('sample missing rate')
ax = fig.add_subplot(122)
ax.hist(dfj['F_MISS'], bins=30, color='red', alpha=0.5)
ax[2].set_title('SNP missing rate')
fig.savefig('~/work/GWAS_Example/plink_out/missing_rate_histogram.png')

