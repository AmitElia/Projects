import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

pca = pd.read_table(os.path.join('/home/aelia/work/PCA_Visualization/out/plink_out_projected.sscore'),sep="\t")

ped = pd.read_table(os.path.join('/home/aelia/tutorials/GWASTutorial/01_Dataset/integrated_call_samples_v3.20130502.ALL.panel'),sep="\t")

pcaped=pd.merge(pca,ped,right_on="sample",left_on="IID",how="inner")


ax = sns.scatterplot(data=pcaped,x="PC1_AVG",y="PC2_AVG",hue="pop",s=50)
fig = ax.get_figure()
fig.savefig('/home/aelia/work/PCA_Visualization/out/PCA.png')