
# RNA-seq Analysis of COVID-19-Related Anosmia

**This project is based on the paper "Non-neuronal expression of SARS-CoV-2 entry genes in the olfactory system suggests mechanisms underlying COVID-19-associated anosmia"** - [Link1]

Final Project for CSE185 by Professor Melissa Gymrek.

**Motivation**
The coronavirus encompasses a large class of viruses that are common in humans and animals. SARS CoV-2 is a beta coronavirus and causes a disease named Coronavirus Disease 2019 (COVID-19). SARS CoV-2 uses its spike proteins which are on the outside of the virus to bind to receptors and interact with proteases in human cells. A common receptor and protease that has been linked to COVID-19 is ACE2 and TMPRSS2 respectively. By interacting with ACE2 and TMPRSS2, the virus is able to hijack the machinery of the human cell and make copies of itself (H.A. Rothan). Upon exposure to the virus, some common symptoms are difficulty breathing, fever, chills, and even death. The paper focuses on one specific COVID-19 symptom called anosmia, which is the loss of smell.

**Methods**
The data was obtained from four human nasal tissue samples and was separated by either respiratory (Patients 1 and 4) or olfactory samples (Patients 2 and 3). 10x Genomics sequencing was then performed on each sample. This produced a matrix file, a features file containing the list of genes, and a barcodes file. All three files were used in our pipeline and analysis.

While performing single sequencing analysis on the data, I came across many obstacles with SPRING, the tool used in the paper. So, I switched to using Seurat to perform the same type of analysis on the same dataset. Ultimately, after performing the analysis, I came to the same conclusion as the findings in the paper, which was that ACE2 and TMPRSS2 were not expressed in the olfactory sensory neurons. I also found that ACE2 and TMPRSS2 were primarily expressed in nasal epithelial cells, consistent with the results acquired from the paper.(Fig2)

**Steps**
- Quality Control - Loading the data, Filtering.
- Principal Component Analysis(PCA) and Clustering
- Cluster Annotation by cell type.

**UMAP representation of cell types in human nasal biopsy (Brann et al. / Ours)**

| Brann et al | myself |
| ------ | ------ |
![](https://github.com/AmitElia/Projects/blob/main/Data%20Analysis/RNA-seq%20Analysis%20of%20COVID-19-Related%20Anosmia/plots/Screenshot%202024-12-17%20180916.png)  |  ![](https://github.com/AmitElia/Projects/blob/main/Data%20Analysis/RNA-seq%20Analysis%20of%20COVID-19-Related%20Anosmia/plots/Screenshot%202024-12-17%20182136.png)

The plot shows the cells colored by cell type labels from SingleR. In the plot, labeled cell types like respiratory HBC, olfactory HBC, respiratory ciliated will be a part of the Epithelial cells cluster. We can see a small cluster containing Neurons and neuroepithelial cells, which correspond to the olfactory sensory neurons. The goal is to see whether ACE2 and TMPRSS2 are expressed in these cells.

We can also see many other cell types that are shared with the paper’s plot (B cells, T cells, fibroblasts) and some that are not.


**UMAP representations of all cells, depicting the normalized expression of CoV-2 related genes ACE2 and TMPRSS2, as well as several cell type markers (Brann et al. / Ours)**
| Brann et al | myself |
| ------ | ------ |
![](https://github.com/AmitElia/Projects/blob/main/Data%20Analysis/RNA-seq%20Analysis%20of%20COVID-19-Related%20Anosmia/plots/Screenshot%202024-12-17%20180950.png)  |  ![](https://github.com/AmitElia/Projects/blob/main/Data%20Analysis/RNA-seq%20Analysis%20of%20COVID-19-Related%20Anosmia/plots/Screenshot%202024-12-17%20182149.png)

Both of the plots show which clusters express the following 6 genes: ACE2, KRT5, CYP2A13, TMPRSS2, FOXJ1, and MUC5AC. These genes were chosen because their expression levels can be used for the determination of major cell types. We can see that cells that express ACE2 and TMPRSS2 are epithelial cells. More importantly, ACE 2 and TMPRSS2 are not expressed in the OSN cluster containing the neuroepithelial cells, which agrees with the paper’s result.

The paper’s claim that ACE2 and TMPRSS2 are not expressed in the olfactory sensory neuron cells is supported by the plots, showing that these genes are expressed almost exclusively in epithelial cells.


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Link1]: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10715684/>
