
# RNA-seq Analysis of COVID-19-Related Anosmia

**This project is based on the paper "Non-neuronal expression of SARS-CoV-2 entry genes in the olfactory system suggests mechanisms underlying COVID-19-associated anosmia"** - [Link1]

Final Project for CSE185 by Professor Melissa Gymrek.

**Motivation**
The coronavirus encompasses a large class of viruses that are common in humans and animals. SARS CoV-2 is a beta coronavirus and causes a disease named Coronavirus Disease 2019 (COVID-19). SARS CoV-2 uses its spike proteins which are on the outside of the virus to bind to receptors and interact with proteases in human cells. A common receptor and protease that has been linked to COVID-19 is ACE2 and TMPRSS2 respectively. By interacting with ACE2 and TMPRSS2, the virus is able to hijack the machinery of the human cell and make copies of itself (H.A. Rothan). Upon exposure to the virus, some common symptoms are difficulty breathing, fever, chills, and even death. The paper focuses on one specific COVID-19 symptom called anosmia, which is the loss of smell.

**Methods**
The data was obtained from four human nasal tissue samples and was separated by either respiratory (Patients 1 and 4) or olfactory samples (Patients 2 and 3). 10x Genomics sequencing was then performed on each sample. This produced a matrix file, a features file containing the list of genes, and a barcodes file. All three files were used in our pipeline and analysis.

While performing single sequencing analysis on the data, I came across many obstacles with SPRING, the tool used in the paper. So, I switched to using Seurat to perform the same type of analysis on the same dataset. Ultimately, after performing our analysis, I came to the same conclusion as the findings in the paper, which was that ACE2 and TMPRSS2 were not expressed in the olfactory sensory neurons. I also found that ACE2 and TMPRSS2 were primarily expressed in nasal epithelial cells, consistent with the results acquired from the paper.(Fig2)

**Steps**
- Quality Control - Loading the data, Filtering.
- Principal Component Analysis(PCA) and Clustering
- Cluster Annotation by cell type.

| Brann et al | myself |
| ------ | ------ |
![](outputs/output_Biotouch/18-15_02-02-2018/Identification/ITALIC/ITALIC_movementPoints_cmc.png)  |  ![](outputs/output_Biotouch/18-15_02-02-2018/Identification/BLOCK_LETTERS/BLOCK_LETTERS_movementPoints_cmc.png)





[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Link1]: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10715684/>
