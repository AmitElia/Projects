rm(list=ls()) #Clear environment

Sys.getenv("R_MAX_VSIZE")

library(Seurat)
library(dplyr)

#loading the data
patient1.data <- Read10X(data.dir = "/Users/amitelia/CSE185Proj/Patient1")
patient1 <- CreateSeuratObject(counts = patient1.data, project = "patient1", min.cells = 3, min.features = 50)
patient2.data <- Read10X(data.dir = "/Users/amitelia/CSE185Proj/Patient2")
patient2 <- CreateSeuratObject(counts = patient2.data, project = "patient2", min.cells = 3, min.features = 50)
patient3.data <- Read10X(data.dir = "/Users/amitelia/CSE185Proj/Patient3")
patient3 <- CreateSeuratObject(counts = patient3.data, project = "patient3", min.cells = 3, min.features = 50)
patient4.data <- Read10X(data.dir = "/Users/amitelia/CSE185Proj/Patient4")
patient4 <- CreateSeuratObject(counts = patient4.data, project = "patient4", min.cells = 3, min.features = 50)

#merging datasets
pbmc.combined <- merge(patient1, y = c(patient2, patient3, patient4), add.cell.ids = c("P1", "P2","P3", "P4"), project = "MergedAll")
pbmc <- pbmc.combined

#QC

#adding percent mitochondrial metadata.
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
#plotting distribution
VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
pbmc

#filtering
# & nFeature_RNA < 3000  & percent.mt < 5
pbmc <- subset(pbmc, subset = nCount_RNA > 500 & nCount_RNA <35000 & nFeature_RNA > 50 & nFeature_RNA < 6000 & percent.mt < 20)
pbmc
VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

#normalizing, finding most variable genes.
pbmc <- NormalizeData(pbmc)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)
# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(pbmc), 10)
top10

pbmc
LabelPoints(plot = VariableFeaturePlot(pbmc), points = top10, repel = TRUE)

#running PCA
all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))
print(pbmc[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(pbmc, dims = 1:2, reduction = "pca")
DimPlot(pbmc, reduction = "pca")
DimHeatmap(pbmc, dims = 1:15, cells = 500, balanced = TRUE)

#generating knn graph and clustering
pbmc <- FindNeighbors(pbmc, dims = 1:15, k.param = 15)
pbmc <- FindClusters(pbmc, resolution = 0.4)

head(Idents(pbmc), 5)

#running UMAP and plotting.
pbmc <- RunUMAP(pbmc, dims = 1:15)
DimPlot(pbmc, reduction = "umap", label = TRUE,label.box = TRUE, repel = TRUE)

library(SingleR)
library(celldex)
library(ggplot2)
#data set for cell types
hpca.se <- HumanPrimaryCellAtlasData()
hpca.se$label.main

#adding cell type label as metadata, then sets the cell's identities to the cell type.
results <- SingleR(test = as.SingleCellExperiment(pbmc), ref = hpca.se, assay.type.test =1, labels = hpca.se$label.main)
pbmc[["singler"]] <- CreateAssayObject(counts = results$labels)
Idents(object = pbmc) <- results$labels

pbmc[["old.ident"]] <- Idents(object = pbmc)
pbmc <- StashIdent(object = pbmc, save.name = "old.ident")

#UMAP plot with cell types as colors
DimPlot(pbmc, reduction = "umap", label = TRUE,label.box = TRUE, repel = TRUE, label.size = 3, group.by = "ident") + theme(legend.position="none")


#Six feature plots for selected genes
FeaturePlot(pbmc, features = c("ACE2", "KRT5", "CYP2A13", "TMPRSS2", "FOXJ1", "MUC5AC"), 
            pt.size = 0.005, 
            cols = c("yellow","forestgreen","darkgreen","darkslategrey", "midnightblue")) & 
  theme(legend.position="none",
        line = element_blank(), 
        axis.title = element_blank(), 
        axis.text = element_blank())
