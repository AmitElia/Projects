rm(list = ls()) #clear environment

wdir = "/Users/eliaa/Documents/BENG182/project/"

library(ape)
library(diversitree)
#library(phangorn)
#library(ade4)
#library(adegenet)
#library(stats)
#library(ggplot2)


dat = read.csv(paste(wdir, "spike_2.csv", sep = ""), sep = ",", row.names = 1)
dat = dat[-1,]

dat_mat = as.matrix(dat)

D <- dist.gene(dat_mat)
length(D)

temp <- as.data.frame(as.matrix(D))
table.paint(temp, cleg=0, clabel.row=.5, clabel.col=.5)

stree = nj(D)

plot(stree, cex=0.6)

colorvec = c("lightblue", "blue")

lst = ls



trait.plot(stree, dat, cols= lst, type="p")




col_names = as.vector(names(dat))

h_cluster <- hclust(D,  method = "average", members = NULL)

plot(h_cluster, cex = 0.6)



View(dat)
