# Distance matrix and phylogeny tree calculation

**Motivation**

queries we selected were all from Illinois, and their collection dates evenly spanned the whole pandemic, from February 2020 to May 2022, with three to four queries from each month. 
These queries were chosen because I was interested in seeing how the virus’s mutations accumulated over time in a specific region.

**Steps**

- Acquired files: SARS_CoV2_full_genome.fasta (Full Genome), gene_annotation.gff (gene annotations). (Reference Sequence: NC_045512.2)
- Used the gene annotations file to extract the name, start, and end indices of each mature protein. Examples of these proteins are nsp2, nsp3, helicase, RNA-dependent RNA polymerase, and more. Combined these indices with those of the classified proteins for a total of 37 separate sequences.
- Parse protein sequences from the full genome file using the gene annotations. Used the ‘Biopython’ and ‘gffutils’ python libraries to parse each sequence from the full genome and convert it into amino-acid sequence in the correct frame.
- Used gffutils to read the .gff file into a list of features and used the .sequence() method in .gffutils to extract the sequence of that feature.
- Translated the sequence into amino-acid sequence using the Seq library inside Biopython.
- Chose blastx for its ability to align nucleotide sequences with a protein database. Used the stand-alone version of blast+ to run on a custom protein dataset. To create the BLAST protein database was required to convert the .fasta file into a BLAST formatted database with the command: “ >makeblastdb -in proteins.fasta -dbtype prot -out DB “.
- I ran blastx with the following command on our query file: “ blastx -query queries.fasta -db DB -out BLAST_output.txt ”.
- Created two main forms of dataframes, one as a binary mutation matrix, which enabled me to create phylogenetic trees of the queries, and the other as an amino acid mutation matrix, which stored which amino acid mutated to what other amino acid and at what position. The second matrix was helpful in identifying key mutations, as well as plotting mutations across time.
- Used ‘Ape’, an R library to calculate distance matrix from binary SNP matrix. Cells in the matrix are either 0 or 1, with 1 indicating a base-pair different from that of the reference.
- Used two types of rooted trees to represent the data, Neighbor-joining (NJ) and Unweighted Pair Group Method with Arithmetic mean (UPGMA). Focused on Surface Glycoprotein (Spike protein).
- UPGMA and Neighbor-Joining trees are two types of hierarchical clustering algorithms that use distance-based calculations to construct the tree. These two methods can quickly make trees but are not very reliable, in other words, they can give us a good idea about the data but can not be taken as the solution to a phylogeny, since the tree outcome depends on the algorithm. Maximum parsimony on the other hand seeks to minimize the number of mutations needed in order to make an optimal tree, but doesnt take convergence or homoplasy into account. Maximum-likelihood is the most ideal for building a phylogeny using sequence data, however it is the most computationally expensive when dealing with large datasets. After looking for ways to visualize the data, I only found a way to run NJ and UPGMA through ‘Ape’ and ‘ade4’ to produce trees with no labels for SNP loci. The next steps in visualizing the trees will be to add loci labels and to use a genome-wide SNP table instead of a specific protein.




**Challenges**

- I ran into several issues with this process. Firstly, some alignments were split into multiple chunks for one protein. This meant we had to concatenate these chunks together into one alignment. Additionally, not all queries had alignments against all the proteins in my custom database. That meant I had to store positional information about each protein in each query, and sort them out afterwards.
- Insertions in the reference genome do not count towards incrementing genomic position in the reference, it took me a while to find out this problem, so for a long time I was dealing with reference genomes that were slightly shifted between alignment to alignment. An example of this is that mutations at position 452 would show up as mutations in position 449. Once I sorted out these problems, I was able to move on to changing the shape of the data into forms that were more simple to analyze.

- **Visualizations**

*plots of the mutation rate by genomic position across the entire COVID reference genome.*

![](https://github.com/AmitElia/Projects/blob/main/Data%20Analysis/Distance%20matrix%20and%20phylogeny%20tree%20calculation/plots/Screenshot%202024-12-19%20152735.png)
From the figure we can identify key positions that are highly variable. Additionally, the spike protein has many positions with small mutation frequencies. I chose to focus the analysis on the spike protein and nsp3, highlighted in the figure above.

