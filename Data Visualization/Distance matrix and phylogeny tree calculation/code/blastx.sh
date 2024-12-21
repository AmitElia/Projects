set -uex

makeblastdb -in proteins.fasta -dbtype prot -out DB

blastx -query queries.fasta -db DB -out output.txt



