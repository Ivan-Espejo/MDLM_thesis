# Data transformation

This file gives instructions on how to translate the OTU tables and fasta files from different projects into one single global directory that contains the files required for the execution of the MDLM. Steps:

1. Translation of individual sequences tags to generalized GG2 tags that are common for all the projects. (Done by Ays: `1-single_study.py`)
 - Generalization of the v3, v3-v4 and v4 into 16S databases.   


2. Merging OTU tables into one single big table. (`2-merge.py`)


3. Extracting the forward sequence of the v4 region of all the tags and generate new fasta and otu file. (`3-generate_fasta.py`) 
- Technical Filtering + Resolution Loss (Deduplication).
It uses cutadapt to hard-crop full-length sequences down to just the first 140bp of the V4 region.

4. Blast the GG2/SILVA   sequence tags to the embed tags. (`4-blast.sh`) 
- Vocabulary filtering:
Sequences are filtered following `seqs_.07_embed.fasta`. Vocabulary used by Pope et al. to generate embeddings.

5. Compressing OTU table into a numpy array that has the next shape. (`5-generate_512_npys.py`)
- Dimensionality Reduction (TSS/CLR).
- Mechanical Truncation:
For each individual sample, it sorts the transformed abundances from highest to lowest and extracts exactly the top 512 taxa.

6. Generating numpy array of labels from the metadata. (`6-generate_label_npy.py`)
- Sample Filtering (Metadata Matching).


| | 0 | 1 | 2 | 3 | 4 | … | 511 |
|---|---|---|---|---|---|---|---|
| **0** | [0.0, 6281.0] | [2.0, 937.0] | [65.0, 585.0] | [29.0, 198.0] | [750.0, 112.0] | … | [0.0, 0.0] |
| **1** | [286.0, 9652.0] | [65.0, 4894.0] | [188.0, 3793.0] | [2.0, 1861.0] | [3.0, 1499.0] | … | [0.0, 0.0] |
| **2** | [0.0, 5443.0] | [65.0, 1442.0] | [14.0, 1196.0] | [28.0, 1168.0] | [1065.0, 794.0] | … | [0.0, 0.0] |

* Where rows are ordered samples as in the OTU table and columns are features ordered in relative abundance for each sample. Being limited to 512 taxas. 
* Where [0.0, 6281.0] would mean that embed1 has 6281 rel abundance (TSS).


It is important to emphasize that in this process there is a loss of resolution in the shape of lost taxa. Sequences that are not present in the GG2/SILVA databases nor in the embeddings vocabulary, or sequences that are merged into less informative 140 bp V4 regions are not included in the final embed table.