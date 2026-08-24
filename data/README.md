# External pipeline data

This directory is intentionally excluded from GitHub. Package its contents as an external archive and unzip the archive at the repository root so the paths below are preserved.

## Required layout and files

- `input/`: `raw_global_otu_dedup.tsv`, `merged_seqs.fasta`, and `labels.csv`
- `metadata/`: `metadata_clean_2.txt`
- `references/gg2_data/`: `2024.09.backbone.full-length.fna.qza`, `2024.09.backbone.tax.qza`, `2024.09.taxonomy.asv.nwk.qza`, and `gg2-2024.09.fasta`
- `references/silva_data/`: `silva-138.2-ssu-nr99-seqs.fasta`, `silva-138.2-ssu-nr99-seqs.qza`, and `translate_hash_silva.tsv`
- `references/silva132/`: `SILVA_132_SSURef_NR99_DNA.fasta`
- `references/GloVe_data/`: `seqs_.07_embed.fasta` and `vocab_embeddings.npy`
- `models/MDLM_Models/`: `ensemble.zip`, `pretrainedmodels.zip`, and their extracted checkpoint directories
- `models/`: `marker.zip`, `abundance.zip`, `AE[20]_global_input_otu.h5`, `AE[50]_.h5`, `global_input_otu.csv`, `global_input_samples.txt`, `AE[20]_global_input_otu_rep.csv`, and `AE[50]__rep.csv`

The large reference databases, FASTA files, QIIME 2 artifacts, model archives/checkpoints, NumPy arrays, HDF5 files, and generated global tables are not suitable for normal GitHub storage.

## Important

The data-transformation script expects `references/silva_data/silva_138.2_ssu_nr99_taxonomy.tsv` for SILVA taxonomy import. The current bundle contains `translate_hash_silva.tsv` instead, so provide the expected taxonomy TSV or adapt that mapping before running `1-single_study.py` with SILVA taxonomy classification.
