# Data transformation

This directory harmonizes project-level OTU tables and sequence files into the global inputs used by the downstream pipelines. The implemented workflow has two scripts:

1. `1-single_study.py` processes each project independently and writes one unified OTU table per project.
2. `2-merge.py` combines those project tables, validates them against metadata, and writes the global OTU table, reference FASTA, and binary labels.

The later normalization, embedding, truncation, and model-specific preprocessing steps are performed by scripts in the downstream method directories; they are not performed by these two scripts.

## 1. Process individual projects

`1-single_study.py` expects each project directory to contain an abundance table and a FASTA file. By default, these files are `ZOTUs-Table.tab` and `ZOTUs-Seqs.fasta`.

```bash
python 1-single_study.py \
	-p PRJNAXXX/ PRJNAYYY/ \
	--table ZOTUs-Table.tab \
	--seqs ZOTUs-Seqs.fasta \
	--outdir ../data/input
```

For each project, the script:

- reads the abundance table as tab- or comma-delimited data and keeps only features with sequence records;
- records kept and dropped input features in `logs_<project>/zotu_status.tsv`;
- imports the cleaned table and sequences into temporary QIIME 2 artifacts;
- maps sequences to the Greengenes2 backbone using QIIME 2 and then VSEARCH;
- searches sequences that did not receive a Greengenes2 mapping against SILVA;
- assigns a stable prefixed identifier for each mapped or unmapped feature: `GG2:<id>`, `SILVA:<id>`, `SILVATAX:<md5>`, or `NOVEL:<md5>`;
- combines rows that resolve to the same `UnifiedID` and retains the associated taxonomy when available.

The output for a project is `<project>_unified_tax.tsv`. It contains `UnifiedID`, one column per sample, and `Taxon`. Temporary QIIME 2 and BIOM files are removed from the project log directory after processing, while the other log files remain for inspection. A project with no matching input files is skipped; a project with no features that have sequences produces no unified table.

The mapping thresholds currently used by the script are 99.8% identity for the Greengenes2 QIIME 2 mapping, 99% for the VSEARCH Greengenes2 confirmation, and 99% for the VSEARCH SILVA search. A feature is not discarded merely because it fails both reference searches: it is retained as `NOVEL:<md5>`.

## 2. Merge projects and validate metadata

`2-merge.py` scans its input directory for every `*_unified_tax.tsv` file. The metadata argument is required by the command-line interface and must be a tab-separated file or a SQLite database. The default metadata columns are `sample_id` and `diagnosis`.

```bash
python 2-merge.py \
	--input_dir path/to/project_unified_tables \
	--outdir ../data/input \
	--meta ../data/metadata/metadata_clean_2.txt \
	--col_sample sample_id \
	--col_diag diagnosis
```

During the merge, the script:

- accumulates abundances by `UnifiedID` and carries forward non-`Unassigned` taxonomy;
- detects sample names that occur in multiple projects;
- merges repeated sample names when their project-specific abundance values are identical, and otherwise renames them to `<sample>_<project>`;
- checks both directions of the sample-to-metadata match;
- interactively removes OTU samples missing from metadata, or aborts if the user declines;
- interactively removes samples whose diagnosis is not included in `--ibd` or `--non_ibd`, or aborts if the user declines;
- writes label `1` for diagnoses in `--ibd` and label `0` for diagnoses in `--non_ibd`;
- writes a diagnostic log to `tmp/pipeline_drops.log`.

The default diagnosis groups are `IBD-UC`, `IBD-CD`, `Colorectal Cancer` for label 1 and `Healthy` for label 0. These lists can be overridden on the command line.


## Generated global files

The output directory contains:

- `raw_global_otu_dedup.tsv`: merged abundance table with `UnifiedID`, sample columns, and `Taxon`;
- `merged_seqs.fasta`: sequences extracted from the configured SILVA FASTA and Greengenes2 backbone for `SILVA:` and `GG2:` identifiers;
- `labels.csv`: generated when metadata validation produces retained labels, with `Sample_Name,Label` columns.

The FASTA extraction step does not reconstruct sequences for `NOVEL:` or `SILVATAX:` identifiers, so those features can remain in the OTU table without a corresponding record in `merged_seqs.fasta`. No cutadapt cropping, embedding-vocabulary filtering, TSS/CLR transformation, top-512 selection, or NumPy conversion occurs in `1-single_study.py` or `2-merge.py`.

## Reference files

`1-single_study.py` uses the Greengenes2 QIIME 2 artifacts and SILVA FASTA under `../data/references/`. `2-merge.py` defaults to the corresponding root `data/` paths. The SILVA taxonomy TSV path referenced by `1-single_study.py` must also be available if SILVA taxonomy import and taxonomy lookup are required. In the current data bundle, the available file is named `translate_hash_silva.tsv`, while the script expects `silva_138.2_ssu_nr99_taxonomy.tsv`; this filename or path mismatch must be resolved for SILVA taxonomy lookup to work.