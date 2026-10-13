# MDLM Thesis Repository

## Research Context

Inflammatory bowel disease (IBD), including Crohn's disease and ulcerative
colitis, is a chronic condition for which the gut microbiome may provide useful
predictive signals. However, cohorts differ in geography and laboratory
methods, so random train/test splits can make models appear more generalizable
than they are. This project harmonizes 1,175 samples from 14 cohorts and
compares classical classifiers, autoencoder embeddings, and transformer-based
models using leave-one-cohort-out validation. The central objective is to test
whether learned microbiome representations improve prediction in populations
not seen during training; random splits provide a complementary learning-curve
benchmark.

This repository contains the data transformation, model training, benchmarking,
and stability-analysis workflows for that experiment.

## Layout

- `Data_transformation/`: builds the shared OTU table, sequence table, and labels.
- `bin/`: creates LOCO or random splits and evaluates model results.
- `0-alpha_diversity/`, `1-mikropml/`, `2-DeepMicro/`, `4-MDLM/`, and `5-MGM/`:
	method-specific preprocessing and training pipelines.
- `data/`: external shared inputs, references, and model assets. See
	`data/README.md` for the required bundle.
- `output/`: generated splits, preprocessing results, model results, benchmarks,
	and plots. This directory is created by the workflows.

## Setup

Clone the repository with its pinned DeepMicro source and restore the external
data bundle without flattening it:

```bash
git clone --recurse-submodules https://github.com/Ivan-Espejo/MDLM_thesis.git
cd MDLM_thesis
git submodule update --init --recursive
```

Restore the external data bundle under this repository's `data/` directory
before running either master script. The privately shared bundle is available
from the [Google Drive data folder](https://drive.google.com/drive/folders/1MGJQrpwkntn_UFIeT3QpcxHz8qf8wiA3?usp=sharing).
Access is controlled by the Drive sharing permissions. Keep the folder structure
intact when downloading it.

The main inputs are `data/input/`; shared references and model assets are in
`data/references/` and `data/models/`. The master scripts also accept an
alternate input directory containing `labels.csv`, `raw_global_otu_dedup.tsv`,
and `merged_seqs.fasta`.

Create the Conda environments needed by the selected models; for example:

```bash
conda env create -f data/environments/mikropml.yml
conda env create -f data/environments/benchmark_env.yml
```

The batch scripts expect the cluster's Conda installation at `/opt/mambaforge`
and use the Slurm partitions, GPU types, and resource requests declared in their
`#SBATCH` headers. Adjust those settings only when targeting a different HPC
configuration.

## Sample inputs

`data/sample_input/` is included in Git for a small smoke test. It contains 80
samples from 8 balanced cohorts and 32 sequence-backed features. This approved
fixture is real data for checking pipeline wiring; it is not large or
representative enough for scientific performance conclusions.

## Usage

1. **Check the inputs.** From the repository root, validate the included sample
	fixture or the full Drive bundle before submitting jobs:

	```bash
	bash 01-master_loco.sh --input-dir data/sample_input --mode loco --validate-only
	bash 01-master_loco.sh --mode loco --validate-only
	```

2. **Generate splits and submit model jobs.** LOCO holds out complete cohorts;
	random mode creates stratified sample splits for learning curves. Select
	models with `--models`, or omit it to run all supported models:

	```bash
	bash 01-master_loco.sh --mode loco --runs 10 --models MDLM MGM
	bash 01-master_loco.sh --mode random --models MIKROPML
	```

	For a wiring smoke test, use the included fixture and a CPU model:

	```bash
	bash 01-master_loco.sh --input-dir data/sample_input --mode loco --runs 10 --models MIKROPML
	```

3. **Wait for the Slurm training jobs to finish**, then submit benchmarking for
	the same mode and input directory:

	```bash
	sbatch 02-master_benchmark.sh --mode loco
	sbatch 02-master_benchmark.sh --input-dir data/sample_input --mode loco
	```

	Benchmarking writes per-run metrics and figures under
	`output/<mode>/benchmark/`. LOCO also compares cohort presence and selected
	folds; random mode generates learning curves.

The master records input hashes with generated splits and rejects stale or
unmarked split folders. If it reports a provenance mismatch, remove only the
relevant generated directory (`output/loco/split_data_loco/` or
`output/random/split_data_random/`) and rerun the training master with the
intended input directory.

## Large files

Large sequence databases, QIIME 2 artifacts, checkpoints, NumPy arrays, HDF5
files, and generated tables are excluded from normal Git storage. Keep them in
the external bundle described by `data/README.md` or distribute them separately
using an artifact store or Git LFS.
