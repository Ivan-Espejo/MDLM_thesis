# MDLM Thesis Repository

This repository contains the data transformation, model training, benchmarking,
and stability-analysis workflows used in the MDLM thesis project.

## Layout

- `Data_transformation/`: builds the shared OTU table, sequence table, and labels.
- `bin/`: creates LOCO or random splits and evaluates model results.
- `0-alpha_diversity/`, `1-mikropml/`, `2-DeepMicro/`, `4-MDLM/`, and `5-MGM/`:
	method-specific preprocessing and training pipelines.
- `4-MDLM/microbiome_transformers/`: transformer implementation used by MDLM.
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
before running either master script.

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
configuration. The `1-single_study_env` used for rebuilding per-study inputs is
not exported in `data/environments/`; it is not needed when starting from the
supplied global input files.

## Main workflows

Generate splits and run selected models:

```bash
bash 01-master_loco.sh --mode loco --runs 10
bash 01-master_loco.sh --mode loco --models MDLM MGM
bash 01-master_loco.sh --mode random --models MIKROPML
```

After model jobs finish, submit benchmarking and aggregation as a Slurm job:

```bash
sbatch 02-master_benchmark.sh --mode loco
sbatch 02-master_benchmark.sh --mode random
```

## Sample inputs

Build a small fixture from the bundled global inputs, then run a CPU-only
smoke workflow from the repository root:

```bash
python bin/build_master_sample.py
bash 01-master_loco.sh --input-dir data/sample_input --mode loco --runs 10 --models MIKROPML
sbatch 02-master_benchmark.sh --input-dir data/sample_input --mode loco
```

The fixture has 80 samples from 8 balanced cohorts and 32 sequence-backed
features. It is intended to verify input wiring and orchestration, not to
produce scientifically meaningful performance estimates. Run it in a clean
workspace/output directory so its results are not mixed with full-data runs.
Use `--validate-only` with either master to check the selected input directory
without creating splits or submitting jobs. Existing split folders are reused
only when their recorded input hash and split settings match; legacy folders
without provenance must be removed before regenerating them.

LOCO benchmarking also runs the cohort-presence analysis for `PRJNA1259137` and
compares the configured best and worst folds (`run4` and `run10`). Random mode
generates learning-curve outputs instead. `01-master_loco.sh` is an orchestrator
run from the login node; it performs global preprocessing when requested and
submits model jobs. `02-master_benchmark.sh` is submitted with `sbatch`.

## Large files

Large sequence databases, QIIME 2 artifacts, checkpoints, NumPy arrays, HDF5
files, and generated tables are excluded from normal Git storage. Keep them in
the external bundle described by `data/README.md` or distribute them separately
using an artifact store or Git LFS.
