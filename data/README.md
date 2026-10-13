# External pipeline data

The contents of this directory are excluded from normal Git storage. Restore the
external bundle at the repository root so the paths below remain relative to
`data/`.

## Shared bundle

- `input/`: `raw_global_otu_dedup.tsv`, `merged_seqs.fasta`, and `labels.csv`.
- `sample_input/`: small, generated fixture for smoke-running the master scripts.
- `metadata/`: `metadata_clean_2.txt`.
- `references/gg2_data/`: Greengenes2 FASTA, taxonomy, and QIIME 2 artifacts.
- `references/silva_138/`: SILVA 138.2 FASTA and `translate_hash_silva.tsv`.
- `references/silva_132/`: SILVA 132 reference FASTA used by MGM.
- `references/GloVe_data/`: MDLM embedding FASTA and vocabulary array.
- `models/MDLM_Models/`: extracted MDLM ensemble and pretrained checkpoints.

The active MDLM and MGM scripts use the shared `references/GloVe_data/`,
`models/MDLM_Models/`, and `references/silva_132/` paths, so those assets can be
distributed with this directory. Do not flatten the bundle when extracting it.

## Completeness and limits

The shared inputs, reference databases, model checkpoints, and Conda YAML files
are present in this workspace, including `environments/deep_env.yml`. The
DeepMicro source is pinned as a Git submodule at the repository level, not part
of this data bundle. The bundle is approximately 2.8 GB before compression.
`Final_Thesis_version.pdf` is supplementary and is not a pipeline input.
Existing copies under `4-MDLM/data/` and `5-MGM/data/` are redundant for the
active scripts and do not need to be uploaded.

This is sufficient to start the downstream workflows from the supplied global
files in `input/`; it is not yet a complete reproduction of every workflow:

- `Data_transformation/1-single_study.py` can optionally import SILVA taxonomy
	into QIIME 2 from
	`references/silva_138/silva_138.2_ssu_nr99_taxonomy.tsv`; that file is absent.
	It is a tab-separated `Feature ID`/`Taxon` table (optionally with `Confidence`)
	for the SILVA sequence IDs. `translate_hash_silva.tsv` instead maps SILVA
	target IDs to project-specific unified IDs and is not a substitute. This gap
	affects SILVA taxonomy annotation in that transformation step, not runs
	starting from `input/`.
- Rebuilding the global input files from scratch also requires each source
	project's abundance table and FASTA. Those source project directories are not
	in this bundle; the supplied `input/` files are already-merged outputs.
- Conda YAMLs do not package Conda itself, system tools, GPU drivers, scheduler
	configuration, or the DeepMicro Git submodule. Those still need to be
	installed or initialized on the target machine.
- The `1-single_study_env` environment used to rebuild per-study inputs is not
	exported here. It is not needed for workflows that start from `input/`.

Unused DeepMicro example datasets, generated autoencoder outputs, a duplicate
SILVA 132 FASTA, unused SILVA and Greengenes2 QIIME artifacts, duplicate MDLM
ZIP archives, unused pretrained checkpoint weights, and macOS `.DS_Store`
metadata were removed. The extracted MDLM ensemble checkpoints and the
`epoch105` fallback config remain because the training and attribution scripts
read them directly.

## Sample fixture

From the repository root, run `python bin/build_master_sample.py` to derive
`sample_input/` from the full `input/` files. Use it with either master script
via `--input-dir data/sample_input`. The fixture is for wiring/smoke tests only;
it does not replace the full input bundle or the references required by MDLM and
MGM.

## Generated data

Pipeline outputs are written under the repository-level `output/` directory,
including `output/loco/`, `output/random/`, preprocessing results, split files,
model results, benchmark metrics, stability tables, and plots. They are not part
of this input bundle.
