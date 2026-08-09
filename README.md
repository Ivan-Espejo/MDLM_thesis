# MDLM Thesis Repository

This repository contains the code and workflow used for the MDLM thesis project, including the preprocessing, benchmarking, and model-training pipelines for the different methods in this study.

## Repository structure

- 0-Data_transformation/: code for harmonizing individual cohort datasets into a single OTU table and label file for downstream pipeline input.
- 01-*.py and 01-master_*.sh: initial data-splitting and orchestration scripts.
- 1-mikropml/, 2-DeepMicro/, 4-MDLM/, and 5-MGM/: method-specific pipelines and scripts.
- 4-MDLM/microbiome_transformers/: transformer-based training code and utilities.

## Important note about large assets

This repository intentionally does not include large data files, pretrained model weights, or reference databases. Those assets are expected to be downloaded or generated separately before running the workflows.

The following paths are excluded from version control:

- 0-Data_transformation/.Database/
- 0-Data_transformation/.output/
- 0-Data_transformation/User_input/
- 4-MDLM/data/GloVe_data/
- 4-MDLM/data/MDLM_Models/
- 5-MGM/data/silva132/
- large model weight files such as .bin, .pt, .pth, .ckpt, .npy, .fasta, and .zip artifacts

## Getting started

1. Clone the repository.
2. Create a Python environment and install the required dependencies for the relevant workflow.
3. Download or generate the required datasets and model files outside of version control.
4. Run the scripts from the appropriate module directory.

## Notes

If you want to keep large files in the repository, consider using Git LFS or storing them in a separate release artifact or external data repository.
