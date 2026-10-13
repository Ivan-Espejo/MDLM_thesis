#!/usr/bin/env Rscript

# =============================================================================
# Script: preprocessing.R
# Author: Ivan Espejo
#
# Description:
#     Converts the global OTU TSV into a total-sum-scaled sample-by-feature CSV
#     for DeepMicro and records the sample order in a sidecar text file.
#
# Inputs:
#     --input_otu: global OTU table in TSV format.
#     --output: destination for the numeric CSV matrix.
#
# Outputs:
#     The CSV matrix at --output and global_input_samples.txt beside it.
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(optparse)

options(width = as.numeric(system("tput cols", intern = TRUE)))

option_list <- list(
  make_option(c("--input_otu"), type = "character", default = "../data/input/raw_global_otu_dedup.tsv",
              help = "Path to the Global OTU table (TSV format)"),
  make_option(c("-o", "--output"), type = "character", default = "DeepMicro/data/global_input_otu.csv",
              help = "Path for the output DeepMicro CSV")
)
opt <- parse_args(OptionParser(option_list = option_list))

# 1. Load Data
cat(sprintf("Loading Global OTU table: %s\n", opt$input_otu))
otu_raw <- read_tsv(opt$input_otu, show_col_types = FALSE) %>% mutate(UnifiedID = as.character(UnifiedID))

# 2. Normalization (TSS) & Alignment
cat("Normalizing (TSS) and structuring feature space...\n")
aligned_otu <- otu_raw |>
  select(-any_of(c("Taxon", "taxonomy", "Confidence"))) |>
  mutate(across(where(is.numeric), ~ .x / sum(.x))) |>
  pivot_longer(cols = -UnifiedID, names_to = "Sample_Name", values_to = "count") |>
  pivot_wider(names_from = UnifiedID, values_from = count, values_fill = 0)

# Extract the Sample Names in exact order
sample_names <- aligned_otu$Sample_Name

# Create the pure numeric matrix required by DM.py
final_matrix <- aligned_otu |>
  select(-Sample_Name) |>
  as.matrix()

# ==========================================
# EXPORT TO DEEPMICRO FORMAT
# ==========================================
output_dir <- dirname(opt$output)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. Save the pure numeric matrix
write.table(final_matrix, file = opt$output, sep = ",", col.names = FALSE, row.names = FALSE)

# 2. Save the ordered sample names to a sidecar text file
writeLines(sample_names, con = file.path(output_dir, "global_input_samples.txt"))

cat("=======================================================\n")
cat("SUCCESS: Transformed and normalized global matrix for DeepMicro.\n")
cat(sprintf("Dimensions: %d samples (rows) x %d microbes (columns).\n", nrow(final_matrix), ncol(final_matrix)))
cat("=======================================================\n")