#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(tidyr)
library(optparse)

# Dynamically set width to match terminal window
options(width = as.numeric(system("tput cols", intern = TRUE)))

# ==========================================
# COMMAND LINE ARGUMENT PARSING
# ==========================================
option_list <- list(
  make_option(c("--train_otu"), type = "character", default = NULL,
              help = "Path to the Training OTU table (TSV format)", metavar = "character"),
  make_option(c("--test_otu"), type = "character", default = NULL,
              help = "Path to the Testing OTU table (TSV format)", metavar = "character"),
  make_option(c("-o", "--output"), type = "character", default = "DeepMicro/data/aligned_lopo_otu.csv",
              help = "Path for the output DeepMicro CSV [default= %default]", metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$train_otu) | is.null(opt$test_otu)) {
  print_help(opt_parser)
  stop("Critical Error: You must provide both --train_otu and --test_otu.", call. = FALSE)
}

# ==========================================
# DATA LOADING
# ==========================================
cat(sprintf("Loading Train OTU table: %s\n", opt$train_otu))
train_otu_raw <- read_tsv(opt$train_otu, show_col_types = FALSE) %>% mutate(UnifiedID = as.character(UnifiedID))

cat(sprintf("Loading Test OTU table: %s\n", opt$test_otu))
test_otu_raw <- read_tsv(opt$test_otu, show_col_types = FALSE) %>% mutate(UnifiedID = as.character(UnifiedID))

# ==========================================
# NORMALIZATION & ALIGNMENT
# ==========================================
cat("Normalizing (TSS) and aligning feature spaces...\n")

# 1. Normalize Train and Pivot Long
train_long <- train_otu_raw |>
  select(-any_of(c("Taxon", "taxonomy", "Confidence"))) |>
  mutate(across(where(is.numeric), ~ .x / sum(.x))) |>
  pivot_longer(cols = -UnifiedID, names_to = "Sample_Name", values_to = "count")

# 2. Normalize Test and Pivot Long
test_long <- test_otu_raw |>
  select(-any_of(c("Taxon", "taxonomy", "Confidence"))) |>
  mutate(across(where(is.numeric), ~ .x / sum(.x))) |>
  pivot_longer(cols = -UnifiedID, names_to = "Sample_Name", values_to = "count")

# 3. Combine and Pivot Wider (Fills gaps with 0)
# Because bind_rows puts Train first, Train samples will strictly occupy rows 1:n_train.
aligned_otu <- bind_rows(train_long, test_long) |>
  pivot_wider(names_from = UnifiedID, values_from = count, values_fill = 0)

# Extract pure numeric matrix (Rows = Samples, Columns = Microbes)
final_matrix <- aligned_otu |>
  select(-Sample_Name) |>
  as.matrix()

# ==========================================
# EXPORT TO DEEPMICRO FORMAT
# ==========================================
# Ensure output directory exists before writing
output_dir <- dirname(opt$output)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save as a clean CSV without headers or index columns (DeepMicro standard)
write.table(final_matrix, 
            file = opt$output, 
            sep = ",", 
            col.names = FALSE, 
            row.names = FALSE)

cat("=======================================================\n")
cat("SUCCESS: Transformed, normalized, and aligned matrices for DeepMicro.\n")
cat(sprintf("Dimensions: %d samples (rows) x %d microbes (columns).\n", nrow(final_matrix), ncol(final_matrix)))
cat(sprintf("Output saved to: %s\n", opt$output))
cat("=======================================================\n")