#!/usr/bin/env Rscript

# =============================================================================
# Script: lopo_execution_alpha.R
# Author: Ivan Espejo
#
# Description:
#     Calculates richness and effective Shannon diversity from the global OTU
#     table, subsets samples using the supplied split IDs, and trains a
#     mikropml classifier using only those two diversity metrics.
#
# Inputs:
#     --mode: loco or random.
#     --train_file and --test_file: files listing study IDs or sample IDs.
#     --otu: global abundance table in TSV format.
#     --labels: sample labels CSV; LOCO mode also requires study_id.
#     --output_dir: destination for this run's results.
#
# Outputs:
#     Alpha Diversity model predictions, performance metrics, and plots under
#     --output_dir.
# =============================================================================

# 0. Package loading
library(ggplot2)
library(tidyverse)
library(mikropml)
library(doFuture)
library(caret)
library(optparse)
library(data.table)
library(vegan)

# Parallel execution
registerDoFuture()
plan(multisession, workers = 8)
options(doFuture.rng.onMisuse = "ignore")
options(width = as.numeric(system("tput cols", intern = TRUE)))

################################################# 
################# DATA LOADING ##################
#################################################

option_list = list(
  make_option(c("--mode"), type="character", default="loco", help="Execution mode: 'loco' or 'random'"),
  make_option(c("--train_file"), type="character", default=NULL, help="Path to text file with Training IDs"),
  make_option(c("--test_file"), type="character", default=NULL, help="Path to text file with Testing IDs"),
  make_option(c("--otu"), type="character", default=NULL, help="Path to Global OTU table (TSV)"),
  make_option(c("--labels"), type="character", default=NULL, help="Path to Global Labels CSV"),
  make_option(c("--output_dir"), type="character", default=NULL, help="Output directory")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

mode <- tolower(opt$mode)

cat("=== ALPHA DIVERSITY PIPELINE INITIALIZED ===\n")
cat("Mode:       ", toupper(mode), "\n")
cat("Global OTU: ", opt$otu, "\n")
cat("Labels:     ", opt$labels, "\n")

# Load Global Files
otu_raw <- read_tsv(opt$otu, show_col_types = FALSE)
labels_all <- read_csv(opt$labels, show_col_types = FALSE) %>%
  mutate(Sample_Name = as.character(Sample_Name))

# Read text lists for subsetting
train_list <- scan(opt$train_file, what = "", sep = "\n", quiet = TRUE)
test_list <- scan(opt$test_file, what = "", sep = "\n", quiet = TRUE)
train_list <- train_list[train_list != ""]
test_list <- test_list[test_list != ""]

# Subsetting Logic
if (mode == "loco") {
  if (!"study_id" %in% colnames(labels_all)) stop("labels.csv must contain 'study_id' for LOCO mode.")
  train_labels <- labels_all[labels_all$study_id %in% train_list, , drop = FALSE]
  test_labels  <- labels_all[labels_all$study_id %in% test_list, , drop = FALSE]
} else if (mode == "random") {
  train_labels <- labels_all[labels_all$Sample_Name %in% train_list, , drop = FALSE]
  test_labels  <- labels_all[labels_all$Sample_Name %in% test_list, , drop = FALSE]
} else {
  stop("Invalid mode provided.")
}

combined_labels <- bind_rows(train_labels, test_labels)

#################################################
######## PREPROCESSING TO ALPHA DIVERSITY #######
#################################################

cat("Calculating Global Alpha Diversity metrics...\n")

# 1. Format OTU
num_matrix <- otu_raw %>% select(-1, -any_of(c("Taxon", "taxonomy", "Confidence")))
transposed_matrix <- as.data.frame(t(num_matrix))

# 2. Normalize to Relative Abundances (TSS)
otu_input <- as.data.frame(sweep(transposed_matrix, 1, rowSums(transposed_matrix), "/"))

# 3. Calculate Diversity
alpha_df <- data.frame(
  Richness = specnumber(otu_input),
  Shannon = diversity(otu_input, index = "shannon")
) %>%
  mutate(Shannon_Effective = exp(Shannon)) %>%
  select(Richness, Shannon_Effective) %>%
  rownames_to_column("Sample_Name")

# 4. Join precisely with the subset labels
mikropml_input <- combined_labels %>% 
  inner_join(alpha_df, by = "Sample_Name") %>% 
  select(Sample_Name, dx = Label, Richness, Shannon_Effective) %>% 
  mutate(dx = factor(dx, levels = c("0", "1"), labels = c("Healthy", "IBD"))) %>%  
  as.data.frame() %>% 
  column_to_rownames("Sample_Name")

original_sample_names <- rownames(mikropml_input)

# Mikropml Prep
mikropml_input_preproc <- preprocess_data(
  dataset = mikropml_input, 
  outcome_colname= "dx",
  method = NULL   # Safely cleans 0-variance features WITHOUT global scaling
)
combined_transformed <- mikropml_input_preproc$dat_transformed
rownames(combined_transformed) <- original_sample_names

#################################################
################ RUNNING MODELS #################
#################################################

# Dual-Mode Partitions for run_ml()
if (mode == "loco") {
  sample_to_study <- combined_labels %>% select(Sample_Name, study_id) %>% deframe()
  group_vector <- sapply(rownames(combined_transformed), function(s) sample_to_study[[s]])
  train_split_arg <- train_list
  test_split_arg <- test_list
} else {
  group_vector <- rownames(combined_transformed) 
  train_split_arg <- train_list
  test_split_arg <- test_list
}

n_train_samples <- sum(rownames(combined_transformed) %in% train_labels$Sample_Name)
cat(sprintf("\nInitiating Training on %d samples based on 2 Alpha metrics...\n", n_train_samples))

# Helper function
execute_mikropml <- function(method_name) {
  run_ml(
    dataset = combined_transformed,        
    method = method_name, 
    outcome_colname = "dx",
    groups = group_vector,                                         
    group_partitions = list(train = train_split_arg, test = test_split_arg), 
    kfold = 2,
    cv_times = 5,  # Reduced cross-validation aggressiveness for only 2 features
    seed = 23102004,
    preProcess = c("center", "scale")
  )
}

# Training
print("Training GLMNET...")
results_glmnet <- execute_mikropml("glmnet")

# We only run GLMNET for the baseline to avoid redundant computational overhead.
# Alpha Diversity is purely linear/bivariate, so RF and SVM rarely perform better than Logistic Regression here.

model_results <- list(
  GLMNET = results_glmnet
)

################################################# 
############### GENERATING RESULTS ##############
#################################################

print("Extracting Predictions and Evaluating Test Sets...")

out_dirs <- list(
  preds = file.path(opt$output_dir, "predictions"),
  roc = file.path(opt$output_dir, "roc_curves"),
  metrics = file.path(opt$output_dir, "performance_metrics")
)
invisible(lapply(out_dirs, function(d) if(!dir.exists(d)) dir.create(d, recursive = TRUE)))

for (model_name in names(model_results)) {
  current_res <- model_results[[model_name]]
  
  probs <- predict(current_res$trained_model, newdata = current_res$test_data, type = "prob")
  preds <- predict(current_res$trained_model, newdata = current_res$test_data)
  
  # A. Predictions Format
  pred_table <- data.table(
    SampleID = rownames(current_res$test_data),
    Actual = current_res$test_data$dx,
    Predicted = preds
  )
  pred_table <- cbind(pred_table, probs)
  fwrite(pred_table, file.path(out_dirs$preds, paste0("Predictions_", model_name, ".csv")))
}

# Cross-Validation Metrics
perf_list <- lapply(names(model_results), function(m) {
  df <- model_results[[m]]$performance
  df$Model <- m
  return(df)
})
all_perf <- do.call(rbind, perf_list)
fwrite(all_perf, file.path(out_dirs$metrics, "All_Models_Performance.csv"))

print("Pipeline execution completed successfully.")