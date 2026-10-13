#!/usr/bin/env Rscript

# =============================================================================
# Script: lopo_execution.R
# Author: Ivan Espejo
#
# Description:
#     Trains and evaluates mikropml OTU classifiers for either LOCO study-level
#     splits or random sample-level splits. Aligns train/test feature columns by
#     filling absent OTUs with zero before fitting each model.
#
# Inputs:
#     --mode: loco or random.
#     --train_file and --test_file: files listing study IDs or sample IDs.
#     --otu: global abundance table in TSV format.
#     --labels: CSV containing Sample_Name, Label, and study_id for LOCO.
#     --output_dir: destination for this run's results.
#
# Outputs:
#     Model predictions and evaluation metrics under --output_dir.
# =============================================================================

# 0. Package loading
library(ggplot2)
library(tidyverse)
library(mikropml)
library(doFuture)
library(caret)
library(this.path)
library(optparse)
library(data.table)

# 0.5 Doing parallel execution
registerDoFuture()
plan(multisession, workers = 8)
options(doFuture.rng.onMisuse = "ignore")
options(width = as.numeric(system("tput cols", intern = TRUE)))

################################################# 
################# DATA LOADING ##################
#################################################

# Parsing arguments 
option_list = list(
  make_option(c("--mode"), type="character", default="loco", help="Execution mode: 'loco' or 'random'"),
  make_option(c("--train_file"), type="character", default=NULL, help="Path to text file listing train IDs (projects or samples)"),
  make_option(c("--test_file"), type="character", default=NULL, help="Path to text file listing test IDs (projects or samples)"),
  make_option(c("--otu"), type="character", default="../data/input/raw_global_otu_dedup.tsv", help="Path to unified OTU table"),
  make_option(c("--labels"), type="character", default="../data/input/labels.csv", help="Path to labels CSV (Sample_Name,Label,study_id)"),
  make_option(c("--output_dir"), type="character", default=NULL, help="Directory to save output files")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

# Enforce mandatory arguments
if (any(sapply(list(opt$train_file, opt$test_file, opt$output_dir), is.null))) {
  print_help(opt_parser)
  stop("Critical Error: --train_file, --test_file and --output_dir are required.", call.=FALSE)
}

mode <- tolower(opt$mode)

cat("=== MIKROPML PIPELINE INITIALIZED ===\n")
cat("Mode:        ", toupper(mode), "\n")
cat("Unified OTU: ", opt$otu, "\n")
cat("Labels CSV:  ", opt$labels, "\n")
cat("Train File:  ", opt$train_file, "\n")
cat("Test File:   ", opt$test_file, "\n")
cat("Output Dir:  ", opt$output_dir, "\n\n")

setwd(this.path::this.dir())

# 1. Load Data (Ensuring UnifiedID is character to prevent merge errors)
otu_all <- read_tsv(opt$otu, show_col_types = FALSE) %>% mutate(UnifiedID = as.character(UnifiedID))
labels_all <- read_csv(opt$labels, show_col_types = FALSE) %>% mutate(Sample_Name = as.character(Sample_Name))

# Read list files (newline separated)
train_list <- scan(opt$train_file, what = "", sep = "\n", quiet = TRUE)
train_list <- unlist(strsplit(train_list, "\\s+"))
test_list <- scan(opt$test_file, what = "", sep = "\n", quiet = TRUE)
test_list <- unlist(strsplit(test_list, "\\s+"))

# Clean vectors: remove any empty strings
train_list <- train_list[train_list != ""]
test_list <- test_list[test_list != ""]

# 2. Dual-Mode Subsetting Logic
if (mode == "loco") {
  if (!"study_id" %in% colnames(labels_all)) stop("labels.csv must contain 'study_id' column for LOCO mode.")
  train_labels <- labels_all[labels_all$study_id %in% train_list, , drop = FALSE]
  test_labels  <- labels_all[labels_all$study_id %in% test_list, , drop = FALSE]
} else if (mode == "random") {
  train_labels <- labels_all[labels_all$Sample_Name %in% train_list, , drop = FALSE]
  test_labels  <- labels_all[labels_all$Sample_Name %in% test_list, , drop = FALSE]
} else {
  stop("Invalid mode provided. Must be 'loco' or 'random'.")
}

# Subset OTU table to only columns present in labels (plus UnifiedID)
train_sample_cols <- intersect(colnames(otu_all), train_labels$Sample_Name)
test_sample_cols  <- intersect(colnames(otu_all), test_labels$Sample_Name)
train_otu_raw <- otu_all %>% select(UnifiedID, any_of(train_sample_cols))
test_otu_raw  <- otu_all %>% select(UnifiedID, any_of(test_sample_cols))


#################################################
################# PREPROCESSING #################
#################################################

print("Aligning and Preprocessing Feature Spaces...")

# 3. Normalize Individually, then Pivot to Long Format
train_long <- train_otu_raw |>
  select(-any_of(c("Taxon", "taxonomy", "Confidence"))) |>
  mutate(across(where(is.numeric), ~ .x / sum(.x))) |>
  pivot_longer(cols = -UnifiedID, names_to = "Sample_Name", values_to = "count")

test_long <- test_otu_raw |>
  select(-any_of(c("Taxon", "taxonomy", "Confidence"))) |>
  mutate(across(where(is.numeric), ~ .x / sum(.x))) |>
  pivot_longer(cols = -UnifiedID, names_to = "Sample_Name", values_to = "count")

# 4. Combine and Pivot Wider
# values_fill = 0 safely fills in gaps where an OTU exists in Train but not Test (and vice-versa).
otu_input <- bind_rows(train_long, test_long) |>
  pivot_wider(names_from = UnifiedID, values_from = count, values_fill = 0) |>
  rename_with(~ gsub("`", "", .x))

# 5. Combine Labels and Map to OTU Table
combined_labels <- bind_rows(train_labels, test_labels)

mikropml_input <- combined_labels |> 
  inner_join(otu_input, by = "Sample_Name") |> 
  select(Sample_Name, dx = Label, everything()) |> 
  mutate(dx = factor(dx, levels = c("0", "1"), labels = c("Healthy", "IBD"))) |>  
  as.data.frame() |> 
  column_to_rownames("Sample_Name") |>
  rename_with(~ gsub(":", "_", .x))

# 6. Preprocess together 
original_sample_names <- rownames(mikropml_input)

mikropml_input_preproc <- preprocess_data(
  dataset = mikropml_input, 
  outcome_colname= "dx",
  method = NULL   # Safely cleans 0-variance features WITHOUT global scaling (prevents data leakage)
)
combined_transformed <- mikropml_input_preproc$dat_transformed

# Re-attach the sample names to the transformed dataset
rownames(combined_transformed) <- original_sample_names

#################################################
################ RUNNING MODELS #################
#################################################

# 7. Define Dual-Mode Partitions for run_ml()
if (mode == "loco") {
  # LOCO: Map samples to their biological study/project
  sample_to_study <- combined_labels %>% select(Sample_Name, study_id) %>% deframe()
  group_vector <- sapply(rownames(combined_transformed), function(s) sample_to_study[[s]])
  train_split_arg <- train_list
  test_split_arg <- test_list
} else {
  # RANDOM: Every sample is its own independent group. 
  # This forces mikropml to perfectly respect our train_list and test_list.
  group_vector <- rownames(combined_transformed) 
  train_split_arg <- train_list
  test_split_arg <- test_list
}

n_train_samples <- sum(rownames(combined_transformed) %in% train_labels$Sample_Name)

print(paste("Initiating Training on", n_train_samples, "samples..."))
print(paste("Train IDs:", paste(head(train_list, 5), collapse = ", "), "..."))
print(paste("Test IDs:", paste(head(test_list, 5), collapse = ", "), "..."))

# Helper function for unified execution
execute_mikropml <- function(method_name) {
  run_ml(
    dataset = combined_transformed,        # Pass the FULL dataset
    method = method_name, 
    outcome_colname = "dx",
    groups = group_vector,                 # Dual-mode group vector                         
    group_partitions = list(train = train_split_arg, test = test_split_arg), # Python splits
    kfold = 5,
    cv_times = 10,
    seed = 23102004,
    preProcess = c("center", "scale")      # Dynamic, leakage-free scaling
  )
}

# --- GLMNET ---
print("Training glmnet...")
results_glmnet <- execute_mikropml("glmnet")

# --- SVM (Radial) ---
print("Training SVM (Radial)...")
results_svm <- execute_mikropml("svmRadial")

# --- Random Forest ---
print("Training Random Forest...")
results_rf <- execute_mikropml("rf")


model_results <- list(
  GLMNET = results_glmnet,
  SVM = results_svm,
  RF = results_rf
)

################################################# 
############### GENERATING RESULTS ##############
#################################################

print(paste("Evaluating models on unseen test samples..."))

# 1. Create Organized Subdirectories
out_dirs <- list(
  preds = file.path(opt$output_dir, "predictions"),
  roc = file.path(opt$output_dir, "roc_curves"),
  metrics = file.path(opt$output_dir, "performance_metrics")
)
invisible(lapply(out_dirs, function(d) if(!dir.exists(d)) dir.create(d, recursive = TRUE)))

# 2. Iterate through trained models to save predictions and ROCs
for (model_name in names(model_results)) {
  print(paste("Processing outputs for:", model_name))
  current_res <- model_results[[model_name]]
  
  # A. ROC Curve Generation (run_ml returns the full object natively)
  sens_dat <- calc_model_sensspec(
    trained_model = current_res$trained_model,
    test_data = current_res$test_data,
    outcome_colname = "dx"
  )
  
  p_roc <- ggplot(sens_dat, aes(x = specificity, y = sensitivity)) +
    geom_line(color = "#0072B2", linewidth = 1) +
    geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "grey50") +
    coord_equal() +
    scale_x_reverse(expand = c(0, 0), limits = c(1.01, -0.01)) +
    scale_y_continuous(expand = c(0, 0), limits = c(-0.01, 1.01)) +
    labs(x = "Specificity", y = "Sensitivity", title = paste("ROC Curve:", model_name)) +
    theme_bw() + theme(plot.title = element_text(face="bold"))
    
  ggsave(p_roc, filename = file.path(out_dirs$roc, paste0("ROC_", model_name, ".png")), width = 6, height = 4)
  
  # B. Prediction Table (Probabilities & Classes)
  probs <- predict(current_res$trained_model, newdata = current_res$test_data, type = "prob")
  preds <- predict(current_res$trained_model, newdata = current_res$test_data)
  
  pred_table <- data.table(
    SampleID = rownames(current_res$test_data),
    Actual = current_res$test_data$dx,
    Predicted = preds
  )
  pred_table <- cbind(pred_table, probs)
  fwrite(pred_table, file.path(out_dirs$preds, paste0("Predictions_", model_name, ".csv")))
  
  # C. Confusion Matrix Output
  cm <- confusionMatrix(data = preds, reference = current_res$test_data$dx)
  capture.output(print(cm), file = file.path(out_dirs$preds, paste0("ConfusionMatrix_", model_name, ".txt")))
}

# 3. Global Performance Summary & Advanced Plotting
perf_list <- lapply(names(model_results), function(m) {
  df <- model_results[[m]]$performance
  df$Model <- m
  return(df)
})
all_perf <- do.call(rbind, perf_list)
fwrite(all_perf, file.path(out_dirs$metrics, "All_Models_Performance.csv"))

# Build Cross-Validation Distribution Plot
p_cv <- ggplot(all_perf, aes(x = Model, y = cv_metric_AUC, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1) +
  theme_bw() +
  labs(
    title = "Cross-Validation AUROC Stability",
    subtitle = paste("Distribution of AUC across repeated CV folds -", toupper(mode), "Mode"),
    y = "Cross-Validated AUC",
    x = "Algorithm"
  ) +
  theme(legend.position = "none", plot.title = element_text(face="bold"))

ggsave(p_cv, filename = file.path(out_dirs$metrics, paste0("CV_AUC_Distribution_", toupper(mode), ".png")), width = 7, height = 5)

print(paste("Dual-Mode Pipeline execution completed successfully for", toupper(mode), "mode."))