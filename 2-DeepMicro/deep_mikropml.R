#!/usr/bin/env Rscript

# =============================================================================
# Script: deep_mikropml.R
# Author: Ivan Espejo
#
# Description:
#     Trains and evaluates mikropml classifiers using the global DeepMicro
#     embeddings for LOCO study-level or random sample-level splits.
#
# Inputs:
#     --mode: loco or random.
#     --embeddings: DeepMicro sample-by-feature embedding CSV.
#     --train_file and --test_file: files listing study IDs or sample IDs.
#     --labels: CSV containing Sample_Name, Label, and study_id for LOCO.
#     --output_dir: destination for this run's results.
#
# Outputs:
#     Predictions and model evaluation results under --output_dir.
# =============================================================================

# Setup
library(ggplot2)
library(tidyverse)
library(mikropml)
library(doFuture)
library(caret)
library(optparse)
library(data.table)

registerDoFuture()
plan(multisession, workers = 8)  # Upped to 8 to match mikropml_sbatch
options(doFuture.rng.onMisuse = "ignore")

# Dynamically set width to match your terminal window
options(width = as.numeric(system("tput cols", intern = TRUE)))

################################################# 
################# DATA LOADING ##################
#################################################

option_list = list(
  make_option(c("--mode"), type="character", default="loco", help="Execution mode: 'loco' or 'random'"),
  make_option(c("--embeddings"), type="character", default=NULL, help="Path to global DeepMicro embeddings CSV"),
  make_option(c("--train_file"), type="character", default=NULL, help="Path to train list TXT"),
  make_option(c("--test_file"), type="character", default=NULL, help="Path to test list TXT"),
  make_option(c("--labels"), type="character", default=NULL, help="Path to global label CSV"),
  make_option(c("--output_dir"), type="character", default=NULL, help="Output directory")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (any(sapply(list(opt$embeddings, opt$train_file, opt$test_file, opt$labels, opt$output_dir), is.null))) {
  print_help(opt_parser)
  stop("Critical Error: Missing required arguments.", call.=FALSE)
}

mode <- tolower(opt$mode)

if (!dir.exists(opt$output_dir)) dir.create(opt$output_dir, recursive = TRUE)

cat("=== DEEPMICRO CLASSIFIER PIPELINE INITIALIZED ===\n")
cat("Mode:        ", toupper(mode), "\n")
cat("Embeddings:  ", opt$embeddings, "\n")
cat("Labels CSV:  ", opt$labels, "\n")
cat("Train File:  ", opt$train_file, "\n")
cat("Test File:   ", opt$test_file, "\n")
cat("Output Dir:  ", opt$output_dir, "\n\n")

# 1. Load Embeddings
# Global DeepMicro output has no headers. Column 1 is Sample Name, rest are latent dimensions.
embeds <- read_csv(opt$embeddings, col_names = FALSE, show_col_types = FALSE) %>%
  rename(Sample_Name = 1) %>%
  mutate(Sample_Name = as.character(Sample_Name))

# Name the dimensions safely for formulas
colnames(embeds)[-1] <- paste0("dim_", 1:(ncol(embeds)-1))

# 2. Load Labels
labels_all <- read_csv(opt$labels, show_col_types = FALSE) %>% 
  mutate(Sample_Name = as.character(Sample_Name))

# 3. Read text lists
train_list <- scan(opt$train_file, what = "", sep = "\n", quiet = TRUE)
test_list <- scan(opt$test_file, what = "", sep = "\n", quiet = TRUE)
train_list <- train_list[train_list != ""]
test_list <- test_list[test_list != ""]

# 4. Dual-Mode Subsetting Logic
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

combined_labels <- bind_rows(train_labels, test_labels)

# 5. Bind Labels to Embeddings
# We inner join to only keep the embeddings for the samples active in this split
mikropml_input <- combined_labels |> 
  inner_join(embeds, by = "Sample_Name") |> 
  select(Sample_Name, dx = Label, starts_with("dim_")) |> 
  mutate(dx = factor(dx, levels = c("0", "1"), labels = c("Healthy", "IBD"))) |>  
  as.data.frame() |> 
  column_to_rownames("Sample_Name")


#################################################
################# PREPROCESSING #################
#################################################
# Capture the original sample names before mikropml strips them
original_sample_names <- rownames(mikropml_input)

print("Preprocessing latent embeddings...")
mikropml_input_preproc <- preprocess_data(
  dataset = mikropml_input,
  outcome_colname = "dx",
  method = NULL   # Skip global scaling to prevent leakage
)

combined_transformed <- mikropml_input_preproc$dat_transformed

# Re-attach the sample names
rownames(combined_transformed) <- original_sample_names


#################################################
################ RUNNING MODELS #################
#################################################

# 6. Define Dual-Mode Partitions for run_ml()
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

print(paste("Initiating Training on", n_train_samples, "samples..."))
print(paste("Train IDs:", paste(head(train_list, 5), collapse = ", "), "..."))
print(paste("Test IDs:", paste(head(test_list, 5), collapse = ", "), "..."))

# Helper function for unified execution
execute_mikropml <- function(method_name) {
  run_ml(
    dataset = combined_transformed,
    method = method_name, 
    outcome_colname = "dx",
    groups = group_vector,                                         
    group_partitions = list(train = train_split_arg, test = test_split_arg),
    kfold = 5,
    cv_times = 10,
    seed = 23102004,
    preProcess = c("center", "scale")   # Dynamically scales the embeddings securely
  )
}

# --- GLMNET ---
print("Training GLMNET baseline...")
results_glmnet <- execute_mikropml("glmnet")

# --- SVM (Radial) ---
print("Training Radial SVM...")
results_svm <- execute_mikropml("svmRadial")

model_results <- list(
  GLMNET = results_glmnet,
  SVM = results_svm
)

# --- Random Forest ---
if (n_train_samples > 100) {
  print("Training sample size > 100 detected. Training Random Forest...")
  results_rf <- execute_mikropml("rf")
  model_results$rf <- results_rf
} else {
  print("Training sample size <= 100. Skipping Random Forest.")
}


################################################# 
############### GENERATING RESULTS ##############
#################################################

print("Saving output files...")

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
  
  # A. ROC Curve Generation
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

print("Pipeline execution completed successfully.")