#!/usr/bin/env Rscript

# =============================================================================
# LOPO OTU Classification Pipeline (via mikropml)
# =============================================================================
# Description:
# Adapted for Leave-One-Project-Out (LOPO) cross-validation. This script takes
# strictly defined Train and Test datasets. It dynamically aligns their features 
# (filling missing OTUs with 0s) and processes them simultaneously to ensure 
# identical scaling and topography before model evaluation.
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

# 0.5 Doing parallel execution #
registerDoFuture()
plan(multisession, workers = 8)
options(doFuture.rng.onMisuse = "ignore")
options(width = as.numeric(system("tput cols", intern = TRUE)))

################################################# 
################# DATA LOADING ##################
#################################################

# Parsing arguments 
option_list = list(
  make_option(c("--train_otu"), type="character", default=NULL, help="Path to the Training OTU table file"),
  make_option(c("--train_labels"), type="character", default=NULL, help="Path to the Training label file"),
  make_option(c("--test_otu"), type="character", default=NULL, help="Path to the Testing OTU table file"),
  make_option(c("--test_labels"), type="character", default=NULL, help="Path to the Testing label file"),
  make_option(c("--output_dir"), type="character", default=NULL, help="Directory to save output files")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

# Enforce mandatory arguments
if (any(sapply(list(opt$train_otu, opt$train_labels, opt$test_otu, opt$test_labels, opt$output_dir), is.null))) {
  print_help(opt_parser)
  stop("Critical Error: All input paths (--train_otu, --train_labels, --test_otu, --test_labels, --output_dir) are required.", call.=FALSE)
}

cat("=== LOPO PIPELINE INITIALIZED ===\n")
cat("Training OTU:  ", opt$train_otu, "\n")
cat("Training LBL:  ", opt$train_labels, "\n")
cat("Testing OTU:   ", opt$test_otu, "\n")
cat("Testing LBL:   ", opt$test_labels, "\n")
cat("Output Dir:    ", opt$output_dir, "\n\n")

setwd(this.path::this.dir())

# 1. Load Data (Ensuring UnifiedID is character to prevent merge errors)
train_otu_raw <- read_tsv(opt$train_otu, show_col_types = FALSE) %>% mutate(UnifiedID = as.character(UnifiedID))
test_otu_raw <- read_tsv(opt$test_otu, show_col_types = FALSE) %>% mutate(UnifiedID = as.character(UnifiedID))

train_labels <- read_csv(opt$train_labels, show_col_types = FALSE) %>% mutate(Sample_Name = as.character(Sample_Name))
test_labels <- read_csv(opt$test_labels, show_col_types = FALSE) %>% mutate(Sample_Name = as.character(Sample_Name))

# TEST
#setwd(this.path::this.dir()) 
#train_otu_raw <- read_tsv("./input/.output_train_rest/raw_global_otu_dedup.tsv", show_col_types = FALSE) %>% mutate(UnifiedID = as.character(UnifiedID))
#test_otu_raw <- read_tsv("./input/.output_test_china/raw_global_otu_dedup.tsv", show_col_types = FALSE) %>% mutate(UnifiedID = as.character(UnifiedID))

#train_labels <- read_csv("./input/.output_train_rest/labels.csv", show_col_types = FALSE) %>% mutate(Sample_Name = as.character(Sample_Name))
#test_labels <- read_csv("./input/.output_test_china/labels.csv", show_col_types = FALSE) %>% mutate(Sample_Name = as.character(Sample_Name))

#################################################
################# PREPROCESSING #################
#################################################

print("Aligning and Preprocessing Feature Spaces...")

# 2. Normalize Individually, then Pivot to Long Format
train_long <- train_otu_raw |>
  select(-any_of(c("Taxon", "taxonomy", "Confidence"))) |>
  mutate(across(where(is.numeric), ~ .x / sum(.x))) |>
  pivot_longer(cols = -UnifiedID, names_to = "Sample_Name", values_to = "count")

test_long <- test_otu_raw |>
  select(-any_of(c("Taxon", "taxonomy", "Confidence"))) |>
  mutate(across(where(is.numeric), ~ .x / sum(.x))) |>
  pivot_longer(cols = -UnifiedID, names_to = "Sample_Name", values_to = "count")

# 3. Combine and Pivot Wider
# values_fill = 0 safely fills in gaps where an OTU exists in Train but not Test (and vice-versa).
otu_input <- bind_rows(train_long, test_long) |>
  pivot_wider(names_from = UnifiedID, values_from = count, values_fill = 0) |>
  rename_with(~ gsub("`", "", .x))

# 4. Combine Labels and Map to OTU Table
combined_labels <- bind_rows(train_labels, test_labels)

mikropml_input <- combined_labels |> 
  inner_join(otu_input, by = "Sample_Name") |> 
  select(Sample_Name, dx = Label, everything()) |> 
  mutate(dx = factor(dx, levels = c("0", "1"), labels = c("Healthy", "IBD"))) |>  
  as.data.frame() |> 
  column_to_rownames("Sample_Name") |>
  rename_with(~ gsub(":", "_", .x))

# 5. Preprocess together 
original_sample_names <- rownames(mikropml_input)

mikropml_input_preproc <- preprocess_data(
  dataset = mikropml_input, 
  outcome_colname= "dx"
)
combined_transformed <- mikropml_input_preproc$dat_transformed

# Re-attach the sample names to the transformed dataset
rownames(combined_transformed) <- original_sample_names

# 6. Define LOPO Partitions for run_ml()
# Instead of splitting the dataframe, we tag each row as "Train" or "Test"
group_vector <- ifelse(rownames(combined_transformed) %in% train_labels$Sample_Name, "Train", "Test")

# Count the training samples to conditionally trigger Random Forest later
n_train_samples <- sum(group_vector == "Train")

#################################################
################ RUNNING MODELS #################
#################################################

print(paste("Initiating Training on", n_train_samples, "samples..."))

# GLMNET
print("Training glmnet...")
results_glmnet <- run_ml(
  dataset = combined_transformed, 
  method = "glmnet", 
  outcome_colname = "dx",
  groups = group_vector,                                         # Pass the nametags
  group_partitions = list(train = c("Train"), test = c("Test")), # Force the split
  kfold = 5,
  cv_times = 10,
  seed = 23102004
)

# SVM (Radial)
print("Training SVM (Radial)...")
results_svm <- run_ml(
  dataset = combined_transformed, 
  method = "svmRadial", 
  outcome_colname = "dx",
  groups = group_vector,
  group_partitions = list(train = c("Train"), test = c("Test")),
  kfold = 5,
  cv_times = 10,
  seed = 23102004
)

# Structuring results mimicking mikropml's standard output
model_results <- list(
  GLMNET = results_glmnet,
  SVM = results_svm
)

# Random Forest conditionally based on training size
if (n_train_samples > 100) {
  print("Training sample size > 100 detected. Training Random Forest...")
  results_rf <- run_ml(
    dataset = combined_transformed, 
    method = "rf",                      
    outcome_colname = "dx",
    groups = group_vector,
    group_partitions = list(train = c("Train"), test = c("Test")),
    kfold = 5,
    cv_times = 10,
    seed = 23102004
  )
  model_results$rf <- results_rf
} else {
  print("Training sample size <= 100. Skipping Random Forest.")
}

################################################# 
############### GENERATING RESULTS ##############
#################################################

print(paste("Evaluating models on unseen LOPO test samples..."))

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
    subtitle = "Distribution of AUC across repeated CV folds",
    y = "Cross-Validated AUC",
    x = "Algorithm"
  ) +
  theme(legend.position = "none", plot.title = element_text(face="bold"))

ggsave(p_cv, filename = file.path(out_dirs$metrics, "CV_AUC_Distribution.png"), width = 7, height = 5)

print("LOPO Pipeline execution completed successfully.")