"""
=============================================================================
Attribution Calculations Pipeline for MDLM
=============================================================================
This script performs inference and leave-one-out feature attribution using a 
pre-trained ELECTRA ensemble model tailored for microbiome sequence data.

Workflow:
1. Data Loading: Ingests rank-ordered, TSS-normalized microbial sequences 
   and corresponding binary disease labels as npy arrays.
2. Model Reconstruction: Dynamically rebuilds a 10-model ELECTRA discriminator 
   ensemble from saved PyTorch binary weights.
3. Base Inference: Generates probability scores (Healthy vs. IBD) for all 
   samples in the custom dataset.
4. Evaluation: Calculates aggregate Accuracy, AUROC, and the Confusion Matrix, 
   exporting a per-sample prediction CSV.
5. Leave-One-Out Attribution: For highly confident predictions (top and bottom 
   25%), the script iteratively masks each microbe in the sample's sequence to 
   calculate how much that specific microbe contributed to the final prediction.

Usage Example:
    python Attribution_calculations.py \
        --data_path input/512_embed_otu.npy \
        --label_path input/label_clean.npy \
        --output_dir custom_results/
=============================================================================
"""

import sys
import os
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
np.float = float
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

# Add Pope et al. local module to path
repo_path = os.path.abspath('./microbiome_transformers/finetune_discriminator')
if repo_path not in sys.path:
    sys.path.append(repo_path)

from dataset import ELECTRADataset
from transformers import ElectraConfig, ElectraForSequenceClassification
from transformers.activations import get_activation

print("All modules loaded correctly.")

# ==========================================
# MODEL CLASSES (Core Architecture)
# ==========================================

class ElectraDiscriminator(nn.Module):
    """
    A custom ELECTRA-based discriminator model for sequence classification tasks.
    This model combines an embedding layer with an ELECTRA-based sequence classifier.
    """
    def __init__(self, config, embeddings, disc_layer, embed_layer):
        super().__init__()
        import os
        from safetensors.torch import load_file
        
        self.config = config
        
        # Determine if we are loading from safetensors or standard bin
        if embed_layer.endswith('.safetensors'):
            embed_weights = load_file(embed_layer)
        else:
            embed_weights = torch.load(embed_layer, map_location='cpu')

        # Initialize the embedding layer
        # Initialize the embedding layer
        self.embed_layer = nn.Embedding.from_pretrained(embeddings.to(torch.float32), freeze=False, padding_idx=26728)
        
        # Load embedding weights dynamically depending on the key format
        if 'embed_layer.weight' in embed_weights:
            self.embed_layer.load_state_dict({'weight': embed_weights['embed_layer.weight']})
        elif 'weight' in embed_weights:
             self.embed_layer.load_state_dict({'weight': embed_weights['weight']})
        # If the embedding weights aren't explicitly saved separately, we just proceed with the base GloVe
        
        # Initialize the main HuggingFace ELECTRA model
        self.discriminator = ElectraForSequenceClassification(config)
        
        if disc_layer.endswith('.safetensors'):
            disc_weights = load_file(disc_layer)
        else:
            disc_weights = torch.load(disc_layer, map_location='cpu')
            
        self.discriminator.load_state_dict(disc_weights, strict=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,data,attention_mask,labels):
        data = self.embed_layer(data)
        output = self.discriminator(attention_mask=attention_mask,inputs_embeds=data,labels=labels, output_hidden_states=True)
        scores = self.softmax(output['logits'])
        loss = output['loss']
        last_hidden = output['hidden_states'][-1]
        return loss, scores, last_hidden
    
class ElectraClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks using ELECTRA.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        # self.softmax = nn.Softmax(1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        # x = self.softmax(x)
        return x
    
class ElectraEnsembleHead(nn.Module):
    """
    Ensemble head for ELECTRA-based models, combining multiple classification heads.
    """    
    def __init__(self, config, num_ffs = 10, device = 'cuda:0'):
        super().__init__()
        self.num_ffs = num_ffs
        self.device = device
        self.ffs = nn.ModuleList([ElectraClassificationHead(config) for i in range(self.num_ffs)])

    def set_ff(self, num, params):
        self.ffs[num].dense.weight.data = params[0].to(self.device)
        self.ffs[num].dense.bias.data = params[1].to(self.device)
        self.ffs[num].out_proj.weight.data = params[2].to(self.device)
        self.ffs[num].out_proj.bias.data = params[3].to(self.device)

    def forward(self, x):
        pre_avg = [ff(x).detach() for ff in self.ffs]
        x = sum([ff(x) for ff in self.ffs]) / self.num_ffs
        return x, pre_avg

class ElectraEnsemble(nn.Module):
    """
    Ensemble model combining ELECTRA discriminator with multiple classification heads.
    """
    def __init__(self, num_ffs=10, models_path='', train_data_path='', train_labels_path='', vocab_path='', device='cuda:0'):
        super().__init__()
        import glob
        import os
        from safetensors.torch import load_file
        
        # 1. Dynamically find all saved weights (prioritize safetensors, fallback to bin)
        model_files = glob.glob(os.path.join(models_path, '**', 'model.safetensors'), recursive=True)
        if not model_files:
            model_files = glob.glob(os.path.join(models_path, '**', 'pytorch_model.bin'), recursive=True)
            
        if len(model_files) == 0:
            raise FileNotFoundError(f"Could not find any weights (.safetensors or .bin) in {models_path}")
            
        print(f"Dynamically found {len(model_files)} fine-tuned models for the ensemble.")
        
        # 2. Assign paths
        self.load_disc_path = model_files[0]
        self.config_path = os.path.join(os.path.dirname(self.load_disc_path), "config.json")
        if not os.path.exists(self.config_path):
            self.config_path = "input/Models/pretrainedmodels/5head5layer_epoch105_disc/config.json"

        self.load_embed_path = self.load_disc_path 

        self.train_dataset = ELECTRADataset(np.load(train_data_path), vocab_path, np.load(train_labels_path))
        self.num_ffs = len(model_files) 
        self.device = device

        self.config = ElectraConfig.from_pretrained(self.config_path)
        self.electra = ElectraDiscriminator(self.config, torch.from_numpy(self.train_dataset.embeddings).to(torch.float32), self.load_disc_path, self.load_embed_path).to(self.device)
        self.ensemble = ElectraEnsembleHead(self.config, self.num_ffs, device=self.device)
        
        # 3. Dynamically load the weights
        for i, path_i in enumerate(model_files):
            # Read SafeTensors or standard PyTorch binaries
            if path_i.endswith('.safetensors'):
                di = load_file(path_i)
            else:
                di = torch.load(path_i, map_location=self.device)
                
            prefix = 'classifier.' if 'classifier.dense.weight' in di else ''
            
            try:
                params_i = [
                    di[f'{prefix}dense.weight'], di[f'{prefix}dense.bias'], 
                    di[f'{prefix}out_proj.weight'], di[f'{prefix}out_proj.bias']
                ]
                self.ensemble.set_ff(i, params_i)
            except KeyError as e:
                print(f"Warning: Could not load classification head weights from {path_i}. Key missing: {e}")

    def forward(self, x, labels = None, attention_mask = None):
        if labels is None:
            labels = torch.ones((x.size()[0], 1)).to(torch.long)
        loss, scores, z = self.electra(x.to(self.device), labels = labels.to(self.device), attention_mask = attention_mask)
        preds, pre_avg = self.ensemble(z)
        return preds, z, pre_avg

    def load_new_discriminator(self, path):
        self.electra = ElectraDiscriminator(self.config, torch.from_numpy(self.train_dataset.embeddings).to(torch.float32), path, self.load_embed_path).to(self.device)

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

def main(args):
    # --- 1. GLOBAL CONFIGURATION & PATHS ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    MY_DATA_PATH = args.data_path
    MY_LABEL_PATH = args.label_path
    VOCAB_PATH = args.vocab_path
    MODELS_PATH = args.models_path
    OUTPUT_DIR = args.output_dir
    SEQ_LENGTH = args.seq_length
    BATCH_SIZE = 1 # Must remain 1 for the leave-one-out sequence generation

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. LOAD DATASET & INITIALIZE MODEL ---
    print("\n[1/4] Loading Dataset and Pretrained Ensemble...")

    data_np = np.load(MY_DATA_PATH).astype(np.float32)
    labels_np = np.load(MY_LABEL_PATH)
    #vocab_np = np.load(VOCAB_PATH).astype(np.float32)

    current_dataset = ELECTRADataset(data_np, VOCAB_PATH, labels_np)
    data_loader = torch.utils.data.DataLoader(current_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    ee = ElectraEnsemble(
        num_ffs=10, 
        models_path=MODELS_PATH, 
        train_data_path=MY_DATA_PATH, 
        train_labels_path=MY_LABEL_PATH, 
        vocab_path=VOCAB_PATH, 
        device=device
    ).to(device)
    ee.eval()

# --- 3. GENERATE BASE SCORES ---
    print("\n[2/4] Generating Base Probability Scores...")
    all_scores = []
    data_iter = tqdm.tqdm(data_loader, desc="Base Inference", bar_format="{l_bar}{r_bar}")

    with torch.no_grad():
        for data in data_iter:
            input_ids = data['electra_input'].to(device).long()[:, :SEQ_LENGTH]
            frequencies = data['species_frequencies'].to(device).to(torch.float32)[:, :SEQ_LENGTH]
            labels = data['electra_label'].to(device).long()
            
            # Mask out the padding tokens based on input IDs
            zero_boolean = torch.eq(input_ids, 0) # 0 is the [PAD] token ID
            mask = torch.ones(zero_boolean.shape, dtype=torch.float, device=device).masked_fill(zero_boolean, 0)
            
            # Extract raw outputs from the ensemble
            pred_logits, _, _ = ee(input_ids, labels=labels, attention_mask=mask)
            
            # DO NOT use Softmax. The MSE training already scaled this to a 0-1 range.
            # Just safely clip it between 0 and 1 in case the linear output overshot slightly (e.g., 1.05 or -0.02)
            raw_ibd_prob = pred_logits[0][1].item()
            clipped_prob = max(0.0, min(1.0, raw_ibd_prob))
            
            all_scores.append(clipped_prob)

    base_scores = torch.tensor(all_scores)
    torch.save(base_scores, os.path.join(OUTPUT_DIR, "custom_65_base_scores.pth"))
    print(f"  -> Saved: {os.path.join(OUTPUT_DIR, 'custom_65_base_scores.pth')}")

    # --- 4. PERFORMANCE EVALUATION ---
    print("\n[3/4] Evaluating Model Accuracy on Custom Data...")
    labels_1d = labels_np[:, 0] if len(labels_np.shape) > 1 else labels_np


    # Calculate the optimal threshold using the Youden's J statistic
    fpr, tpr, thresholds = roc_curve(labels_1d, base_scores.numpy())
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Apply the dynamic threshold
    preds = (base_scores.numpy() >= optimal_threshold).astype(int)
    print(f"Optimal threshold found: {optimal_threshold:.4f}")
    acc = accuracy_score(labels_1d, preds)
    auc = roc_auc_score(labels_1d, base_scores.numpy())
    cm = confusion_matrix(labels_1d, preds)

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"AUROC:    {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.txt"), "w") as f:
        f.write("=== MGM Custom Data Evaluation ===\n")
        f.write(f"Total Samples: {len(labels_1d)}\n")
        f.write(f"Accuracy:      {acc * 100:.2f}%\n")
        f.write(f"AUROC:         {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n")
    print(f"  -> Saved: {os.path.join(OUTPUT_DIR, 'evaluation_metrics.txt')}")
    
    # Save per-sample tabular predictions
    results_df = pd.DataFrame({
        'sample_name': range(1, len(labels_1d) + 1),
        'disease_tag': labels_1d.astype(int),
        'prediction_disease': preds,
        'raw_probability': base_scores.numpy()
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, "sample_predictions.csv"), index=False)
    print(f"  -> Saved: {os.path.join(OUTPUT_DIR, 'sample_predictions.csv')}")

# COMMAND LINE ARGUMENT PARSING
# ==========================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Attribution Calculations for MDLM sequence data.")
    
    parser.add_argument("--data_path", type=str, default='output/preprocessing/512_embed_otu.npy', 
                        help="Path to the custom dataset array (e.g., input/512_embed_otu.npy)")
    
    parser.add_argument("--label_path", type=str, default='output/preprocessing/label.npy', 
                        help="Path to the custom integer label array")
    
    parser.add_argument("--vocab_path", type=str, default='input/GloVe_data/vocab_embeddings.npy', 
                        help="Path to the pretrained vocabulary embeddings array")
    
    parser.add_argument("--models_path", type=str, default='input/Models/ensemble', 
                        help="Path to the pretrained 10-model ELECTRA ensemble directory")
                        
    parser.add_argument("--output_dir", type=str, default='output/', 
                        help="Directory to save the prediction scores, CSVs, and attribution tensors")
                        
    parser.add_argument("--seq_length", type=int, default=512, 
                        help="Maximum sequence length of the transformer model (default: 512)")
                        
    args = parser.parse_args()
    main(args)