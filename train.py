# train.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score
import numpy as np
from tqdm import tqdm
import pickle
import sys
import warnings

# Suppress the specific UserWarning from transformers
warnings.filterwarnings(
    "ignore",
    message="Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the `Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`.",
    category=UserWarning
)

# Add the parent directory of 'utils.py' and 'model.py' to the Python path
# This assumes utils.py, model.py, train.py are in the same directory.
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.append(script_dir)

from utils import (
    CombinedAudioDataset, PreExtractedFeatureDataset,
    extract_wav2vec2_features, extract_mfccs,
    train_gmm, score_gmm,
    init_log, log_metrics,
    get_dataset_paths,
    TARGET_SAMPLE_RATE, LCNN_INPUT_DIM, GMM_N_COMPONENTS, GMM_COVARIANCE_TYPE,
    DEVICE, plot_training_history, compute_equal_error_rate # Import the new plotting function
)
from model import LCNN

# --- Training Configuration ---
BASE_DATASET_PATH = r"C:\Workspace\Deepfake Voice Recognition\datasets\dataset2"
FEATURE_OUTPUT_DIR = "pre_extracted_features_combined"
MODEL_OUTPUT_DIR = "trained_models_combined"
LOG_FILE = "training_log.xlsx"

NUM_EPOCHS_LCNN = 20 # Increased epochs for better convergence
LCNN_BATCH_SIZE = 32 # Increased batch size
LCNN_LEARNING_RATE = 1e-4
LCNN_WEIGHT_DECAY = 1e-5 # L2 regularization
GRADIENT_CLIP_NORM = 1.0 # Gradient clipping

# Early Stopping parameters
PATIENCE = 5 # Number of epochs to wait for improvement
MIN_DELTA = 0.001 # Minimum change to qualify as an improvement

os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def pre_extract_and_save_w2v2_features_wrapper(real_dirs, fake_dirs, output_dir_name):
    output_dir = os.path.join(FEATURE_OUTPUT_DIR, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    dataset = CombinedAudioDataset(real_dirs, fake_dirs, target_sample_rate=TARGET_SAMPLE_RATE)
    print(f"Pre-extracting Wave2Vec2 features for {output_dir_name} ({len(dataset)} samples)...")
    
    saved_count = 0
    skipped_count = 0
    for i in tqdm(range(len(dataset)), desc="Extracting W2V2"):
        waveform, label = dataset[i]
        
        if label == -1 or waveform.numel() == 0: # Handle invalid or empty samples
            skipped_count += 1
            continue

        features = extract_wav2vec2_features(waveform)
        
        # Additional check for features (e.g., if very short audio produced problematic features)
        if features is None or features.numel() == 0 or torch.isnan(features).any() or torch.isinf(features).any():
             # print(f"Warning: Problematic features for sample {i}. Skipping.")
             skipped_count += 1
             continue

        torch.save({'features': features, 'label': label}, os.path.join(output_dir, f'sample_{i}.pt'))
        saved_count += 1
    print(f"Finished pre-extraction for {output_dir_name}. Saved {saved_count} samples, Skipped {skipped_count} samples.")
    return output_dir

def train_lcnn_model(train_loader, val_loader, num_epochs, model_output_path, log_path, patience, min_delta):
    lcnn_model = LCNN(input_dim=LCNN_INPUT_DIM).to(DEVICE)
    
    # Calculate class weights for imbalanced datasets
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.tolist())
    
    class_counts = np.bincount(train_labels)
    # Ensure no division by zero if a class is missing
    if len(class_counts) < 2:
        print("Warning: Only one class found in training data. Class weighting not applied.")
        class_weights = torch.tensor([1.0, 1.0]).to(DEVICE)
    else:
        total_samples = sum(class_counts)
        class_weights = torch.tensor([total_samples / (2.0 * count) for count in class_counts], dtype=torch.float32).to(DEVICE)
        print(f"Calculated class weights: {class_weights}")


    criterion = nn.CrossEntropyLoss(weight=class_weights) # Apply class weights
    optimizer = optim.Adam(lcnn_model.parameters(), lr=LCNN_LEARNING_RATE, weight_decay=LCNN_WEIGHT_DECAY) # L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 2) # Reduce LR on plateau

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    print("\n--- Training LCNN Model on Combined Data ---")
    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered.")
            break

        lcnn_model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} LCNN Train"):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = lcnn_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(lcnn_model.parameters(), GRADIENT_CLIP_NORM)
            
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='binary', zero_division=0)

        # Validation
        lcnn_model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} LCNN Val"):
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = lcnn_model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() # Probability of fake class
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probabilities)

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='binary', zero_division=0)
        
        # Calculate ROC AUC and EER only if both classes are present in validation labels
        if len(np.unique(all_val_labels)) > 1:
            epoch_val_roc_auc = roc_auc_score(all_val_labels, all_val_probs)
            epoch_val_eer, eer_thr = compute_equal_error_rate(all_val_labels, all_val_probs)
        else:
            epoch_val_roc_auc = 0.0 # Or np.nan
            epoch_val_eer = float('nan')

        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_accuracy:.4f}, F1: {epoch_train_f1:.4f}")
        if epoch_val_eer == epoch_val_eer:
            print(f"              Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_accuracy:.4f}, F1: {epoch_val_f1:.4f}, ROC AUC: {epoch_val_roc_auc:.4f}, EER: {epoch_val_eer:.4f}")
        else:
            print(f"              Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_accuracy:.4f}, F1: {epoch_val_f1:.4f}, ROC AUC: {epoch_val_roc_auc:.4f}, EER: N/A")

        # Log metrics to Excel
        log_metrics(log_path, epoch + 1, epoch_train_loss, epoch_train_accuracy, epoch_train_f1,
                    epoch_val_loss, epoch_val_accuracy, epoch_val_f1, epoch_val_roc_auc, epoch_val_eer, best_val_loss)

        # Learning Rate Scheduler step
        scheduler.step(epoch_val_loss)

        # Early Stopping check
        if epoch_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(lcnn_model.state_dict(), model_output_path)
            print(f"Model improved. Saving best model to {model_output_path}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Epochs without improvement: {epochs_no_improve}")
            if epochs_no_improve >= patience:
                early_stop = True
                print("Early stopping triggered due to no improvement in validation loss.")
    
    # Save the final model state if not already saved by early stopping
    if not os.path.exists(model_output_path):
        torch.save(lcnn_model.state_dict(), model_output_path)
        print(f"Final model saved to {model_output_path} (no early stop improvement).")

    print("LCNN training complete.")


def train_gmm_models(train_audio_dataset_gmm, gmm_real_path, gmm_fake_path):
    print("\n--- Preparing MFCC Features for GMM on Combined Data ---")
    mfcc_features_real = []
    mfcc_features_fake = []
    
    skipped_mfcc_count = 0
    for waveform, label in tqdm(train_audio_dataset_gmm, desc="Extracting MFCCs for GMM"):
        if label == -1 or waveform.numel() == 0: # Skip invalid or empty samples
            skipped_mfcc_count += 1
            continue

        mfccs = extract_mfccs(waveform)
        
        if mfccs.shape[0] == 0: # If MFCC extraction failed (e.g., extremely short audio)
            # print(f"Warning: No MFCCs extracted for a waveform. Skipping.")
            skipped_mfcc_count += 1
            continue

        if label == 0: # Real
            mfcc_features_real.append(mfccs)
        else: # Fake
            mfcc_features_fake.append(mfccs)
    
    print(f"Skipped {skipped_mfcc_count} samples during MFCC extraction.")

    mfcc_features_real_combined = np.vstack(mfcc_features_real) if mfcc_features_real else np.array([])
    mfcc_features_fake_combined = np.vstack(mfcc_features_fake) if mfcc_features_fake else np.array([])


    print("\n--- Training GMM Models on Combined Data ---")
    gmm_real = None
    if mfcc_features_real_combined.size > 0:
        gmm_real = train_gmm(mfcc_features_real_combined, n_components=GMM_N_COMPONENTS, covariance_type=GMM_COVARIANCE_TYPE)
    else:
        print("No real samples for GMM training.")

    gmm_fake = None
    if mfcc_features_fake_combined.size > 0:
        gmm_fake = train_gmm(mfcc_features_fake_combined, n_components=GMM_N_COMPONENTS, covariance_type=GMM_COVARIANCE_TYPE)
    else:
        print("No fake samples for GMM training.")

    if gmm_real:
        with open(gmm_real_path, 'wb') as f:
            pickle.dump(gmm_real, f)
        print(f"Real GMM model saved to {gmm_real_path}")
    
    if gmm_fake:
        with open(gmm_fake_path, 'wb') as f:
            pickle.dump(gmm_fake, f)
        print(f"Fake GMM model saved to {gmm_fake_path}")


if __name__ == "__main__":
    train_dirs_real, train_dirs_fake, val_dirs_real, val_dirs_fake = get_dataset_paths(BASE_DATASET_PATH)

    # --- Pre-extract Wave2Vec2 features for LCNN ---
    train_w2v2_feature_dir = pre_extract_and_save_w2v2_features_wrapper(train_dirs_real, train_dirs_fake, "combined_train_w2v2")
    val_w2v2_feature_dir = pre_extract_and_save_w2v2_features_wrapper(val_dirs_real, val_dirs_fake, "combined_val_w2v2")

    train_w2v2_dataset = PreExtractedFeatureDataset(train_w2v2_feature_dir)
    val_w2v2_dataset = PreExtractedFeatureDataset(val_w2v2_feature_dir)

    train_w2v2_loader = DataLoader(train_w2v2_dataset, batch_size=LCNN_BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    val_w2v2_loader = DataLoader(val_w2v2_dataset, batch_size=LCNN_BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)

    # Initialize logging file
    log_file_path = os.path.join(MODEL_OUTPUT_DIR, LOG_FILE)
    init_log(log_file_path)

    # --- Train LCNN ---
    lcnn_model_path = os.path.join(MODEL_OUTPUT_DIR, "lcnn_model_combined.pth")
    train_lcnn_model(train_w2v2_loader, val_w2v2_loader, NUM_EPOCHS_LCNN, lcnn_model_path, log_file_path, PATIENCE, MIN_DELTA)

    # --- Train GMMs ---
    train_audio_dataset_gmm = CombinedAudioDataset(train_dirs_real, train_dirs_fake, target_sample_rate=TARGET_SAMPLE_RATE)
    gmm_real_path = os.path.join(MODEL_OUTPUT_DIR, "gmm_real_combined.pkl")
    gmm_fake_path = os.path.join(MODEL_OUTPUT_DIR, "gmm_fake_combined.pkl")
    train_gmm_models(train_audio_dataset_gmm, gmm_real_path, gmm_fake_path)

    print("\nTraining process complete. Models and logs saved.")
    
    # Plot training history after training is complete
    plot_training_history(log_file_path, os.path.join(MODEL_OUTPUT_DIR, "training_history.png"))