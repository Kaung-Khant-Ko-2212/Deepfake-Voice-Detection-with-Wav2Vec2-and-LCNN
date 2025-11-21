# utils.py
import os
import glob
import torch
import torchaudio
import numpy as np
import librosa
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import librosa.display
import warnings


# --- Configuration (Centralized) ---
TARGET_SAMPLE_RATE = 16000
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base" # Or "facebook/wav2vec2-large-xlsr-53" for larger model
LCNN_INPUT_DIM = 768 # Hidden size of wav2vec2-base
GMM_N_COMPONENTS = 16
GMM_COVARIANCE_TYPE = 'diag' # 'full' can be more powerful but needs more data, 'tied' is also an option

# For segment-level prediction and real-time
SEGMENT_LENGTH_SECONDS = 3 # Length of audio segment for prediction
SEGMENT_OVERLAP_SECONDS = 1.5 # Overlap between segments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress the specific UserWarning from transformers (for utils.py too)
warnings.filterwarnings(
    "ignore",
    message="Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the `Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`.",
    category=UserWarning
)

# --- Dataset Class ---
class CombinedAudioDataset(Dataset):
    def __init__(self, real_dirs, fake_dirs, target_sample_rate=TARGET_SAMPLE_RATE):
        self.all_files = []
        self.labels = []
        self.target_sample_rate = target_sample_rate

        for real_dir in real_dirs:
            real_files_in_dir = glob.glob(os.path.join(real_dir, "**", "*.wav"), recursive=True)
            self.all_files.extend(real_files_in_dir)
            self.labels.extend([0] * len(real_files_in_dir)) # 0 for real

        for fake_dir in fake_dirs:
            fake_files_in_dir = glob.glob(os.path.join(fake_dir, "**", "*.wav"), recursive=True)
            self.all_files.extend(fake_files_in_dir)
            self.labels.extend([1] * len(fake_files_in_dir)) # 1 for fake

        # Shuffle the data (important for training)
        combined_list = list(zip(self.all_files, self.labels))
        np.random.shuffle(combined_list)
        self.all_files, self.labels = zip(*combined_list)
        self.all_files = list(self.all_files)
        self.labels = list(self.labels)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        audio_path = self.all_files[idx]
        label = self.labels[idx]

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            # print(f"Error loading {audio_path}: {e}. Returning dummy data.")
            return torch.zeros(self.target_sample_rate), -1 # -1 signifies an invalid sample

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform.squeeze(0), label


class PreExtractedFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.data_files = glob.glob(os.path.join(feature_dir, "*.pt"))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        return data['features'], data['label']

# --- Feature Extraction Functions ---
# Load pre-trained Wave2Vec2 model and feature extractor once
feature_extractor_w2v2 = AutoFeatureExtractor.from_pretrained(WAV2VEC2_MODEL_NAME)
wav2vec2_model_global = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME).to(DEVICE)
wav2vec2_model_global.eval() # Set to eval mode; we'll only use it for feature extraction

def extract_wav2vec2_features(audio_waveform, sample_rate=TARGET_SAMPLE_RATE):
    """
    Extracts a single Wav2Vec2 feature vector (averaged over time) from an audio waveform.
    Used for training LCNN on aggregated features.
    """
    if isinstance(audio_waveform, torch.Tensor):
        audio_waveform = audio_waveform.cpu().numpy()

    if audio_waveform.size == 0:
        return torch.zeros(LCNN_INPUT_DIM)

    # For very short audio, the feature extractor might fail. Pad if too short.
    # A common minimum length for Wav2Vec2 is around 400ms (16000 * 0.4 = 6400 samples)
    if len(audio_waveform) < 6400:
        padding_needed = 6400 - len(audio_waveform)
        audio_waveform = np.pad(audio_waveform, (0, padding_needed), 'constant')


    inputs = feature_extractor_w2v2(audio_waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = wav2vec2_model_global(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu() # Return to CPU

def extract_mfccs(waveform, n_mfcc=40):
    """
    Extracts MFCC features from an audio waveform.
    """
    # Ensure waveform is on CPU and convert to numpy
    # Squeeze to remove single-dimensional entries (e.g., batch dimension, channel dimension for mono)
    waveform_np = waveform.squeeze().cpu().numpy()

    # Handle empty waveform after squeezing
    if waveform_np.size == 0:
        return np.array([]) # Return empty array if no audio data

    # Ensure audio_waveform is float for librosa
    if waveform_np.dtype != np.float32 and waveform_np.dtype != np.float64:
        waveform_np = waveform_np.astype(np.float32)

    mfccs = librosa.feature.mfcc(y=waveform_np, sr=TARGET_SAMPLE_RATE, n_mfcc=n_mfcc)
    
    # Transpose to get (n_frames, n_mfcc) format, which is suitable for GMMs
    return mfccs.T

# --- GMM Functions ---
def train_gmm(features, n_components=GMM_N_COMPONENTS, covariance_type=GMM_COVARIANCE_TYPE):
    print(f"Training GMM with {len(features)} samples...")
    if features.shape[0] < n_components:
        print(f"Warning: Number of GMM training samples ({features.shape[0]}) is less than components ({n_components}). Adjusting components.")
        n_components_adjusted = max(1, features.shape[0]) # Use at least 1 component if possible
        if n_components_adjusted == 0:
             print("No valid samples for GMM training. Returning None.")
             return None
        gmm = GaussianMixture(n_components=n_components_adjusted, covariance_type=covariance_type, random_state=42, verbose=0)
    else:
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42, verbose=0)
    
    gmm.fit(features)
    print("GMM training complete.")
    return gmm

def score_gmm(gmm_model, features):
    if features.ndim == 1:
        features = features.reshape(1, -1)
    return gmm_model.score_samples(features)

# --- Metric Logging ---
def init_log(log_path):
    columns = ['Epoch', 'Train_Loss', 'Train_Accuracy', 'Train_F1_Score',
               'Val_Loss', 'Val_Accuracy', 'Val_F1_Score', 'Val_ROC_AUC', 'Val_EER', 'Best_Val_Loss']
    df = pd.DataFrame(columns=columns)
    df.to_excel(log_path, index=False)

def log_metrics(log_path, epoch, train_loss, train_acc, train_f1,
                val_loss, val_acc, val_f1, val_roc_auc, val_eer, best_val_loss):
    df = pd.read_excel(log_path)
    new_row = pd.DataFrame([{
        'Epoch': epoch,
        'Train_Loss': train_loss,
        'Train_Accuracy': train_acc,
        'Train_F1_Score': train_f1,
        'Val_Loss': val_loss,
        'Val_Accuracy': val_acc,
        'Val_F1_Score': val_f1,
        'Val_ROC_AUC': val_roc_auc,
        'Val_EER': val_eer,
        'Best_Val_Loss': best_val_loss
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(log_path, index=False)

def compute_equal_error_rate(y_true, scores):
    """
    Computes Equal Error Rate (EER) and its threshold using scores for the positive class.
    Returns (eer, threshold).
    """
    if len(set(y_true)) < 2:
        return float('nan'), None
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    fnr = 1 - tpr
    diffs = np.abs(fnr - fpr)
    idx = np.argmin(diffs)
    eer = (fnr[idx] + fpr[idx]) / 2.0
    thr = thresholds[idx] if idx < thresholds.shape[0] else None
    return float(eer), (float(thr) if thr is not None else None)

# --- Path Definitions ---
def get_dataset_paths(base_path):
    train_dirs_real = [
        os.path.join(base_path, "for-2sec", "for-2seconds", "training", "real"),
        os.path.join(base_path, "for-norm", "training", "real"),
        os.path.join(base_path, "for-original", "training", "real"),
        os.path.join(base_path, "for-rerec", "training", "real"),
    ]

    train_dirs_fake = [
        os.path.join(base_path, "for-2sec", "for-2seconds", "training", "fake"),
        os.path.join(base_path, "for-norm", "training", "fake"),
        os.path.join(base_path, "for-original", "training", "fake"),
        os.path.join(base_path, "for-rerec", "training", "fake"),
    ]

    val_dirs_real = [
        os.path.join(base_path, "for-2sec", "for-2seconds", "validation", "real"),
        os.path.join(base_path, "for-norm", "validation", "real"),
        os.path.join(base_path, "for-original", "validation", "real"),
        os.path.join(base_path, "for-rerec", "validation", "real"),
    ]

    val_dirs_fake = [
        os.path.join(base_path, "for-2sec", "for-2seconds", "validation", "fake"),
        os.path.join(base_path, "for-norm", "validation", "fake"),
        os.path.join(base_path, "for-original", "validation", "fake"),
        os.path.join(base_path, "for-rerec", "validation", "fake"),
    ]
    return train_dirs_real, train_dirs_fake, val_dirs_real, val_dirs_fake

# --- Plotting Functions ---
def plot_training_history(log_file_path, save_path="training_history.png"):
    """Plots training and validation metrics from the log file."""
    try:
        df = pd.read_excel(log_file_path)
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}. Cannot plot training history.")
        return

    epochs = df['Epoch']
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    fig.suptitle('Training and Validation Metrics Over Epochs', fontsize=16)

    # Plot Loss
    axes[0, 0].plot(epochs, df['Train_Loss'], label='Train Loss')
    axes[0, 0].plot(epochs, df['Val_Loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot Accuracy
    axes[0, 1].plot(epochs, df['Train_Accuracy'], label='Train Accuracy')
    axes[0, 1].plot(epochs, df['Val_Accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot F1 Score
    axes[1, 0].plot(epochs, df['Train_F1_Score'], label='Train F1 Score')
    axes[1, 0].plot(epochs, df['Val_F1_Score'], label='Validation F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot ROC AUC
    axes[1, 1].plot(epochs, df['Val_ROC_AUC'], label='Validation ROC AUC', color='purple')
    axes[1, 1].set_title('ROC AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close(fig) # Close the figure to free memory

def get_segment_predictions(waveform, sample_rate, lcnn_model, segment_length_sec=SEGMENT_LENGTH_SECONDS, overlap_sec=SEGMENT_OVERLAP_SECONDS):
    """
    Processes an audio waveform in overlapping segments and returns LCNN fake probabilities for each segment.
    """
    segment_length_samples = int(segment_length_sec * sample_rate)
    hop_length_samples = int((segment_length_sec - overlap_sec) * sample_rate)

    if hop_length_samples <= 0:
        raise ValueError("Hop length must be greater than 0. Adjust segment_length_sec and overlap_sec.")

    segment_probs = []
    segment_start_times = []

    # Ensure waveform is a numpy array for slicing
    waveform_np = waveform.squeeze().cpu().numpy()

    if waveform_np.size < segment_length_samples:
        # If audio is shorter than one segment, pad and process as a single segment
        padded_waveform = np.pad(waveform_np, (0, segment_length_samples - waveform_np.size), 'constant')
        features = extract_wav2vec2_features(padded_waveform, sample_rate)
        if features.numel() > 0 and not torch.isnan(features).any() and not torch.isinf(features).any():
            features = features.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                lcnn_output = lcnn_model(features)
                probs = torch.softmax(lcnn_output, dim=1).cpu().squeeze().numpy()
                segment_probs.append(probs[1]) # Probability of fake
                segment_start_times.append(0.0)
        return np.array(segment_start_times), np.array(segment_probs)


    for i in range(0, waveform_np.size - segment_length_samples + 1, hop_length_samples):
        segment = waveform_np[i : i + segment_length_samples]
        
        # Ensure segment is not empty after slicing
        if segment.size == 0:
            continue

        features = extract_wav2vec2_features(segment, sample_rate)
        
        # Skip if feature extraction failed for the segment
        if features.numel() == 0 or torch.isnan(features).any() or torch.isinf(features).any():
            segment_probs.append(0.5) # Neutral probability if features are bad
            # print(f"Warning: Problematic features for segment starting at {i/sample_rate:.2f}s. Using neutral prob.")
            continue

        features = features.unsqueeze(0).to(DEVICE) # Add batch dimension
        with torch.no_grad():
            lcnn_output = lcnn_model(features)
            probs = torch.softmax(lcnn_output, dim=1).cpu().squeeze().numpy()
            segment_probs.append(probs[1]) # Probability of fake
            segment_start_times.append(i / sample_rate)

    return np.array(segment_start_times), np.array(segment_probs)

def plot_waveform_predictions(waveform, sample_rate, segment_start_times, segment_probs, threshold=0.5, save_path="prediction_waveform.png"):
    """
    Plots the audio waveform and highlights segments predicted as fake.
    """
    plt.figure(figsize=(18, 6))
    
    # Plot waveform
    librosa.display.waveshow(waveform.squeeze().cpu().numpy(), sr=sample_rate, alpha=0.6, color='gray')
    
    # Overlay fake segments
    for i, start_time in enumerate(segment_start_times):
        prob = segment_probs[i]
        if prob >= threshold:
            end_time = start_time + SEGMENT_LENGTH_SECONDS
            # Use a color gradient or fixed color for fake
            color = plt.cm.Reds(prob) # Redder for higher fake probability
            plt.axvspan(start_time, end_time, color=color, alpha=0.4, label=f'Fake (Prob: {prob:.2f})' if i == 0 else "")
    
    plt.title(f'Audio Waveform with Deepfake Segment Predictions (Threshold: {threshold})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1) # Assuming normalized audio
    plt.grid(True, alpha=0.3)
    
    # Create a custom legend for the color bar if needed, or just rely on the color gradient visually
    # For a simple legend, manually add entries if you have distinct categories
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.savefig(save_path)
    print(f"Waveform prediction plot saved to {save_path}")
    plt.close() # Close the figure to free memory