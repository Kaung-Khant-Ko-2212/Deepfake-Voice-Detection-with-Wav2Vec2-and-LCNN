# predict.py
import os
import torch
import numpy as np
import pickle
import sys
import torchaudio
import librosa
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score, classification_report
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import librosa.display
import sounddevice as sd # For audio recording
import time # For timing the recording

# Suppress the specific UserWarning from transformers
import warnings
warnings.filterwarnings(
    "ignore",
    message="Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the `Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`.",
    category=UserWarning
)

# Add the parent directory of 'utils.py' and 'model.py' to the Python path
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.append(script_dir)

from utils import (
    CombinedAudioDataset,
    extract_wav2vec2_features, extract_mfccs, score_gmm,
    get_dataset_paths,
    TARGET_SAMPLE_RATE, LCNN_INPUT_DIM,
    DEVICE,
    get_segment_predictions, plot_waveform_predictions,
    compute_equal_error_rate,
    SEGMENT_LENGTH_SECONDS, SEGMENT_OVERLAP_SECONDS # Import segment config
)
from model import LCNN

# --- Prediction Configuration ---
BASE_DATASET_PATH = r"C:\Workspace\Deepfake Voice Recognition\datasets\dataset2"
MODEL_OUTPUT_DIR = "trained_models_combined"

LCNN_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "lcnn_model_combined.pth")
GMM_REAL_PATH = os.path.join(MODEL_OUTPUT_DIR, "gmm_real_combined.pkl")
GMM_FAKE_PATH = os.path.join(MODEL_OUTPUT_DIR, "gmm_fake_combined.pkl")


def load_models():
    # Load LCNN model
    lcnn_model = LCNN(input_dim=LCNN_INPUT_DIM).to(DEVICE)
    if os.path.exists(LCNN_MODEL_PATH):
        lcnn_model.load_state_dict(torch.load(LCNN_MODEL_PATH, map_location=DEVICE))
        lcnn_model.eval()
        print(f"LCNN model loaded from {LCNN_MODEL_PATH}")
    else:
        print(f"Error: LCNN model not found at {LCNN_MODEL_PATH}. Please train the model first.")
        return None, None, None

    # Load GMM models
    gmm_real = None
    if os.path.exists(GMM_REAL_PATH):
        with open(GMM_REAL_PATH, 'rb') as f:
            gmm_real = pickle.load(f)
        print(f"Real GMM model loaded from {GMM_REAL_PATH}")
    else:
        print(f"Warning: Real GMM model not found at {GMM_REAL_PATH}.")

    gmm_fake = None
    if os.path.exists(GMM_FAKE_PATH):
        with open(GMM_FAKE_PATH, 'rb') as f:
            gmm_fake = pickle.load(f)
        print(f"Warning: Fake GMM model not found at {GMM_FAKE_PATH}.")
        
    return lcnn_model, gmm_real, gmm_fake

def predict_audio_with_hybrid_model(waveform, sample_rate, lcnn_model, gmm_real, gmm_fake, audio_identifier="<in-memory audio>"):
    """
    Performs a single, overall prediction for an audio waveform using the hybrid model.
    Returns overall prediction, LCNN fake probability, GMM fake score, GMM real score.
    """
    # Ensure waveform is on CPU and convert to numpy for initial processing
    # If the input is a PyTorch tensor, move it to CPU and convert to numpy
    if isinstance(waveform, torch.Tensor):
        waveform_np = waveform.squeeze().cpu().numpy()
    else: # Assume it's already a numpy array if not a tensor
        waveform_np = waveform.squeeze() # Ensure 1D

    if waveform_np.size == 0:
        print(f"Warning: Empty waveform provided for {audio_identifier}. Cannot predict.")
        return -1, 0.5, 0.0, 0.0 # Return -1 for error, neutral probabilities/scores

    # --- LCNN Prediction (overall average feature) ---
    lcnn_prob_fake = 0.5 # Default neutral
    lcnn_prediction = -1 # Default undecided
    try:
        # Extract a single, overall Wav2Vec2 feature vector
        w2v2_features = extract_wav2vec2_features(waveform_np, sample_rate).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            lcnn_output = lcnn_model(w2v2_features)
            lcnn_probs = torch.softmax(lcnn_output, dim=1).cpu().squeeze().numpy()
            lcnn_prob_fake = lcnn_probs[1] # Probability of being fake
            lcnn_prediction = np.argmax(lcnn_probs)
    except Exception as e:
        print(f"Error during LCNN prediction for {audio_identifier}: {e}")

    # --- GMM Prediction ---
    gmm_prediction = -1 # Default undecided
    score_real_gmm = 0.0
    score_fake_gmm = 0.0
    
    if gmm_real and gmm_fake:
        try:
            mfcc_features = extract_mfccs(waveform_np, sample_rate)
            if mfcc_features.shape[0] > 0:
                score_real_gmm = np.mean(score_gmm(gmm_real, mfcc_features))
                score_fake_gmm = np.mean(score_gmm(gmm_fake, mfcc_features))
                if score_fake_gmm > score_real_gmm:
                    gmm_prediction = 1 # GMM leans fake
                else:
                    gmm_prediction = 0 # GMM leans real
            else:
                print(f"Warning: No MFCC features extracted for {audio_identifier}.")
        except Exception as e:
            print(f"Error during GMM prediction for {audio_identifier}: {e}")

    # --- Hybrid Decision (Simple Fusion) ---
    final_prediction = -1 # Default to undecided

    # Prioritize LCNN if it made a valid prediction
    if lcnn_prediction != -1:
        final_prediction = lcnn_prediction
    elif gmm_prediction != -1: # Fallback to GMM if LCNN failed
        final_prediction = gmm_prediction
    else:
        print(f"Warning: Could not make a prediction for {audio_identifier} using either model. Defaulting to Real.")
        final_prediction = 0 # Default to real or handle as an unknown

    return final_prediction, lcnn_prob_fake, score_fake_gmm, score_real_gmm


def evaluate_models(lcnn_model, gmm_real, gmm_fake, val_dirs_real, val_dirs_fake):
    test_audio_dataset = CombinedAudioDataset(val_dirs_real, val_dirs_fake)

    true_labels = []
    hybrid_predictions = []
    lcnn_probs_fake = []
    
    print("\n--- Evaluating Hybrid System ---")
    for idx in tqdm(range(len(test_audio_dataset)), desc="Evaluating Samples"):
        audio_waveform, true_label = test_audio_dataset[idx]
        audio_path = test_audio_dataset.all_files[idx] # Get the path to print errors

        if true_label == -1 or audio_waveform.numel() == 0: # Skip if the dataset returned a dummy for errors
            print(f"Skipping problematic sample: {audio_path}")
            continue

        # Use the overall prediction function
        pred, lcnn_prob, _, _ = predict_audio_with_hybrid_model(audio_waveform, TARGET_SAMPLE_RATE, lcnn_model, gmm_real, gmm_fake, audio_path)
        
        if pred != -1: # Only include if a valid prediction was made
            true_labels.append(true_label)
            hybrid_predictions.append(pred)
            lcnn_probs_fake.append(lcnn_prob)
    
    if len(true_labels) == 0:
        print("No valid samples were evaluated. Check data paths and files.")
        return

    print("\n--- Evaluation Results ---")
    print(f"Total samples evaluated: {len(true_labels)}")
    print(f"Accuracy: {accuracy_score(true_labels, hybrid_predictions):.4f}")
    print(f"Precision: {precision_score(true_labels, hybrid_predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(true_labels, hybrid_predictions, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(true_labels, hybrid_predictions, average='binary', zero_division=0):.4f}")
    
    if len(np.unique(true_labels)) > 1:
        try:
            print(f"ROC AUC (LCNN Probs): {roc_auc_score(true_labels, lcnn_probs_fake):.4f}")
            eer, thr = compute_equal_error_rate(true_labels, lcnn_probs_fake)
            if eer == eer: # check not NaN
                print(f"Equal Error Rate (EER): {eer:.4f} at threshold {thr:.4f}")
            else:
                print("Equal Error Rate (EER): N/A (single-class labels)")
        except ValueError:
            print("ROC AUC not calculable: Only one class present in true labels for ROC AUC.")
            print("Equal Error Rate (EER): N/A (single-class labels)")
    else:
        print("ROC AUC not calculable: Only one class present in true labels.")
        print("Equal Error Rate (EER): N/A (single-class labels)")

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, hybrid_predictions))
    
    print("\nClassification Report:")
    print(classification_report(true_labels, hybrid_predictions, target_names=['Real', 'Fake'], zero_division=0))


# --- Record and Predict Function ---
def record_and_predict_audio(duration_seconds, lcnn_model, gmm_real, gmm_fake):
    """
    Records audio from the microphone for a specified duration and then predicts.
    """
    print(f"\n--- Recording for {duration_seconds} seconds ---")
    print("Speak now...")

    try:
        # Record audio
        recorded_audio = sd.rec(int(duration_seconds * TARGET_SAMPLE_RATE),
                                samplerate=TARGET_SAMPLE_RATE,
                                channels=1,
                                dtype='float32')
        sd.wait() # Wait until recording is finished
        print("Recording finished. Processing...")

        # Convert numpy array to PyTorch tensor for processing
        recorded_waveform_torch = torch.from_numpy(recorded_audio.squeeze())
        
        if recorded_waveform_torch.numel() == 0:
            print("Error: Recorded audio is empty. Cannot predict.")
            return

        # Get overall prediction for the recorded audio
        overall_pred, overall_lcnn_prob, overall_gmm_fake_score, overall_gmm_real_score = \
            predict_audio_with_hybrid_model(recorded_waveform_torch, TARGET_SAMPLE_RATE, lcnn_model, gmm_real, gmm_fake, "<Recorded Audio>")
        
        if overall_pred != -1:
            print(f"Overall Prediction: {'Fake' if overall_pred == 1 else 'Real'}")
            print(f"Overall LCNN Fake Probability: {overall_lcnn_prob:.4f}")
            if gmm_real and gmm_fake:
                print(f"Overall GMM Fake Score: {overall_gmm_fake_score:.4f}, GMM Real Score: {overall_gmm_real_score:.4f}")
        else:
            print("Could not make an overall prediction for the recorded audio.")

        # Get segment-level predictions for plotting the recorded audio
        print("\nGenerating segment-level predictions for visualization...")
        segment_start_times, segment_probs = get_segment_predictions(recorded_waveform_torch, TARGET_SAMPLE_RATE, lcnn_model)
        
        if len(segment_start_times) > 0:
            plot_waveform_predictions(recorded_waveform_torch, TARGET_SAMPLE_RATE, segment_start_times, segment_probs, save_path=os.path.join(MODEL_OUTPUT_DIR, "recorded_audio_prediction_waveform.png"))
            print(f"Waveform plot for recorded audio saved to {os.path.join(MODEL_OUTPUT_DIR, 'recorded_audio_prediction_waveform.png')}")
        else:
            print("No segments generated for plotting the recorded audio. It might be too short or problematic.")

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during recording or prediction: {e}")
    finally:
        sd.stop() # Ensure sounddevice is stopped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Voice Recognition Prediction Script")
    parser.add_argument("--audio_path", type=str, help="Path to a single audio file for prediction and visualization.")
    parser.add_argument("--record_and_predict", action="store_true", help="Enable recording from microphone, then predict.")
    parser.add_argument("--record_duration", type=int, default=5, help="Duration in seconds to record for --record_and_predict mode. Default is 5 seconds.")
    args = parser.parse_args()

    # Ensure only one prediction mode is selected
    if args.audio_path and args.record_and_predict:
        parser.error("Cannot use --audio_path and --record_and_predict simultaneously. Please choose one.")

    lcnn_model, gmm_real, gmm_fake = load_models()

    if lcnn_model is None:
        print("Exiting as LCNN model could not be loaded.")
        sys.exit(1)
        
    if args.audio_path:
        # Single audio file prediction and visualization
        if not os.path.exists(args.audio_path):
            print(f"Error: Audio file not found at {args.audio_path}")
            sys.exit(1)
        
        print(f"\n--- Predicting for: {args.audio_path} ---")
        
        # Load the full waveform for plotting
        try:
            waveform, sample_rate = torchaudio.load(args.audio_path)
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if waveform.numel() == 0:
                print(f"Error: Empty audio file {args.audio_path}. Cannot predict.")
                sys.exit(1)

        except Exception as e:
            print(f"Error loading audio file {args.audio_path}: {e}. Cannot proceed.")
            sys.exit(1)

        # Get overall prediction
        overall_pred, overall_lcnn_prob, overall_gmm_fake_score, overall_gmm_real_score = \
            predict_audio_with_hybrid_model(waveform, TARGET_SAMPLE_RATE, lcnn_model, gmm_real, gmm_fake, args.audio_path)
        
        if overall_pred != -1:
            print(f"Overall Prediction: {'Fake' if overall_pred == 1 else 'Real'}")
            print(f"Overall LCNN Fake Probability: {overall_lcnn_prob:.4f}")
            if gmm_real and gmm_fake:
                print(f"Overall GMM Fake Score: {overall_gmm_fake_score:.4f}, GMM Real Score: {overall_gmm_real_score:.4f}")
        else:
            print("Could not make an overall prediction for the provided audio file.")

        # Get segment-level predictions for plotting
        print("\nGenerating segment-level predictions for visualization...")
        segment_start_times, segment_probs = get_segment_predictions(waveform, TARGET_SAMPLE_RATE, lcnn_model)
        
        if len(segment_start_times) > 0:
            plot_waveform_predictions(waveform, TARGET_SAMPLE_RATE, segment_start_times, segment_probs, save_path=os.path.join(MODEL_OUTPUT_DIR, "audio_prediction_waveform.png"))
        else:
            print("No segments generated for plotting. Audio might be too short or problematic.")

    elif args.record_and_predict:
        # Record audio then predict
        record_and_predict_audio(args.record_duration, lcnn_model, gmm_real, gmm_fake)

    else:
        # Default behavior: Evaluate on the combined validation set
        _, _, val_dirs_real, val_dirs_fake = get_dataset_paths(BASE_DATASET_PATH)
        evaluate_models(lcnn_model, gmm_real, gmm_fake, val_dirs_real, val_dirs_fake)