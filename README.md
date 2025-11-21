## Deepfake Voice Recognition

This repository contains a hybrid deepfake-speech detection pipeline that pairs self-supervised speech representations (Wav2Vec 2.0) with a lightweight convolutional neural network (LCNN) and augments them with Gaussian Mixture Models (GMMs) trained on MFCCs. The goal is to learn complementary cues that generalize across recording conditions, re-recordings, and short utterances.

### Highlights
- **Two-branch detector**: LCNN classifies pooled Wav2Vec2 embeddings; parallel GMMs score MFCC statistics for real vs fake speech.
- **Pre-extraction workflow**: Audio is converted into `.pt` feature blobs so LCNN training stays GPU-friendly and restartable.
- **Robust metrics & visualizations**: Training logs are written to Excel (`training_log.xlsx`) and summarized with accuracy, F1, ROC-AUC, and Equal Error Rate plots. Segment-level predictions can be visualized over raw waveforms.
- **Realtime-friendly inferencing**: `predict.py` supports single-file checks, microphone recordings, and validation-set sweeps with tqdm progress bars.

### Repository Layout
- `train.py` – orchestrates preprocessing, LCNN training (with early stopping and LR scheduling), GMM training, and metric logging/plotting.
- `predict.py` – loads trained artifacts for dataset evaluation, one-off inference, or live microphone recordings; generates waveform overlays of fake segments.
- `model.py` – defines the feedforward LCNN classifier that consumes 768-d Wav2Vec2 embeddings.
- `utils.py` – shared datasets, feature extraction helpers (Wav2Vec2 + MFCC), GMM utilities, metric logging, plotting, and segment-level inference logic.
- `datasets/` – expected home for dataset2 (see *Data* below).
- `pre_extracted_features_combined/` – cached Wave2Vec2 embeddings generated during training.
- `trained_models_combined/` – LCNN weights (`.pth`), GMM pickles (`.pkl`), logs, and plots.

### Data
The scripts assume the HiFi TTS / ASVspoof-style dataset structure rooted at:
```
C:\Workspace\Deepfake Voice Recognition\datasets\dataset2
```
Within this tree, training/validation splits are organized into `for-2sec`, `for-norm`, `for-original`, and `for-rerec` subfolders, each with `real/` and `fake/` leaves containing `.wav` files. Update `BASE_DATASET_PATH` in `train.py` and `predict.py` if your dataset lives elsewhere.

### Environment Setup
1. Create/activate a Python 3.9+ environment (GPU-enabled PyTorch recommended).
2. Install dependencies:
   ```powershell
   pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install transformers soundfile sounddevice librosa pandas matplotlib scikit-learn tqdm openpyxl
   ```
   Adjust the PyTorch index URL to match your CUDA/cuDNN version or use the default CPU wheels.
3. (Optional) If you plan to record audio, grant microphone permission to your Python runtime and ensure `sounddevice` can access WASAPI/Core Audio.

### Training Workflow
1. **Pre-extract Wave2Vec2 features** (handled automatically by `train.py`):
   - Audio is resampled to 16 kHz, converted to mono, passed through `facebook/wav2vec2-base`, and averaged over time.
   - Each sample is serialized to `pre_extracted_features_combined/<split>/*.pt`.
2. **Train LCNN**:
   - Uses class-balanced cross-entropy, gradient clipping, Adam + weight decay, ReduceLROnPlateau, and patience-based early stopping.
   - Best checkpoint is stored at `trained_models_combined/lcnn_model_combined.pth`.
3. **Train GMMs**:
   - MFCC stacks feed separate Gaussian mixtures for real/fake speech to capture complementary spectral cues.
4. **Log & plot**:
   - Metrics per epoch go into `training_log.xlsx`; `training_history.png` visualizes loss, accuracy, F1, and ROC-AUC trends.

Run everything via:
```powershell
python train.py
```
Use `CUDA_VISIBLE_DEVICES` or PyTorch device flags if you need to select a GPU.

### Prediction & Evaluation
Load trained artifacts and:
- **Evaluate validation split** (default):
  ```powershell
  python predict.py
  ```
- **Single audio file**:
  ```powershell
  python predict.py --audio_path path\to\clip.wav
  ```
  Prints hybrid predictions plus LCNN probability and (if available) GMM scores, then saves a waveform overlay to `trained_models_combined/audio_prediction_waveform.png`.
- **Live microphone recording**:
  ```powershell
  python predict.py --record_and_predict --record_duration 8
  ```
  Records `record_duration` seconds, classifies, and writes `recorded_audio_prediction_waveform.png`.

Hybrid decision logic prioritizes LCNN output (when available) and falls back to GMM scores; both branches report intermediate confidences so you can diagnose disagreements.

### Tips & Customization
- **Change encoder**: Update `WAV2VEC2_MODEL_NAME` in `utils.py` if you want a multilingual or larger backbone (remember to adjust `LCNN_INPUT_DIM`).  
- **Segment-level tuning**: Modify `SEGMENT_LENGTH_SECONDS` and `SEGMENT_OVERLAP_SECONDS` in `utils.py` for higher temporal resolution during visualization.  
- **GMM capacity**: Tweak `GMM_N_COMPONENTS` and `GMM_COVARIANCE_TYPE` to balance runtime and expressiveness.  
- **Dataset extensions**: Add new modality folders to `get_dataset_paths` or point `BASE_DATASET_PATH` to an alternate directory layout.  
- **Logging format**: `log_metrics` currently appends to Excel; adapt to CSV/JSON if you prefer repo-friendly plain text.

### Troubleshooting
- Ensure the dataset paths actually contain `.wav` files; corrupted or missing audio will be skipped with warnings.
- Wave2Vec2 feature extraction requires ~600 KB of VRAM per sample; reduce `LCNN_BATCH_SIZE` if you hit OOM.
- Recording mode needs a functioning input device; set `sounddevice.default.device` if autodetection fails.
- If you see *“gradient_checkpointing is deprecated”* warnings, they’re suppressed by default but can be ignored.

### License & Citation
No explicit license is provided in this repo. If you use this codebase in academic work, please cite Facebook AI’s Wav2Vec 2.0 paper and the relevant ASVspoof/Deepfake dataset sources alongside your own publication.

Feel free to open issues or pull requests with improvements, reproducibility results, or new dataset adapters!

