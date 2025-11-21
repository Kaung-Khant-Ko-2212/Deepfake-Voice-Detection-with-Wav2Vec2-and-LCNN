🎙️ Deepfake Voice Detection with Wav2Vec2 & LCNNA hybrid deep learning framework designed to detect synthetic speech and voice cloning attacks. This system leverages the self-supervised power of Wav2Vec2 for feature extraction and a custom Light CNN (LCNN) classifier, augmented by a Gaussian Mixture Model (GMM) for robust verification.🧠 System ArchitectureThe project employs a dual-model approach to ensure high accuracy in distinguishing between Bona Fide (Real) and Spoofed (Deepfake) audio.graph TD
    A[Input Audio] --> B{Feature Extraction}
    B -->|Wav2Vec2| C[Transformer Embeddings]
    B -->|MFCC| D[Spectral Features]
    C --> E[LCNN Classifier]
    D --> F[GMM Verifier]
    E --> G{Decision Fusion}
    F --> G
    G --> H[Final Prediction: Real vs Fake]
Frontend (Feature Extractor): Uses facebook/wav2vec2-base to extract rich context-aware acoustic representations.Backend (Classifier):LCNN: A dense neural network optimized for classifying Wav2Vec2 embeddings.GMM: A probabilistic model trained on MFCCs to capture spectral distributions.Hybrid Scoring: Combines predictions from both models to handle diverse attack types.📊 FeaturesReal-time Microphone Analysis: Record and analyze voice clips instantly (sounddevice integration).Hybrid Detection: Fuses Deep Learning (Wav2Vec2) with Signal Processing (GMM/MFCC).Visual Forensics: Generates waveform heatmaps highlighting specific segments detected as "Fake".Comprehensive Metrics: Tracks Accuracy, F1-Score, ROC-AUC, and EER (Equal Error Rate).Segment-Level Detection: Analyzes audio in overlapping windows (default 3s) to pinpoint fake artifacts.🛠️ InstallationClone the repository:git clone [https://github.com/yourusername/deepfake-voice-detection.git](https://github.com/yourusername/deepfake-voice-detection.git)
cd deepfake-voice-detection
Install dependencies:pip install torch torchaudio transformers librosa scikit-learn pandas matplotlib tqdm sounddevice openpyxl
📂 Dataset StructureThe training script expects a specific directory structure. Ensure your data is organized as follows inside your BASE_DATASET_PATH:datasets/dataset2/
├── for-2sec/
│   ├── training/ (real/fake)
│   └── validation/ (real/fake)
├── for-norm/
├── for-original/
└── for-rerec/
Modify BASE_DATASET_PATH in train.py if your data is stored elsewhere.🚀 Usage1. Training the ModelTrain the LCNN and GMM models simultaneously. The script automatically handles feature pre-extraction and saves the best models.python train.py
Outputs:trained_models_combined/lcnn_model_combined.pth (Best Model)trained_models_combined/training_history.png (Loss/Acc Graphs)trained_models_combined/training_log.xlsx (Excel logs)2. Prediction (Single File)Analyze a specific .wav file. This generates a waveform visualization showing which parts of the audio are fake.python predict.py --audio_path "path/to/audio.wav"
3. Real-time RecordingRecord voice from your microphone and analyze it instantly.# Records for 5 seconds (default)
python predict.py --record_and_predict

# Customize duration
python predict.py --record_and_predict --record_duration 10
📈 Results & VisualizationTraining PerformanceThe graph below shows the training/validation loss and accuracy over epochs.(Note: Run train.py to generate this graph)Deepfake HeatmapThe system analyzes audio in segments. Red areas indicate high probability of synthetic manipulation.(Note: Run predict.py on a file to generate this visualization)Performance MetricsMetricScoreAccuracy98.5%F1 Score0.98EER0.02ROC AUC0.99These are example metrics. Check your training_log.xlsx for your actual model performance.🧩 Modules Descriptiontrain.py: Main training loop with Early Stopping, Class Weighting, and Gradient Clipping.predict.py: Inference script for files and live recording; handles visualization.model.py: Defines the LCNN architecture (Input -> Linear -> Dropout -> Linear -> Output).utils.py: Helper functions for Wav2Vec2/MFCC extraction, Dataset loading, and Plotting.📄 LicenseDistributed under the MIT License. See LICENSE for more information.
