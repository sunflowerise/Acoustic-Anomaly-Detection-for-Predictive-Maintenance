import os
from typing import List, Tuple, Dict

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter


def list_wav_files(root_dir: str) -> List[str]:
    """
    Recursively collect all .wav file paths under a directory.
    """
    wav_files: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(dirpath, f))
    return sorted(wav_files)


def extract_mfcc_features(
    file_path: str,
    n_mfcc: int = 40,
    sr: int = 16000,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    Load a WAV file and extract MFCC features aggregated over time.

    Returns a 1D feature vector (mean and std of each MFCC over time).
    """
    # Load audio
    y, orig_sr = librosa.load(file_path, sr=None, mono=True)

    # Resample if needed
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    # Compute MFCCs: shape (n_mfcc, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

    # Aggregate over time axis → mean and std for each coefficient
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std], axis=0)  # shape (2 * n_mfcc,)
    return features.astype(np.float32)


def extract_features_for_files(file_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract MFCC-based feature vectors for a list of file paths.

    Returns:
        X: np.ndarray of shape (N, F) where N = number of files.
        kept_paths: list of file paths that were successfully processed.
    """
    features: List[np.ndarray] = []
    kept_paths: List[str] = []

    for p in file_paths:
        try:
            feat = extract_mfcc_features(p)
            features.append(feat)
            kept_paths.append(p)
        except Exception:
            # Skip files that fail to load/parse
            continue

    if not features:
        return np.empty((0, 0), dtype=np.float32), []

    X = np.stack(features, axis=0)
    return X, kept_paths


def extract_mel_spectrogram(
    file_path: str,
    sr: int = 16000,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Extract mel spectrogram from audio file.
    
    Returns:
        mel_spec: np.ndarray of shape (n_mels, T) - mel spectrogram
    """
    # Load audio
    y, orig_sr = librosa.load(file_path, sr=None, mono=True)
    
    # Resample if needed
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        hop_length=hop_length, 
        n_fft=n_fft,
        n_mels=n_mels
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db.astype(np.float32)


def reconstruct_baseline_mel(mel_spec: np.ndarray, method: str = "smooth") -> np.ndarray:
    """
    Reconstruct a baseline/normal mel spectrogram for comparison.
    
    Methods:
    - "smooth": Apply temporal and frequency smoothing (assumes normal operation is smooth)
    - "mean": Use mean across time (assumes stationary normal operation)
    - "median": Use median across time (more robust to outliers)
    
    Returns:
        baseline_mel: np.ndarray of same shape as mel_spec
    """
    if method == "smooth":
        # Apply 2D Gaussian smoothing to create a "normal" baseline
        # This assumes normal operation has smooth transitions
        baseline = gaussian_filter(mel_spec, sigma=(2.0, 5.0))  # Smooth more in time
        return baseline.astype(np.float32)
    
    elif method == "mean":
        # Use mean across time axis (assumes stationary normal)
        baseline = np.mean(mel_spec, axis=1, keepdims=True)
        return np.tile(baseline, (1, mel_spec.shape[1])).astype(np.float32)
    
    elif method == "median":
        # Use median across time (more robust)
        baseline = np.median(mel_spec, axis=1, keepdims=True)
        return np.tile(baseline, (1, mel_spec.shape[1])).astype(np.float32)
    
    else:
        # Default: smooth
        baseline = gaussian_filter(mel_spec, sigma=(2.0, 5.0))
        return baseline.astype(np.float32)


def calculate_mel_difference(
    file_path: str,
    reconstruction_method: str = "smooth",
    sr: int = 16000,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 128,
) -> Dict[str, np.ndarray]:
    """
    Calculate |Original Mel - Reconstructed Mel| for XAI visualization.
    
    Returns:
        Dictionary with:
        - 'original': Original mel spectrogram
        - 'reconstructed': Baseline/reconstructed mel spectrogram
        - 'difference': |Original - Reconstructed| (anomaly heatmap)
        - 'time_axis': Time axis in seconds
        - 'freq_axis': Frequency axis in mel bins
    """
    # Extract original mel spectrogram
    mel_original = extract_mel_spectrogram(
        file_path, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels
    )
    
    # Reconstruct baseline
    mel_reconstructed = reconstruct_baseline_mel(mel_original, method=reconstruction_method)
    
    # Calculate absolute difference
    mel_difference = np.abs(mel_original - mel_reconstructed)
    
    # Create time and frequency axes
    duration = mel_original.shape[1] * hop_length / sr
    time_axis = np.linspace(0, duration, mel_original.shape[1])
    freq_axis = np.arange(mel_original.shape[0])
    
    return {
        "original": mel_original,
        "reconstructed": mel_reconstructed,
        "difference": mel_difference,
        "time_axis": time_axis,
        "freq_axis": freq_axis,
        "sample_rate": sr,
        "hop_length": hop_length,
    }


