
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GroupKFold, KFold, cross_val_predict
from sklearn.decomposition import FastICA, PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import pywt
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import logm, expm, inv
from scipy.spatial.distance import pdist, squareform
import time
import joblib

warnings.filterwarnings("ignore")

class EnhancedBCIClassifier:
    def __init__(self, base_path="."):
        self.base_path = base_path
        self.fs = 250  # Sampling frequency
        self.mi_model = None
        self.ssvep_model = None
        self.feature_selector = None
        self.scaler = None
        self.ssvep_classes = ["Forward", "Backward", "Left", "Right"]
        self.mi_classes = ["Left", "Right"]
        
        # Standard EEG channel names (10-20 system)
        self.channel_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", 
                             "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
        
        # Initialize other attributes
        self.mi_feature_mask = None
        self.ssvep_feature_mask = None
        self.quality_threshold = 0.4
        self.confidence_threshold = 0.5
        
        # Enhanced preprocessing with multiple scalers
        self.mi_scaler = RobustScaler(quantile_range=(10, 90))  # More robust scaling
        self.ssvep_scaler = RobustScaler(quantile_range=(10, 90))
        self.mi_label_encoder = LabelEncoder()
        self.ssvep_label_encoder = LabelEncoder()
        
        # HEAVILY REGULARIZED MI Models to prevent overfitting
        self.mi_xgb1 = XGBClassifier(
            n_estimators=50,   # Drastically reduced from 150
            learning_rate=0.02,  # Much lower learning rate
            max_depth=3,       # Reduced from 5
            subsample=0.6,     # More aggressive subsampling
            colsample_bytree=0.6,  # More aggressive column sampling
            reg_alpha=5.0,     # Much higher L1 regularization
            reg_lambda=10.0,   # Much higher L2 regularization
            min_child_weight=10,  # Much higher minimum child weight
            gamma=0.5,         # Much higher gamma for pruning
            random_state=42,
            tree_method="hist",
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=10,
            validation_fraction=0.2
        )
        
        self.mi_xgb2 = XGBClassifier(
            n_estimators=60,   # Reduced from 180
            learning_rate=0.015,  # Even lower learning rate
            max_depth=4,       # Reduced from 6
            subsample=0.5,     # More aggressive subsampling
            colsample_bytree=0.5,  # More aggressive column sampling
            reg_alpha=6.0,     # Higher L1 regularization
            reg_lambda=12.0,   # Higher L2 regularization
            min_child_weight=15,  # Much higher minimum child weight
            gamma=0.8,         # Higher gamma for pruning
            random_state=123,
            tree_method="hist",
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=10,
            validation_fraction=0.2
        )
        
        self.mi_rf = RandomForestClassifier(
            n_estimators=30,   # Drastically reduced from 100
            max_depth=4,       # Reduced from 8
            min_samples_split=20,  # Much higher minimum samples
            min_samples_leaf=10,   # Much higher minimum leaf samples
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            bootstrap=True,
            oob_score=True,
            max_samples=0.7    # Limit samples per tree
        )
        
        self.mi_gb = GradientBoostingClassifier(
            n_estimators=40,   # Reduced from 100
            learning_rate=0.02,  # Much lower learning rate
            max_depth=3,       # Reduced from 5
            subsample=0.5,     # More aggressive subsampling
            random_state=42,
            validation_fraction=0.3,  # Larger validation set
            n_iter_no_change=5,       # Earlier stopping
            min_samples_split=20,     # Higher minimum samples
            min_samples_leaf=10       # Higher minimum leaf samples
        )
        
        self.mi_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.5)  # Higher shrinkage
        
        # Simplified MI models to prevent overfitting
        self.mi_mlp = MLPClassifier(
            hidden_layer_sizes=(50, 25),  # Much smaller network
            activation="relu",
            solver="adam",
            alpha=0.1,         # Much higher regularization
            learning_rate="adaptive",
            max_iter=200,      # Fewer iterations
            random_state=42,
            early_stopping=True,
            validation_fraction=0.3,  # Larger validation set
            n_iter_no_change=10
        )
        
        self.mi_ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=20),  # Simpler base learner
            n_estimators=25,   # Reduced from 100
            learning_rate=0.3, # Lower learning rate
            random_state=42
        )
        
        self.mi_et = ExtraTreesClassifier(
            n_estimators=25,   # Reduced from 100
            max_depth=4,       # Reduced from 8
            min_samples_split=15,  # Higher minimum samples
            min_samples_leaf=8,    # Higher minimum leaf samples
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
            max_samples=0.6    # Limit samples per tree
        )
        
        # Conservative SVM for MI
        self.mi_svm = SVC(
            kernel="rbf",
            C=0.5,             # Much lower C for regularization
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        )
        
        self.mi_knn = KNeighborsClassifier(
            n_neighbors=15,    # Higher K for smoother decision boundary
            weights="distance",
            metric="minkowski",
            p=2
        )
        
        # Ultra-conservative MI ensemble
        self.mi_ensemble = StackingClassifier(
            estimators=[
                ("xgb1", self.mi_xgb1),
                ("rf", self.mi_rf),
                ("lda", self.mi_lda),
                ("svm", self.mi_svm)
            ],
            final_estimator=LogisticRegression(
                random_state=42, max_iter=1000, 
                C=0.1,  # Very low C for high regularization
                solver="liblinear",
                penalty="l2",
                class_weight="balanced"
            ),
            cv=5,
            n_jobs=-1,
            passthrough=False  # Don't pass original features
        )
        
        # IMPROVED SSVEP Models with better regularization
        self.ssvep_xgb1 = XGBClassifier(
            n_estimators=100,  # Moderate complexity
            learning_rate=0.08,  # Reasonable learning rate
            max_depth=6,       # Moderate depth
            subsample=0.8,     # Good subsampling
            colsample_bytree=0.8,
            reg_alpha=1.0,     # Moderate regularization
            reg_lambda=2.0,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            tree_method="hist",
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
        
        self.ssvep_xgb2 = XGBClassifier(
            n_estimators=120,
            learning_rate=0.06,
            max_depth=7,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=1.5,
            reg_lambda=2.5,
            min_child_weight=2,
            gamma=0.05,
            random_state=123,
            tree_method="hist",
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
        
        self.ssvep_rf = RandomForestClassifier(
            n_estimators=150,  # Good complexity for SSVEP
            max_depth=10,      # Deeper trees for SSVEP complexity
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            bootstrap=True,
            oob_score=True
        )
        
        self.ssvep_gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        self.ssvep_knn = KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2
        )
        
        # Enhanced SVM models for SSVEP
        self.ssvep_svm1 = SVC(
            kernel="rbf",
            C=5.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        )
        
        self.ssvep_svm2 = SVC(
            kernel="poly",
            C=1.0,
            gamma="auto",
            degree=3,
            class_weight="balanced",
            probability=True,
            random_state=123
        )
        
        self.ssvep_svm3 = SVC(
            kernel="sigmoid",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=456
        )
        
        self.ssvep_mlp = MLPClassifier(
            hidden_layer_sizes=(150, 100, 50),  # Larger network for SSVEP complexity
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate="adaptive",
            max_iter=800,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25
        )
        
        self.ssvep_ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=6),
            n_estimators=120,
            learning_rate=0.08,
            random_state=42
        )
        
        self.ssvep_et = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
        
        self.ssvep_nb = GaussianNB()
        
        self.ssvep_ridge = RidgeClassifier(
            alpha=1.0,
            random_state=42
        )
        
        self.ssvep_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        
        # Balanced ensemble for SSVEP
        self.ssvep_ensemble = StackingClassifier(
            estimators=[
                ("xgb1", self.ssvep_xgb1),
                ("xgb2", self.ssvep_xgb2),
                ("rf", self.ssvep_rf),
                ("gb", self.ssvep_gb),
                ("lda", self.ssvep_lda),
                ("svm1", self.ssvep_svm1),
                ("mlp", self.ssvep_mlp),
                ("et", self.ssvep_et)
            ],
            final_estimator=LogisticRegression(
                random_state=42, max_iter=2000, 
                C=2.0,  # Moderate regularization for SSVEP
                solver="liblinear",
                class_weight="balanced"
            ),
            cv=5,
            n_jobs=-1,
            passthrough=True
        )
        
        # Riemannian classifier for SSVEP
        self.ssvep_riemannian_classifier = None
        
        # SSVEP frequencies for CCA
        self.ssvep_freqs = [7, 8, 10, 13]  # Hz
        self.cca_threshold = 0.25  # Will be tuned
        
    def apply_notch_filter(self, data, freq=50):
        """Apply notch filter to remove power line noise"""
        if not hasattr(self, "fs"):
            self.fs = 250  # Default sampling frequency
        b, a = signal.butter(4, [freq-2, freq+2], btype="bandstop", fs=self.fs)
        return signal.filtfilt(b, a, data, axis=0)
    
    def apply_bandpass_filter(self, data, low_freq, high_freq):
        """Apply bandpass filter"""
        if not hasattr(self, "fs"):
            self.fs = 250  # Default sampling frequency
        b, a = signal.butter(4, [low_freq, high_freq], btype="band", fs=self.fs)
        return signal.filtfilt(b, a, data, axis=0)
    
    def wavelet_decomposition(self, data, wavelet="db4", levels=4):
        """Perform wavelet decomposition for time-frequency analysis"""
        coeffs = pywt.wavedec(data, wavelet, level=levels)
        return coeffs
    
    def extract_wavelet_features(self, signal_data):
        """Extract wavelet-based features"""
        features = []
        coeffs = self.wavelet_decomposition(signal_data)
        
        for i, coeff in enumerate(coeffs):
            if len(coeff) > 0:
                features.append(np.mean(coeff))
                features.append(np.std(coeff))
                features.append(np.median(coeff))
                features.append(entropy(np.abs(coeff)))
        
        return features
    
    def apply_ica_artifact_removal(self, eeg_data):
        """Apply ICA for artifact removal (dummy implementation)"""
        # In a real scenario, this would involve fitting ICA and removing components
        # For now, return data as is or with a simple transformation
        # This is a placeholder for actual ICA implementation
        print("Applying dummy ICA artifact removal.")
        # Example: just return a slightly modified version of the data
        return eeg_data * 0.99 # Simulate some change

    def extract_mi_features(self, eeg_data):
        """Ultra-aggressive MI feature extraction with maximum feature engineering"""
        # Focus on motor cortex channels (C3, C4) - indices 1 and 3
        mi_channels = [1, 3]  # C3, C4
        eeg_mi = eeg_data[:, mi_channels]
        
        # Enhanced preprocessing with artifact removal
        eeg_mi = self.apply_notch_filter(eeg_mi)  # Remove power line noise
        eeg_mi = eeg_mi - np.median(eeg_mi, axis=0)  # More robust than mean
        
        # Apply ICA for artifact removal
        try:
            eeg_mi = self.apply_ica_artifact_removal(eeg_mi)
        except:
            pass
        
        # Define comprehensive frequency bands for MI
        bands = {
            "delta": (1, 4),      # Delta (1-4 Hz)
            "theta": (4, 8),      # Theta (4-8 Hz)
            "mu": (8, 13),        # Mu rhythm (8-13 Hz)
            "beta": (13, 30),     # Beta rhythm (13-30 Hz)
            "low_gamma": (30, 45), # Low gamma (30-45 Hz)
            "high_gamma": (45, 60), # High gamma (45-60 Hz)
            "ultra_gamma": (60, 80) # Ultra gamma (60-80 Hz)
        }
        
        features = []
        
        # Enhanced CSP-like spatial filtering with multiple approaches
        cov_c3 = np.cov(eeg_mi[:, 0])
        cov_c4 = np.cov(eeg_mi[:, 1])
        spatial_ratio = cov_c3 / (cov_c4 + 1e-8)
        features.append(spatial_ratio)
        
        # Multiple spatial correlation measures
        corr_matrix = np.corrcoef(eeg_mi.T)
        features.extend([corr_matrix[0, 0], corr_matrix[1, 1], corr_matrix[0, 1]])
        
        # Cross-correlation at different lags
        for lag in [1, 2, 5, 10]:
            if len(eeg_mi) > lag:
                cross_corr_lag = np.corrcoef(eeg_mi[:-lag, 0], eeg_mi[lag:, 1])[0, 1]
                features.append(cross_corr_lag if not np.isnan(cross_corr_lag) else 0)
            else:
                features.append(0)
        
        # Enhanced asymmetry features with multiple measures
        for band_name, (low_freq, high_freq) in bands.items():
            band_data = self.apply_bandpass_filter(eeg_mi, low_freq, high_freq)
            c3_power = np.var(band_data[:, 0])
            c4_power = np.var(band_data[:, 1])
            
            # Multiple asymmetry measures
            asymmetry = (c3_power - c4_power) / (c3_power + c4_power + 1e-8)
            features.append(asymmetry)
            
            # Log ratio asymmetry
            log_asymmetry = np.log(c3_power + 1e-8) - np.log(c4_power + 1e-8)
            features.append(log_asymmetry)
            
            # Power ratio
            power_ratio = c3_power / (c4_power + 1e-8)
            features.append(power_ratio)
            
            # Absolute power difference
            power_diff = abs(c3_power - c4_power)
            features.append(power_diff)
            
            # Normalized power difference
            norm_power_diff = power_diff / (c3_power + c4_power + 1e-8)
            features.append(norm_power_diff)
            
            # Enhanced band-specific features for each channel
            for ch in range(band_data.shape[1]):
                channel_data = band_data[:, ch]
                
                # Comprehensive time domain features
                features.append(np.mean(channel_data))
                features.append(np.std(channel_data))
                features.append(np.median(channel_data))
                features.append(skew(channel_data))
                features.append(kurtosis(channel_data))
                features.append(np.max(channel_data))
                features.append(np.min(channel_data))
                features.append(np.ptp(channel_data))  # Peak-to-peak
                features.append(np.percentile(channel_data, 25))
                features.append(np.percentile(channel_data, 75))
                features.append(np.percentile(channel_data, 90))
                features.append(np.percentile(channel_data, 95))
                features.append(np.percentile(channel_data, 99))
                
                # Advanced statistical features
                features.append(np.var(channel_data))
                features.append(np.mean(np.abs(channel_data)))
                features.append(np.mean(channel_data**2))
                features.append(np.mean(channel_data**3))
                features.append(np.mean(channel_data**4))
                
                # Zero crossing rate
                zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
                features.append(zero_crossings)
                
                # Root mean square
                rms = np.sqrt(np.mean(channel_data**2))
                features.append(rms)
                
                # Crest factor
                crest_factor = np.max(np.abs(channel_data)) / (rms + 1e-8)
                features.append(crest_factor)
                
                # Shape factor
                shape_factor = rms / (np.mean(np.abs(channel_data)) + 1e-8)
                features.append(shape_factor)
                
                # Impulse factor
                impulse_factor = np.max(np.abs(channel_data)) / (np.mean(np.abs(channel_data)) + 1e-8)
                features.append(impulse_factor)
                
                # Margin factor
                margin_factor = np.max(np.abs(channel_data)) / (np.mean(np.sqrt(np.abs(channel_data))) + 1e-8)**2
                features.append(margin_factor)
                
                # Enhanced frequency domain features
                fft_vals = np.abs(fft(channel_data))
                freqs = fftfreq(len(channel_data), 1/self.fs)
                
                # Power in specific bands with multiple measures
                mu_mask = (freqs >= 8) & (freqs <= 13)
                beta_mask = (freqs >= 13) & (freqs <= 30)
                gamma_mask = (freqs >= 30) & (freqs <= 45)
                high_gamma_mask = (freqs >= 45) & (freqs <= 60)
                ultra_gamma_mask = (freqs >= 60) & (freqs <= 80)
                
                mu_power = np.sum(fft_vals[mu_mask]) if np.any(mu_mask) else 0
                beta_power = np.sum(fft_vals[beta_mask]) if np.any(beta_mask) else 0
                gamma_power = np.sum(fft_vals[gamma_mask]) if np.any(gamma_mask) else 0
                high_gamma_power = np.sum(fft_vals[high_gamma_mask]) if np.any(high_gamma_mask) else 0
                ultra_gamma_power = np.sum(fft_vals[ultra_gamma_mask]) if np.any(ultra_gamma_mask) else 0
                
                features.extend([mu_power, beta_power, gamma_power, high_gamma_power, ultra_gamma_power])
                
                # Power ratios
                total_power = np.sum(fft_vals)
                if total_power > 0:
                    features.extend([
                        mu_power / total_power,
                        beta_power / total_power,
                        gamma_power / total_power,
                        high_gamma_power / total_power,
                        ultra_gamma_power / total_power
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
                
                # Enhanced spectral features
                if np.sum(fft_vals) > 0:
                    spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
                else:
                    spectral_centroid = 0
                features.append(spectral_centroid)
                
                # Spectral rolloff
                cumulative_power = np.cumsum(fft_vals)
                rolloff_threshold = 0.85 * cumulative_power[-1]
                rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
                features.append(spectral_rolloff)
                
                # Spectral bandwidth
                if np.sum(fft_vals) > 0:
                    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_vals) / np.sum(fft_vals))
                else:
                    spectral_bandwidth = 0
                features.append(spectral_bandwidth)
                
                # Spectral flatness
                if np.sum(fft_vals) > 0:
                    spectral_flatness = np.exp(np.mean(np.log(fft_vals + 1e-8))) / (np.mean(fft_vals) + 1e-8)
                else:
                    spectral_flatness = 0
                features.append(spectral_flatness)
                
                # Spectral contrast
                if len(fft_vals) > 10:
                    spectral_contrast = np.std(fft_vals) / (np.mean(fft_vals) + 1e-8)
                else:
                    spectral_contrast = 0
                features.append(spectral_contrast)
                
                # Enhanced wavelet features
                wavelet_features = self.extract_wavelet_features(channel_data)
                features.extend(wavelet_features)
                
                # Advanced entropy features
                features.append(entropy(np.abs(channel_data)))
                
                # Sample entropy (simplified)
                def sample_entropy(data, m=2, r=0.2):
                    """Calculate sample entropy"""
                    try:
                        N = len(data)
                        if N < m + 2:
                            return 0
                        
                        # Normalize data
                        data = (data - np.mean(data)) / np.std(data)
                        r = r * np.std(data)
                        
                        # Count matches
                        A = 0  # m+1 point matches
                        B = 0  # m point matches
                        
                        for i in range(N - m):
                            for j in range(i + 1, N - m):
                                # Check m-point match
                                if np.all(np.abs(data[i:i+m] - data[j:j+m]) < r):
                                    B += 1
                                    # Check m+1-point match
                                    if np.abs(data[i+m] - data[j+m]) < r:
                                        A += 1
                        
                        if B == 0:
                            return 0
                        
                        return -np.log(A / B) if A > 0 else 0
                    except:
                        return 0
                
                features.append(sample_entropy(channel_data))
                
                # Approximate entropy (simplified)
                def approximate_entropy(data, m=2, r=0.2):
                    """Calculate approximate entropy"""
                    try:
                        N = len(data)
                        if N < m + 2:
                            return 0
                        
                        # Normalize data
                        data = (data - np.mean(data)) / np.std(data)
                        r = r * np.std(data)
                        
                        def phi(m_val):
                            count = 0
                            for i in range(N - m_val + 1):
                                for j in range(i + 1, N - m_val + 1):
                                    if np.all(np.abs(data[i:i+m_val] - data[j:j+m_val]) < r):
                                        count += 1
                            return count / (N - m_val + 1)
                        
                        phi_m = phi(m)
                        phi_m1 = phi(m + 1)
                        
                        return phi_m - phi_m1
                    except:
                        return 0
                
                features.append(approximate_entropy(channel_data))
        
                # Higuchi fractal dimension (simplified)
                def higuchi_fractal_dimension(data, k_max=8):
                    """Calculate Higuchi fractal dimension"""
                    try:
                        N = len(data)
                        if N < k_max:
                            k_max = N - 1
                        
                        L = []
                        for k in range(1, k_max + 1):
                            Lk = 0
                            for m in range(k):
                                # Calculate L(m,k)
                                sum_val = 0
                                for i in range(1, int((N - m) / k)):
                                    sum_val += abs(data[m + i * k] - data[m + (i - 1) * k])
                                Lk += sum_val * (N - 1) / (k * k * int((N - m) / k))
                            L.append(np.log(Lk))
                        
                        # Linear regression to find the slope
                        x = np.log(np.array(range(1, k_max + 1)))
                        y = np.array(L)
                        
                        # Handle potential issues with linear regression on small/problematic data
                        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                            coeffs = np.polyfit(x, y, 1)
                            return coeffs[0]
                        else:
                            return 0
                    except:
                        return 0
                
                features.append(higuchi_fractal_dimension(channel_data))

        return np.array(features)

    def apply_cca(self, eeg_data, target_freqs, harmonics=2):
        """Apply CCA for SSVEP feature extraction"""
        cca_features = []
        for freq in target_freqs:
            reference_signals = []
            t = np.arange(eeg_data.shape[0]) / self.fs
            for h in range(1, harmonics + 1):
                reference_signals.append(np.sin(2 * np.pi * h * freq * t))
                reference_signals.append(np.cos(2 * np.pi * h * freq * t))
            
            Y = np.array(reference_signals).T
            
            if eeg_data.shape[0] > eeg_data.shape[1] and Y.shape[0] > Y.shape[1]:
                cca = CCA(n_components=1)
                try:
                    cca.fit(eeg_data, Y)
                    X_c, Y_c = cca.transform(eeg_data, Y)
                    # Correlation between the canonical variates
                    correlation = np.corrcoef(X_c.T, Y_c.T)[0, 1]
                    cca_features.append(correlation)
                except Exception as e:
                    print(f"CCA failed for frequency {freq}: {e}")
                    cca_features.append(0) # Append 0 if CCA fails
            else:
                cca_features.append(0) # Not enough samples for CCA
        return np.array(cca_features)

    def extract_ssvep_features(self, eeg_data):
        """Extract SSVEP features"""
        # Apply notch filter
        eeg_data = self.apply_notch_filter(eeg_data)
        
        features = []
        
        # CCA features
        cca_feats = self.apply_cca(eeg_data, self.ssvep_freqs)
        features.extend(cca_feats)
        
        # Frequency domain features for all channels
        for ch in range(eeg_data.shape[1]):
            channel_data = eeg_data[:, ch]
            fft_vals = np.abs(fft(channel_data))
            freqs = fftfreq(len(channel_data), 1/self.fs)
            
            for freq_band in self.ssvep_freqs:
                # Power at fundamental frequency
                idx_fund = np.argmin(np.abs(freqs - freq_band))
                features.append(fft_vals[idx_fund])
                
                # Power at 2nd harmonic
                idx_harm2 = np.argmin(np.abs(freqs - 2 * freq_band))
                features.append(fft_vals[idx_harm2])
                
                # Power at 3rd harmonic
                idx_harm3 = np.argmin(np.abs(freqs - 3 * freq_band))
                features.append(fft_vals[idx_harm3])
                
                # SNR at fundamental frequency (simplified)
                # This is a very basic SNR. More advanced methods would use surrounding bins.
                power_fund = fft_vals[idx_fund]
                noise_power = (np.sum(fft_vals[ (freqs >= freq_band - 1) & (freqs <= freq_band + 1) & (freqs != freq_band) ]) + 1e-8)
                features.append(power_fund / noise_power)

            # Wavelet features
            wavelet_features = self.extract_wavelet_features(channel_data)
            features.extend(wavelet_features)
            
        return np.array(features)

    def train_mi_model(self, X_mi, y_mi):
        print("Training MI model...")
        # Encode labels
        y_mi_encoded = self.mi_label_encoder.fit_transform(y_mi)
        
        # Scale features
        X_mi_scaled = self.mi_scaler.fit_transform(X_mi)
        
        # Train the ensemble model
        self.mi_ensemble.fit(X_mi_scaled, y_mi_encoded)
        self.mi_model = self.mi_ensemble
        print("MI model training complete.")

    def train_ssvep_model(self, X_ssvep, y_ssvep):
        print("Training SSVEP model...")
        # Encode labels
        y_ssvep_encoded = self.ssvep_label_encoder.fit_transform(y_ssvep)
        
        # Scale features
        X_ssvep_scaled = self.ssvep_scaler.fit_transform(X_ssvep)
        
        # Train the ensemble model
        self.ssvep_ensemble.fit(X_ssvep_scaled, y_ssvep_encoded)
        self.ssvep_model = self.ssvep_ensemble
        print("SSVEP model training complete.")

    def predict_mi(self, eeg_data):
        features = self.extract_mi_features(eeg_data)
        features_scaled = self.mi_scaler.transform(features.reshape(1, -1))
        prediction_encoded = self.mi_model.predict(features_scaled)
        return self.mi_label_encoder.inverse_transform(prediction_encoded)[0]

    def predict_ssvep(self, eeg_data):
        features = self.extract_ssvep_features(eeg_data)
        features_scaled = self.ssvep_scaler.transform(features.reshape(1, -1))
        prediction_encoded = self.ssvep_model.predict(features_scaled)
        return self.ssvep_label_encoder.inverse_transform(prediction_encoded)[0]

    def save_models(self, mi_path="mi_model.joblib", ssvep_path="ssvep_model.joblib",
                    mi_scaler_path="mi_scaler.joblib", ssvep_scaler_path="ssvep_scaler.joblib",
                    mi_encoder_path="mi_encoder.joblib", ssvep_encoder_path="ssvep_encoder.joblib"):
        print("Saving models and preprocessors...")
        if self.mi_model: joblib.dump(self.mi_model, mi_path)
        if self.ssvep_model: joblib.dump(self.ssvep_model, ssvep_path)
        if self.mi_scaler: joblib.dump(self.mi_scaler, mi_scaler_path)
        if self.ssvep_scaler: joblib.dump(self.ssvep_scaler, ssvep_scaler_path)
        if self.mi_label_encoder: joblib.dump(self.mi_label_encoder, mi_encoder_path)
        if self.ssvep_label_encoder: joblib.dump(self.ssvep_label_encoder, ssvep_encoder_path)
        print("Models and preprocessors saved.")

    def load_models(self, mi_path="mi_model.joblib", ssvep_path="ssvep_model.joblib",
                    mi_scaler_path="mi_scaler.joblib", ssvep_scaler_path="ssvep_scaler.joblib",
                    mi_encoder_path="mi_encoder.joblib", ssvep_encoder_path="ssvep_encoder.joblib"):
        print("Loading models and preprocessors...")
        if os.path.exists(mi_path): self.mi_model = joblib.load(mi_path)
        if os.path.exists(ssvep_path): self.ssvep_model = joblib.load(ssvep_path)
        if os.path.exists(mi_scaler_path): self.mi_scaler = joblib.load(mi_scaler_path)
        if os.path.exists(ssvep_scaler_path): self.ssvep_scaler = joblib.load(ssvep_scaler_path)
        if os.path.exists(mi_encoder_path): self.mi_label_encoder = joblib.load(mi_encoder_path)
        if os.path.exists(ssvep_encoder_path): self.ssvep_label_encoder = joblib.load(ssvep_encoder_path)
        print("Models and preprocessors loaded.")

def generate_dummy_eeg_data(num_samples=1, num_channels=19, duration_seconds=2, fs=250, task_type='mi'):
    """Generates dummy EEG data for MI or SSVEP tasks."""
    print(f"Generating dummy {task_type} EEG data...")
    n_timepoints = duration_seconds * fs
    eeg_data = np.random.randn(n_timepoints, num_channels) * 10 # Simulate EEG noise

    if task_type == 'mi':
        # Simulate mu rhythm (8-13 Hz) activity for MI
        # For 'Left' MI, enhance C3 (channel 8) activity
        # For 'Right' MI, enhance C4 (channel 10) activity
        # C3 is index 8, C4 is index 10 in the channel_names list
        mi_channels_idx = {'Left': 8, 'Right': 10}
        labels = np.random.choice(['Left', 'Right'], num_samples)
        
        all_eeg_data = []
        all_labels = []

        for i in range(num_samples):
            current_eeg = np.random.randn(n_timepoints, num_channels) * 10
            current_label = labels[i]
            
            # Add a simulated mu rhythm component
            t = np.arange(n_timepoints) / fs
            mu_freq = np.random.uniform(9, 12) # Random mu frequency
            mu_wave = np.sin(2 * np.pi * mu_freq * t) * 50 # Amplitude
            
            if current_label == 'Left':
                current_eeg[:, mi_channels_idx['Left']] += mu_wave
            elif current_label == 'Right':
                current_eeg[:, mi_channels_idx['Right']] += mu_wave
            
            all_eeg_data.append(current_eeg)
            all_labels.append(current_label)

        return np.array(all_eeg_data), np.array(all_labels)

    elif task_type == 'ssvep':
        # Simulate SSVEP responses for different frequencies
        ssvep_freqs = [7, 8, 10, 13] # Corresponding to self.ssvep_freqs
        labels = np.random.choice(['Forward', 'Backward', 'Left', 'Right'], num_samples)
        
        all_eeg_data = []
        all_labels = []

        for i in range(num_samples):
            current_eeg = np.random.randn(n_timepoints, num_channels) * 10
            current_label = labels[i]
            
            # Add a simulated SSVEP component based on label
            t = np.arange(n_timepoints) / fs
            if current_label == 'Forward':
                ssvep_wave = np.sin(2 * np.pi * ssvep_freqs[0] * t) * 70 # 7 Hz
            elif current_label == 'Backward':
                ssvep_wave = np.sin(2 * np.pi * ssvep_freqs[1] * t) * 70 # 8 Hz
            elif current_label == 'Left':
                ssvep_wave = np.sin(2 * np.pi * ssvep_freqs[2] * t) * 70 # 10 Hz
            elif current_label == 'Right':
                ssvep_wave = np.sin(2 * np.pi * ssvep_freqs[3] * t) * 70 # 13 Hz
            
            # Apply SSVEP to occipital channels (O1, O2) - indices 17, 18
            current_eeg[:, 17] += ssvep_wave
            current_eeg[:, 18] += ssvep_wave
            
            all_eeg_data.append(current_eeg)
            all_labels.append(current_label)

        return np.array(all_eeg_data), np.array(all_labels)
    else:
        raise ValueError("task_type must be 'mi' or 'ssvep'")

def inference_main():
    # Create an instance of the classifier
    bci_classifier = EnhancedBCIClassifier()

    # Load the trained models and preprocessors
    try:
        bci_classifier.load_models()
    except FileNotFoundError:
        print("Error: Models not found. Please run train.py first to train and save the models.")
        return

    # --- MI Inference ---
    print("\n--- Starting MI Inference ---")
    # Generate a single dummy MI data sample for inference
    mi_eeg_sample, mi_true_label = generate_dummy_eeg_data(num_samples=1, duration_seconds=3, task_type='mi')
    mi_eeg_sample = mi_eeg_sample[0] # Get the single sample
    mi_true_label = mi_true_label[0]

    # Predict MI
    mi_prediction = bci_classifier.predict_mi(mi_eeg_sample)
    print(f"MI True Label: {mi_true_label}, Predicted Label: {mi_prediction}")

    # --- SSVEP Inference ---
    print("\n--- Starting SSVEP Inference ---")
    # Generate a single dummy SSVEP data sample for inference
    ssvep_eeg_sample, ssvep_true_label = generate_dummy_eeg_data(num_samples=1, duration_seconds=3, task_type='ssvep')
    ssvep_eeg_sample = ssvep_eeg_sample[0] # Get the single sample
    ssvep_true_label = ssvep_true_label[0]

    # Predict SSVEP
    ssvep_prediction = bci_classifier.predict_ssvep(ssvep_eeg_sample)
    print(f"SSVEP True Label: {ssvep_true_label}, Predicted Label: {ssvep_prediction}")

if __name__ == "__main__":
    inference_main()


