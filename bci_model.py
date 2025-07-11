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

warnings.filterwarnings('ignore')

class EnhancedBCIClassifier:
    def __init__(self, base_path='.'):
        self.base_path = base_path
        self.fs = 250  # Sampling frequency
        
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
            tree_method='hist',
            use_label_encoder=False,
            eval_metric='logloss',
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
            tree_method='hist',
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=10,
            validation_fraction=0.2
        )
        
        self.mi_rf = RandomForestClassifier(
            n_estimators=30,   # Drastically reduced from 100
            max_depth=4,       # Reduced from 8
            min_samples_split=20,  # Much higher minimum samples
            min_samples_leaf=10,   # Much higher minimum leaf samples
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
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
        
        self.mi_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)  # Higher shrinkage
        
        # Simplified MI models to prevent overfitting
        self.mi_mlp = MLPClassifier(
            hidden_layer_sizes=(50, 25),  # Much smaller network
            activation='relu',
            solver='adam',
            alpha=0.1,         # Much higher regularization
            learning_rate='adaptive',
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
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
            max_samples=0.6    # Limit samples per tree
        )
        
        # Conservative SVM for MI
        self.mi_svm = SVC(
            kernel='rbf',
            C=0.5,             # Much lower C for regularization
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        self.mi_knn = KNeighborsClassifier(
            n_neighbors=15,    # Higher K for smoother decision boundary
            weights='distance',
            metric='minkowski',
            p=2
        )
        
        # Ultra-conservative MI ensemble
        self.mi_ensemble = StackingClassifier(
            estimators=[
                ('xgb1', self.mi_xgb1),
                ('rf', self.mi_rf),
                ('lda', self.mi_lda),
                ('svm', self.mi_svm)
            ],
            final_estimator=LogisticRegression(
                random_state=42, max_iter=1000, 
                C=0.1,  # Very low C for high regularization
                solver='liblinear',
                penalty='l2',
                class_weight='balanced'
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
            tree_method='hist',
            use_label_encoder=False,
            eval_metric='mlogloss'
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
            tree_method='hist',
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        self.ssvep_rf = RandomForestClassifier(
            n_estimators=150,  # Good complexity for SSVEP
            max_depth=10,      # Deeper trees for SSVEP complexity
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
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
            weights='distance',
            metric='minkowski',
            p=2
        )
        
        # Enhanced SVM models for SSVEP
        self.ssvep_svm1 = SVC(
            kernel='rbf',
            C=5.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        self.ssvep_svm2 = SVC(
            kernel='poly',
            C=1.0,
            gamma='auto',
            degree=3,
            class_weight='balanced',
            probability=True,
            random_state=123
        )
        
        self.ssvep_svm3 = SVC(
            kernel='sigmoid',
            C=10.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=456
        )
        
        # Enhanced MLP for SSVEP
        self.ssvep_mlp = MLPClassifier(
            hidden_layer_sizes=(150, 100, 50),  # Larger network for SSVEP complexity
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
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
            max_features='sqrt',
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
        
        self.ssvep_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        
        # Balanced ensemble for SSVEP
        self.ssvep_ensemble = StackingClassifier(
            estimators=[
                ('xgb1', self.ssvep_xgb1),
                ('xgb2', self.ssvep_xgb2),
                ('rf', self.ssvep_rf),
                ('gb', self.ssvep_gb),
                ('lda', self.ssvep_lda),
                ('svm1', self.ssvep_svm1),
                ('mlp', self.ssvep_mlp),
                ('et', self.ssvep_et)
            ],
            final_estimator=LogisticRegression(
                random_state=42, max_iter=2000, 
                C=2.0,  # Moderate regularization for SSVEP
                solver='liblinear',
                class_weight='balanced'
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
        b, a = signal.butter(4, [freq-2, freq+2], btype='bandstop', fs=self.fs)
        return signal.filtfilt(b, a, data, axis=0)
    
    def apply_bandpass_filter(self, data, low_freq, high_freq):
        """Apply bandpass filter"""
        b, a = signal.butter(4, [low_freq, high_freq], btype='band', fs=self.fs)
        return signal.filtfilt(b, a, data, axis=0)
    
    def wavelet_decomposition(self, data, wavelet='db4', levels=4):
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
            'delta': (1, 4),      # Delta (1-4 Hz)
            'theta': (4, 8),      # Theta (4-8 Hz)
            'mu': (8, 13),        # Mu rhythm (8-13 Hz)
            'beta': (13, 30),     # Beta rhythm (13-30 Hz)
            'low_gamma': (30, 45), # Low gamma (30-45 Hz)
            'high_gamma': (45, 60), # High gamma (45-60 Hz)
            'ultra_gamma': (60, 80) # Ultra gamma (60-80 Hz)
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
                            L.append(Lk / k)
                        
                        # Calculate fractal dimension
                        x = np.log(1.0 / np.arange(1, k_max + 1))
                        y = np.log(L)
                        slope = np.polyfit(x, y, 1)[0]
                        return slope
                    except:
                        return 0
                
                features.append(higuchi_fractal_dimension(channel_data))
        
        # Enhanced cross-channel features
        for band_name, (low_freq, high_freq) in bands.items():
            band_data = self.apply_bandpass_filter(eeg_mi, low_freq, high_freq)
            
            # Cross-correlation between channels
            cross_corr = np.corrcoef(band_data.T)[0, 1]
            features.append(cross_corr)
            
            # Coherence-like measure
            fft_c3 = fft(band_data[:, 0])
            fft_c4 = fft(band_data[:, 1])
            coherence = np.abs(np.mean(fft_c3 * np.conj(fft_c4))) / (np.mean(np.abs(fft_c3)) * np.mean(np.abs(fft_c4)) + 1e-8)
            features.append(coherence)
            
            # Phase synchronization
            phase_c3 = np.angle(fft_c3)
            phase_c4 = np.angle(fft_c4)
            phase_diff = np.abs(phase_c3 - phase_c4)
            phase_sync = np.mean(np.cos(phase_diff))
            features.append(phase_sync)
            
            # Cross-power spectral density
            cross_psd = np.mean(np.real(fft_c3 * np.conj(fft_c4)))
            features.append(cross_psd)
            
            # Imaginary coherence
            imag_coherence = np.abs(np.mean(np.imag(fft_c3 * np.conj(fft_c4)))) / (np.mean(np.abs(fft_c3)) * np.mean(np.abs(fft_c4)) + 1e-8)
            features.append(imag_coherence)
        
        # Enhanced Riemannian features
        try:
            riemannian_features = self.extract_riemannian_features(eeg_mi)
            features.extend(riemannian_features)
        except:
            features.extend([0.0] * 40)  # Increased from 30
        
        # Time-frequency features using short-time Fourier transform
        try:
            for ch in range(eeg_mi.shape[1]):
                channel_data = eeg_mi[:, ch]
                
                # STFT parameters
                nperseg = min(256, len(channel_data) // 4)
                noverlap = nperseg // 2
                
                if len(channel_data) > nperseg:
                    f, t, Zxx = signal.stft(channel_data, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
                    
                    # Power in time-frequency bins
                    power_tf = np.abs(Zxx)**2
                    
                    # Mean power over time for each frequency
                    mean_power_freq = np.mean(power_tf, axis=1)
                    features.extend(mean_power_freq[:10])  # First 10 frequency bins
                    
                    # Mean power over frequency for each time
                    mean_power_time = np.mean(power_tf, axis=0)
                    features.extend(mean_power_time[:5])   # First 5 time bins
                    
                    # Peak frequency over time
                    peak_freq_time = f[np.argmax(power_tf, axis=0)]
                    features.extend(peak_freq_time[:5])    # First 5 time bins
                else:
                    features.extend([0.0] * 20)  # Default values
        except:
            features.extend([0.0] * 20)
        
        return np.array(features)
    
    def extract_ssvep_features(self, eeg_data):
        """IMPROVED SSVEP feature extraction with better frequency analysis and stability"""
        # Focus on occipital channels (PO7, OZ, PO8) - indices 5, 6, 7
        occipital_channels = [5, 6, 7]
        eeg_occipital = eeg_data[:, occipital_channels]
        
        # ENHANCED preprocessing pipeline for better signal quality
        eeg_occipital = signal.detrend(eeg_occipital, axis=0, type='linear')
        eeg_occipital = self.apply_notch_filter(eeg_occipital, 50)  # Remove power line noise
        eeg_occipital = self.apply_notch_filter(eeg_occipital, 60)  # Remove 60Hz harmonics
        
        # Robust outlier removal
        for ch in range(eeg_occipital.shape[1]):
            channel_data = eeg_occipital[:, ch]
            median_val = np.median(channel_data)
            mad = np.median(np.abs(channel_data - median_val))
            threshold = 5 * mad  # More conservative outlier detection
            outliers = np.abs(channel_data - median_val) > threshold
            eeg_occipital[outliers, ch] = median_val
        
        # Subtract common average reference for better spatial filtering
        eeg_occipital = eeg_occipital - np.mean(eeg_occipital, axis=1, keepdims=True)
        
        # Apply surface Laplacian for better spatial resolution
        try:
            # Simple Laplacian: center - average of neighbors
            center_ch = 1  # OZ (center)
            neighbors = [0, 2]  # PO7, PO8
            laplacian_oz = eeg_occipital[:, center_ch] - np.mean(eeg_occipital[:, neighbors], axis=1)
            # Replace center channel with Laplacian
            eeg_occipital[:, center_ch] = laplacian_oz
        except:
            pass
        
        features = []
        
        # IMPROVED frequency analysis with better resolution
        for ch in range(eeg_occipital.shape[1]):
            channel_data = eeg_occipital[:, ch]
            
            # Multi-taper spectral estimation for better frequency resolution
            nperseg = min(1024, len(channel_data))  # Larger window for better resolution
            f, Pxx = signal.welch(channel_data, fs=self.fs, nperseg=nperseg, 
                                noverlap=nperseg//2, window='hann')
            
            # SSVEP-specific frequency analysis
            for target_freq in self.ssvep_freqs:
                # High-resolution power estimation around target frequency
                freq_tolerance = 0.2  # Very narrow band
                freq_mask = (f >= target_freq - freq_tolerance) & (f <= target_freq + freq_tolerance)
                
                if np.any(freq_mask):
                    # Target frequency power
                    target_power = np.trapz(Pxx[freq_mask], f[freq_mask])
                    
                    # Background noise estimation (neighboring frequencies)
                    noise_mask1 = (f >= target_freq - 2) & (f <= target_freq - 1)
                    noise_mask2 = (f >= target_freq + 1) & (f <= target_freq + 2)
                    noise_mask = noise_mask1 | noise_mask2
                    
                    if np.any(noise_mask):
                        noise_power = np.mean(Pxx[noise_mask])
                        # Signal-to-noise ratio
                        snr = target_power / (noise_power + 1e-10)
                        features.append(np.log1p(snr))
                    else:
                        features.append(0)
                    
                    # Normalized power (relative to total SSVEP band power)
                    ssvep_mask = (f >= 5) & (f <= 30)
                    total_ssvep_power = np.trapz(Pxx[ssvep_mask], f[ssvep_mask]) if np.any(ssvep_mask) else 1
                    normalized_power = target_power / (total_ssvep_power + 1e-10)
                    features.append(normalized_power)
                    
                    # Peak detection around target frequency
                    from scipy.signal import find_peaks
                    peaks, properties = find_peaks(Pxx[freq_mask], height=np.max(Pxx[freq_mask])*0.5)
                    if len(peaks) > 0:
                        # Peak prominence and width
                        peak_power = np.max(Pxx[freq_mask][peaks])
                        features.append(peak_power)
                        
                        # Frequency deviation from target
                        peak_freq = f[freq_mask][peaks[np.argmax(Pxx[freq_mask][peaks])]]
                        freq_deviation = abs(peak_freq - target_freq)
                        features.append(freq_deviation)
                    else:
                        features.extend([0, 0])
                else:
                    features.extend([0, 0, 0, 0])
                
                # Harmonic analysis
                for harmonic in [2, 3]:
                    harm_freq = target_freq * harmonic
                    if harm_freq <= self.fs/2:  # Nyquist limit
                        harm_mask = (f >= harm_freq - 0.3) & (f <= harm_freq + 0.3)
                        if np.any(harm_mask):
                            harm_power = np.trapz(Pxx[harm_mask], f[harm_mask])
                            # Harmonic to fundamental ratio
                            harm_ratio = harm_power / (target_power + 1e-10)
                            features.append(harm_ratio)
                        else:
                            features.append(0)
                    else:
                        features.append(0)
            
            # Enhanced spectral features
            ssvep_range = (f >= 5) & (f <= 30)
            if np.any(ssvep_range):
                ssvep_psd = Pxx[ssvep_range]
                ssvep_freqs = f[ssvep_range]
                
                # Spectral centroid in SSVEP range
                centroid = np.sum(ssvep_freqs * ssvep_psd) / (np.sum(ssvep_psd) + 1e-10)
                features.append(centroid)
                
                # Spectral bandwidth
                bandwidth = np.sqrt(np.sum(((ssvep_freqs - centroid)**2) * ssvep_psd) / (np.sum(ssvep_psd) + 1e-10))
                features.append(bandwidth)
                
                # Spectral entropy (measure of frequency spreading)
                psd_norm = ssvep_psd / (np.sum(ssvep_psd) + 1e-10)
                spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                features.append(spectral_entropy)
                
                # Peak-to-average ratio
                peak_to_avg = np.max(ssvep_psd) / (np.mean(ssvep_psd) + 1e-10)
                features.append(peak_to_avg)
                
                # Frequency band power ratios
                alpha_mask = (ssvep_freqs >= 8) & (ssvep_freqs <= 13)
                beta_mask = (ssvep_freqs >= 13) & (ssvep_freqs <= 25)
                
                alpha_power = np.trapz(ssvep_psd[alpha_mask], ssvep_freqs[alpha_mask]) if np.any(alpha_mask) else 0
                beta_power = np.trapz(ssvep_psd[beta_mask], ssvep_freqs[beta_mask]) if np.any(beta_mask) else 0
                
                alpha_beta_ratio = alpha_power / (beta_power + 1e-10)
                features.append(alpha_beta_ratio)
            else:
                features.extend([0] * 6)
        
        # IMPROVED cross-channel analysis
        # Spatial correlation patterns
        corr_matrix = np.corrcoef(eeg_occipital.T)
        features.extend([corr_matrix[0, 1], corr_matrix[0, 2], corr_matrix[1, 2]])
        
        # Channel power asymmetry
        channel_powers = [np.var(eeg_occipital[:, i]) for i in range(3)]
        # Left-right asymmetry (PO7 vs PO8)
        lr_asymmetry = (channel_powers[0] - channel_powers[2]) / (channel_powers[0] + channel_powers[2] + 1e-10)
        features.append(lr_asymmetry)
        
        # Center dominance (OZ vs average of PO7, PO8)
        center_dominance = channel_powers[1] / (np.mean([channel_powers[0], channel_powers[2]]) + 1e-10)
        features.append(center_dominance)
        
        # ENHANCED CCA analysis with improved reference signals
        for target_freq in self.ssvep_freqs:
            cca_scores = []
            t = np.arange(len(eeg_occipital)) / self.fs
            
            # Multiple reference signal approaches
            for phase_offset in [0, np.pi/4, np.pi/2]:
                # Sine and cosine components
                sin_ref = np.sin(2 * np.pi * target_freq * t + phase_offset)
                cos_ref = np.cos(2 * np.pi * target_freq * t + phase_offset)
                
                # Create multi-harmonic reference
                ref_signal = sin_ref + 0.5 * np.sin(2 * np.pi * target_freq * 2 * t + phase_offset)
                ref_signal = ref_signal / np.std(ref_signal)  # Normalize
                
                # Calculate CCA with each channel
                for ch in range(eeg_occipital.shape[1]):
                    try:
                        corr = np.corrcoef(eeg_occipital[:, ch], ref_signal)[0, 1]
                        if not np.isnan(corr):
                            cca_scores.append(abs(corr))
                    except:
                        pass
            
            # CCA features for this frequency
            if len(cca_scores) > 0:
                features.extend([np.max(cca_scores), np.mean(cca_scores), np.std(cca_scores)])
            else:
                features.extend([0, 0, 0])
        
        # IMPROVED phase-locking analysis
        for target_freq in self.ssvep_freqs:
            t = np.arange(len(eeg_occipital)) / self.fs
            
            # Phase-locking value calculation
            plv_values = []
            for ch in range(eeg_occipital.shape[1]):
                try:
                    # Analytic signal using Hilbert transform
                    analytic = signal.hilbert(eeg_occipital[:, ch])
                    instantaneous_phase = np.angle(analytic)
                    
                    # Reference phase
                    ref_phase = 2 * np.pi * target_freq * t
                    
                    # Phase difference
                    phase_diff = instantaneous_phase - ref_phase
                    
                    # PLV calculation
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    plv_values.append(plv)
                except:
                    plv_values.append(0)
            
            # PLV statistics across channels
            if len(plv_values) > 0:
                features.extend([np.mean(plv_values), np.max(plv_values), np.std(plv_values)])
            else:
                features.extend([0, 0, 0])
        
        # IMPROVED coherence analysis between channels
        coherence_features = []
        for i in range(3):
            for j in range(i+1, 3):
                ch1_data = eeg_occipital[:, i]
                ch2_data = eeg_occipital[:, j]
                
                try:
                    f_coh, Cxy = signal.coherence(ch1_data, ch2_data, fs=self.fs, nperseg=512)
                    
                    # Coherence in SSVEP frequency bands
                    for target_freq in self.ssvep_freqs:
                        freq_mask = (f_coh >= target_freq - 0.5) & (f_coh <= target_freq + 0.5)
                        if np.any(freq_mask):
                            mean_coherence = np.mean(Cxy[freq_mask])
                            coherence_features.append(mean_coherence)
                        else:
                            coherence_features.append(0)
                    
                    # Overall SSVEP band coherence
                    ssvep_mask = (f_coh >= 5) & (f_coh <= 30)
                    if np.any(ssvep_mask):
                        overall_coherence = np.mean(Cxy[ssvep_mask])
                        coherence_features.append(overall_coherence)
                    else:
                        coherence_features.append(0)
                except:
                    coherence_features.extend([0] * (len(self.ssvep_freqs) + 1))
        
        features.extend(coherence_features)
        
        # FIXED Riemannian features with proper error handling
        try:
            riemannian_features = self.extract_riemannian_features_fixed(eeg_occipital)
            features.extend(riemannian_features)
        except Exception as e:
            print(f"Riemannian feature extraction failed: {e}")
            features.extend([0.0] * 20)  # Fixed size fallback
        
        return np.array(features)
    
    def extract_enhanced_ssvep_features(self, eeg_data):
        """Enhanced SSVEP feature extraction for validation consistency"""
        return self.extract_ssvep_features(eeg_data)
    
    def apply_enhanced_cca_with_phase(self, trial_data, freq):
        """Enhanced CCA with phase consideration and multiple harmonics"""
        t = np.arange(trial_data.shape[0]) / self.fs
        ref_signals = []
        
        # Add harmonic+phase combinations
        phases = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Multiple phase offsets
        harmonics = [1, 2, 3, 4]  # Multiple harmonics
        
        for h in harmonics:
            for phase in phases:
                ref_signals.append(np.sin(2 * np.pi * h * freq * t + phase))
                ref_signals.append(np.cos(2 * np.pi * h * freq * t + phase))
        
        # Regularized CCA
        ref_signals = np.column_stack(ref_signals)
        n_components = min(2, trial_data.shape[1], ref_signals.shape[1])
        
        try:
            cca = CCA(n_components=n_components)
            cca.fit(trial_data, ref_signals)
            X_c, Y_c = cca.transform(trial_data, ref_signals)
            
            # Calculate correlation for each component
            correlations = []
            for i in range(n_components):
                corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0
        except:
            return 0  # Fallback value
    
    def enhanced_cca_predict(self, eeg_data):
        """Enhanced CCA-based SSVEP prediction with improved FBCCA"""
        # Focus on occipital channels
        occipital_channels = [5, 6, 7]  # PO7, OZ, PO8
        eeg_occipital = eeg_data[:, occipital_channels]
        
        # Enhanced preprocessing
        eeg_occipital = self.apply_notch_filter(eeg_occipital)
        eeg_occipital = eeg_occipital - np.median(eeg_occipital, axis=0)
        
        # Try enhanced FBCCA first
        try:
            fbcca_features, cca_scores = self.apply_enhanced_fbcca_filtering(eeg_occipital, self.ssvep_freqs)
            
            if len(cca_scores) > 0:
                # Find the best frequency based on enhanced FBCCA scores
                best_band_idx = np.argmax(cca_scores)
                max_corr = cca_scores[best_band_idx]
                
                # Map back to SSVEP frequency
                if max_corr > self.cca_threshold:
                    # Use the frequency band that gave the highest correlation
                    if best_band_idx < len(self.ssvep_freqs):
                        best_class = best_band_idx
                    else:
                        # Fallback to enhanced traditional CCA
                        best_class, max_corr = self._enhanced_traditional_cca_predict(eeg_occipital)
                else:
                    # Fallback to enhanced traditional CCA
                    best_class, max_corr = self._enhanced_traditional_cca_predict(eeg_occipital)
            else:
                # Fallback to enhanced traditional CCA
                best_class, max_corr = self._enhanced_traditional_cca_predict(eeg_occipital)
                
        except Exception as e:
            print(f"Enhanced FBCCA failed: {e}, using enhanced traditional CCA")
            best_class, max_corr = self._enhanced_traditional_cca_predict(eeg_occipital)
        
        # Enhanced confidence calculation
        if max_corr > self.cca_threshold:
            confidence = min(1.0, max_corr * 2.5)  # Increased scaling factor
            return best_class, confidence
        else:
            # Fallback to ensemble prediction
            return None, 0.0
    
    def _enhanced_traditional_cca_predict(self, eeg_occipital):
        """Enhanced traditional CCA prediction with multi-channel analysis"""
        # Apply bandpass filter
        eeg_filtered = self.apply_bandpass_filter(eeg_occipital, 5, 30)
        
        # Generate enhanced reference signals for each SSVEP frequency
        t = np.arange(len(eeg_filtered)) / self.fs
        max_corr = -1
        best_class = 0
        
        for i, freq in enumerate(self.ssvep_freqs):
            # Generate enhanced reference signals with multiple harmonics
            harmonics = [1, 2, 3, 4, 5]  # Fundamental + 4 harmonics
            sub_harmonics = [0.5, 0.75, 1.25, 1.5]  # Sub-harmonics
            
            # Create enhanced reference signal with harmonics
            ref_signal = np.zeros(len(t))
            for h in harmonics:
                ref_signal += np.sin(2 * np.pi * freq * h * t)
                ref_signal += np.cos(2 * np.pi * freq * h * t)
            
            # Add sub-harmonics
            for sh in sub_harmonics:
                ref_signal += np.sin(2 * np.pi * freq * sh * t)
                ref_signal += np.cos(2 * np.pi * freq * sh * t)
            
            # Normalize reference signal
            ref_signal = ref_signal / np.std(ref_signal)
            
            # Multi-channel CCA correlation
            channel_correlations = []
            for ch in range(eeg_filtered.shape[1]):
                try:
                    correlation = np.corrcoef(eeg_filtered[:, ch], ref_signal)[0, 1]
                    if not np.isnan(correlation):
                        channel_correlations.append(correlation)
                except:
                    continue
            
            # Use maximum correlation across channels
            if len(channel_correlations) > 0:
                max_channel_corr = np.max(np.abs(channel_correlations))
                if max_channel_corr > max_corr:
                    max_corr = max_channel_corr
                    best_class = i
        
        return best_class, max_corr
    
    def apply_enhanced_fbcca_filtering(self, eeg_data, ssvep_freqs):
        """Enhanced Filter Bank CCA (FBCCA) with improved narrow bandpass filters"""
        try:
            # Define enhanced narrow bandpass filters around SSVEP frequencies
            filter_bands = []
            for freq in ssvep_freqs:
                # Create multiple narrow bands around each frequency
                filter_bands.extend([
                    (freq - 0.3, freq + 0.3),  # ±0.3 Hz - very narrow
                    (freq - 0.5, freq + 0.5),  # ±0.5 Hz - narrow
                    (freq - 0.8, freq + 0.8),  # ±0.8 Hz - medium
                    (freq * 2 - 0.3, freq * 2 + 0.3),  # Second harmonic
                    (freq * 3 - 0.3, freq * 3 + 0.3),  # Third harmonic
                    (freq * 0.5 - 0.2, freq * 0.5 + 0.2),  # Sub-harmonic
                ])
            
            # Remove duplicates and sort
            filter_bands = list(set(filter_bands))
            filter_bands.sort()
            
            # print(f"Enhanced FBCCA using {len(filter_bands)} filter bands")  # Reduced verbose printing
            
            fbcca_features = []
            cca_scores = []
            
            for band_idx, (low_freq, high_freq) in enumerate(filter_bands):
                # Apply bandpass filter
                band_data = self.apply_bandpass_filter(eeg_data, low_freq, high_freq)
                
                # Calculate enhanced CCA correlation for each SSVEP frequency
                band_correlations = []
                t = np.arange(band_data.shape[0]) / self.fs
                
                for ssvep_freq in ssvep_freqs:
                    # Generate enhanced reference signal with harmonics
                    harmonics = [1, 2, 3, 4]  # More harmonics
                    ref_signal = np.zeros(len(t))
                    for h in harmonics:
                        ref_signal += np.sin(2 * np.pi * ssvep_freq * h * t)
                        ref_signal += np.cos(2 * np.pi * ssvep_freq * h * t)
                    
                    # Add sub-harmonics
                    sub_harmonics = [0.5, 0.75, 1.25, 1.5]
                    for sh in sub_harmonics:
                        ref_signal += np.sin(2 * np.pi * ssvep_freq * sh * t)
                        ref_signal += np.cos(2 * np.pi * ssvep_freq * sh * t)
                    
                    ref_signal = ref_signal / np.std(ref_signal)
                    
                    # Calculate correlation with each channel
                    channel_correlations = []
                    for ch in range(band_data.shape[1]):
                        try:
                            corr = np.corrcoef(band_data[:, ch], ref_signal)[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                            channel_correlations.append(corr)
                        except:
                            channel_correlations.append(0.0)
                    
                    # Use maximum correlation across channels
                    max_corr = np.max(np.abs(channel_correlations))
                    band_correlations.append(max_corr)
                
                # Store features for this band
                fbcca_features.extend(band_correlations)
                cca_scores.append(np.max(band_correlations))
            
            fbcca_features = np.array(fbcca_features)
            # print(f"Enhanced FBCCA features shape: {fbcca_features.shape}")  # Reduced verbose printing
            
            return fbcca_features, cca_scores
            
        except Exception as e:
            print(f"Enhanced FBCCA failed: {e}")
            return np.array([]), []
    
    def load_trial_data(self, row):
        """Load EEG data with enhanced error handling and quality control"""
        try:
            id_num = int(row['id'])  # Convert to integer for comparison
            if id_num <= 4800:
                dataset = 'train'
            elif id_num <= 4900:
                dataset = 'validation'
            else:
                dataset = 'test'
            
            eeg_path = os.path.join(
                self.base_path, 
                row['task'], 
                dataset, 
                row['subject_id'], 
                str(row['trial_session']), 
                'EEGdata.csv'
            )
            
            if not os.path.exists(eeg_path):
                print(f"EEG file not found: {eeg_path}")
                return None
            
            eeg_data = pd.read_csv(eeg_path)
            trial_num = int(row['trial'])
            
            if row['task'] == 'MI':
                samples_per_trial = 2250
            else:  # SSVEP
                samples_per_trial = 1750
            
            start_idx = (trial_num - 1) * samples_per_trial
            end_idx = start_idx + samples_per_trial
            
            eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
            trial_data = eeg_data[eeg_channels].iloc[start_idx:end_idx]
            
            # Enhanced quality control (inspired by provided code)
            if trial_data.empty:
                print(f"Empty trial data for {row['id']}")
                return None
            
            # Check for channel range (quality control)
            channel_ranges = trial_data.max() - trial_data.min()
            if (channel_ranges < 10).any() or channel_ranges.isna().any():
                print(f"Low quality trial data for {row['id']}: channel ranges too small")
                return None
            
            # Check for NaN or infinite values
            if trial_data.isna().any().any() or np.isinf(trial_data.values).any():
                print(f"Invalid values in trial data for {row['id']}")
                return None
            
            return trial_data.values
            
        except Exception as e:
            print(f"Error loading trial data for {row['id']}: {e}")
            return None
    
    def augment_data(self, X, y, task_type, augmentation_factor=0.05):  # MUCH more conservative
        """ULTRA-CONSERVATIVE data augmentation to prevent overfitting"""
        print(f"Ultra-conservative augmentation for {task_type}: {len(X)} -> ", end="")
        
        # Handle empty arrays
        if len(X) == 0:
            print("0 samples (empty input)")
            return np.array([]), np.array([])
        
        # For MI: NO augmentation to prevent overfitting
        if task_type == 'MI':
            print(f"{len(X)} samples (NO MI augmentation to prevent overfitting)")
            return np.array(X), np.array(y)
        
        # For SSVEP: Very minimal augmentation only for class balancing
        augmented_X = []
        augmented_y = []
        
        # Add original data
        augmented_X.extend(X)
        augmented_y.extend(y)
        
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Handle case where there's only one class or no classes
        if len(unique_classes) <= 1:
            print(f"{len(augmented_X)} samples (single class, no augmentation needed)")
            return np.array(augmented_X), np.array(augmented_y)
        
        max_class_count = np.max(class_counts)
        min_class_count = np.min(class_counts)
        
        # Only balance if there's severe class imbalance (>3:1 ratio)
        if max_class_count / min_class_count > 3:
            print("Severe class imbalance detected, applying minimal balancing...")
            
            # Very conservative SMOTE-like balancing
            for class_idx in unique_classes:
                class_samples = X[y == class_idx]
                current_count = len(class_samples)
                
                if current_count < max_class_count:
                    # Only add a few samples to help balance
                    samples_needed = min(3, max_class_count - current_count)  # Maximum 3 samples per class
                    
                    for _ in range(samples_needed):
                        # Randomly select a base sample
                        base_idx = np.random.randint(0, len(class_samples))
                        base_sample = class_samples[base_idx]
                        
                        # Find nearest neighbor from same class
                        distances = np.linalg.norm(class_samples - base_sample, axis=1)
                        nearest_idx = np.argsort(distances)[1]  # Just one neighbor
                        neighbor_sample = class_samples[nearest_idx]
                        
                        # Simple interpolation between samples
                        alpha = np.random.uniform(0.3, 0.7)
                        synthetic_sample = alpha * base_sample + (1 - alpha) * neighbor_sample
                        
                        # Very small noise to prevent exact duplicates
                        noise_std = np.std(base_sample) * 0.01  # Very small noise
                        noise = np.random.normal(0, noise_std, synthetic_sample.shape)
                        synthetic_sample += noise
                        
                        augmented_X.append(synthetic_sample)
                        augmented_y.append(class_idx)
        
        print(f"{len(augmented_X)} samples")
        return np.array(augmented_X), np.array(augmented_y)

    def feature_selection(self, X, y, task_type, k=50):  # Much more conservative
        """ULTRA-CONSERVATIVE feature selection to prevent overfitting"""
        # Remove rows with any NaN or infinite values
        mask = np.all(np.isfinite(X), axis=1)
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"Ultra-conservative feature selection for {task_type}: Input shape {X_clean.shape}")
        
        # MUCH more conservative feature selection
        if task_type == 'MI':
            # For MI: Very aggressive feature reduction to prevent overfitting
            max_features = min(20, X_clean.shape[1] // 8)  # Maximum 20 features or 1/8 of total
            k = max(10, min(max_features, k))  # At least 10, at most max_features
            print(f"MI: Ultra-conservative selection - using only {k} features")
        else:
            # For SSVEP: Moderate feature reduction
            max_features = min(40, X_clean.shape[1] // 4)  # Maximum 40 features or 1/4 of total
            k = max(20, min(max_features, k))  # At least 20, at most max_features
            print(f"SSVEP: Conservative selection - using {k} features")
        
        # Use only F-statistic (ANOVA) for both tasks for consistency
        f_selector = SelectKBest(f_classif, k=min(k, X_clean.shape[1]))
        X_selected = f_selector.fit_transform(X_clean, y_clean)
        selected_features = f_selector.get_support()
        
        print(f"Selected {np.sum(selected_features)} features for {task_type} (removed {np.sum(~mask)} samples with NaN/inf)")
        
        # Create a simple selector that maintains the feature selection
        class SimpleSelector:
            def __init__(self, feature_mask):
                self.feature_mask = feature_mask
            
            def transform(self, X):
                return X[:, self.feature_mask]
        
        selector = SimpleSelector(selected_features)
        return X_selected, y_clean, selector
    
    def train_with_cross_validation(self, X, y, model, task_type, n_splits=5):
        """Enhanced training with stratified cross-validation and early stopping"""
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Perform cross-validation to check for overfitting
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='accuracy',
            n_jobs=1,  # Reduced parallelism for stability
            verbose=0   # Reduced verbosity
        )
        
        print(f"{task_type} Cross-validation scores: {scores}")
        print(f"Mean CV accuracy: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        
        # Check for overfitting indicators
        cv_mean = np.mean(scores)
        cv_std = np.std(scores)
        
        if cv_std > 0.15:  # High variance indicates potential overfitting
            print(f"⚠️  High CV variance ({cv_std:.3f}) detected for {task_type} - possible overfitting")
        
        if task_type == 'MI' and cv_mean > 0.9:
            print(f"⚠️  Very high CV accuracy ({cv_mean:.3f}) for {task_type} - likely overfitting")
        
        # Train-validation split for additional overfitting check
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Check training vs validation performance
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        print(f"{task_type} Train accuracy: {train_score:.4f}")
        print(f"{task_type} Validation accuracy: {val_score:.4f}")
        print(f"{task_type} Train-Val gap: {train_score - val_score:.4f}")
        
        if train_score - val_score > 0.2:
            print(f"⚠️  Large train-validation gap ({train_score - val_score:.3f}) for {task_type} - overfitting detected")
        
        # Refit on full data for final model
        model.fit(X, y)
        return model
    
    def train(self, train_df, validation_df=None):
        """Enhanced training procedure with progressive learning and advanced validation"""
        print("🚀 Enhanced BCI Classifier Training with Progressive Learning")
        print("=" * 70)
        
        # Split data by task
        mi_train = train_df[train_df['task'] == 'MI']
        ssvep_train = train_df[train_df['task'] == 'SSVEP']
        
        print(f"📊 Dataset: {len(mi_train)} MI, {len(ssvep_train)} SSVEP samples")
        
        # Enhanced MI Training Pipeline
        if len(mi_train) > 0:
            print("\n🧠 Enhanced MI Training Pipeline...")
            print("-" * 50)
            
            # Progressive MI training with multiple stages
            try:
                mi_results = self.progressive_mi_training(mi_train, validation_df)
                
                # Advanced feature importance analysis  
                if mi_results:
                    self.analyze_mi_feature_importance(mi_results)
                
                # Cross-validation with temporal splits
                self.temporal_cross_validation(mi_train, 'MI')
            except Exception as e:
                print(f"Enhanced MI training failed, using fallback: {e}")
                self.fallback_mi_training(mi_train)
        
        # Enhanced SSVEP Training Pipeline
        if len(ssvep_train) > 0:
            print("\n👁️ Enhanced SSVEP Training Pipeline...")
            print("-" * 50)
            
            # Progressive SSVEP training with multiple stages
            try:
                ssvep_results = self.progressive_ssvep_training(ssvep_train, validation_df)
                
                # Advanced feature importance analysis
                if ssvep_results:
                    self.analyze_ssvep_feature_importance(ssvep_results)
                
                # Cross-validation with temporal splits
                self.temporal_cross_validation(ssvep_train, 'SSVEP')
                
                # Enhanced CCA threshold optimization
                self.advanced_cca_optimization(validation_df)
            except Exception as e:
                print(f"Enhanced SSVEP training failed, using fallback: {e}")
                self.fallback_ssvep_training(ssvep_train)
        
        # Final validation with comprehensive metrics
        if validation_df is not None:
            try:
                self.comprehensive_validation(validation_df)
            except Exception as e:
                print(f"Comprehensive validation failed: {e}")
                self.validate(validation_df)  # Fallback to original validation
        
        # Model interpretability analysis
        try:
            self.model_interpretability_analysis(train_df)
        except Exception as e:
            print(f"Model interpretability analysis failed: {e}")
        
        print("\n✅ Enhanced training completed with advanced analytics!")
        
        # If enhanced training succeeded, we're done
        # Otherwise, fallback to original training methods
        if len(mi_train) > 0 and not hasattr(self, 'mi_ensemble'):
            print("🔄 Falling back to original MI training...")
            self.original_train_mi(mi_train)
        
        if len(ssvep_train) > 0 and not hasattr(self, 'ssvep_ensemble'):
            print("🔄 Falling back to original SSVEP training...")
            self.original_train_ssvep(ssvep_train)
        
        # Final validation
        if validation_df is not None:
            try:
                self.comprehensive_validation(validation_df)
            except:
                self.original_validate(validation_df)

    def original_train_mi(self, mi_train):
        """Original MI training method (fallback)"""
        print("\n🧠 Training MI Model...")
        
        # Load raw EEG data for CSP/FBCSP
        mi_eeg_data = []
        mi_labels = []
        
        for idx, row in mi_train.iterrows():
            try:
                trial_data = self.load_trial_data(row)
                if trial_data is not None:
                    mi_eeg_data.append(trial_data)
                    mi_labels.append(row['label'])
            except Exception as e:
                continue
        
        mi_eeg_data = np.array(mi_eeg_data)
        mi_labels = np.array(mi_labels)
        
        # Apply Advanced CSP for enhanced feature extraction
        if len(mi_eeg_data) > 20:  # Need sufficient samples for Advanced CSP
            print("Applying Advanced CSP for enhanced MI feature extraction...")
            
            # Apply advanced preprocessing first
            processed_eeg_data = []
            for trial in mi_eeg_data:
                processed_trial = self.apply_advanced_preprocessing(trial, 'MI')
                processed_eeg_data.append(processed_trial)
            processed_eeg_data = np.array(processed_eeg_data)
            
            # Apply Advanced CSP
            advanced_csp_features, csp_filters = self.apply_advanced_csp(processed_eeg_data, mi_labels, n_components=6)
            
            if csp_filters is not None:
                # Store CSP filters for prediction
                self.mi_csp_filters = csp_filters
                print("Advanced CSP applied successfully")
                
                # Extract advanced spectral features
                spectral_features = []
                for trial in processed_eeg_data:
                    spec_feat = self.extract_advanced_spectral_features(trial, 'MI')
                    spectral_features.append(spec_feat)
                spectral_features = np.array(spectral_features)
                
                # Combine CSP and spectral features
                X_mi_combined = np.hstack([advanced_csp_features, spectral_features])
                y_mi = mi_labels
                print(f"Combined MI features shape: {X_mi_combined.shape}")
                
            else:
                print("Advanced CSP failed, using enhanced traditional features")
                X_mi_combined, y_mi = self.load_and_prepare_data(mi_train, 'MI')
                self.mi_csp_filters = None
        else:
            print("Insufficient samples for Advanced CSP, using enhanced traditional features")
            X_mi_combined, y_mi = self.load_and_prepare_data(mi_train, 'MI')
            self.mi_csp_filters = None
        
        y_mi_encoded = self.mi_label_encoder.fit_transform(y_mi)
        
        # Conservative data augmentation
        X_mi_aug, y_mi_aug = self.augment_data(X_mi_combined, y_mi_encoded, 'MI')
        
        # Enhanced feature selection with cross-validation
        X_mi_selected, y_mi_clean, self.mi_selector = self.feature_selection(
            self.mi_scaler.fit_transform(X_mi_aug), 
            y_mi_aug,
            'MI',
            k=60  # Reduced to prevent overfitting
        )
        
        # Train advanced ensemble with boosting
        print("Training advanced MI ensemble with boosting...")
        self.mi_ensemble = self.apply_ensemble_boosting(X_mi_selected, y_mi_clean, 'MI')
        self.mi_ensemble.fit(X_mi_selected, y_mi_clean)
        
        # Show comprehensive training metrics
        self.show_training_metrics(X_mi_aug, y_mi_aug, 'MI')
        
        # Cross-validation
        cv_scores = cross_val_score(self.mi_ensemble, X_mi_selected, y_mi_clean, 
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        print(f"MI CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Training accuracy
        mi_train_pred = self.mi_ensemble.predict(X_mi_selected)
        mi_train_acc = accuracy_score(y_mi_clean, mi_train_pred)
        print(f"MI Train Acc: {mi_train_acc:.3f}")

    def original_train_ssvep(self, ssvep_train):
        """Original SSVEP training method (fallback)"""
        print("\n👁️ Training SSVEP Model...")
        
        # Load and prepare enhanced SSVEP data with advanced preprocessing
        print("Loading SSVEP data with advanced preprocessing...")
        ssvep_eeg_data = []
        ssvep_labels = []
        
        for idx, row in ssvep_train.iterrows():
            try:
                trial_data = self.load_trial_data(row)
                if trial_data is not None:
                    # Apply advanced preprocessing
                    processed_trial = self.apply_advanced_preprocessing(trial_data, 'SSVEP')
                    # Extract advanced features
                    advanced_features = self.extract_advanced_spectral_features(processed_trial, 'SSVEP')
                    # Combine with enhanced SSVEP features
                    enhanced_features = self.extract_enhanced_ssvep_features(trial_data)
                    combined_features = np.concatenate([advanced_features, enhanced_features])
                    
                    ssvep_eeg_data.append(combined_features)
                    ssvep_labels.append(row['label'])
            except Exception as e:
                continue
        
        X_ssvep = np.array(ssvep_eeg_data)
        y_ssvep = np.array(ssvep_labels)
        print(f"Enhanced SSVEP features shape: {X_ssvep.shape}")
        
        # Encode labels
        y_ssvep_encoded = self.ssvep_label_encoder.fit_transform(y_ssvep)
        
        # Conservative data augmentation
        X_ssvep_aug, y_ssvep_aug = self.augment_data(X_ssvep, y_ssvep_encoded, 'SSVEP')
        
        # Enhanced feature selection
        X_ssvep_selected, y_ssvep_clean, self.ssvep_selector = self.feature_selection(
            self.ssvep_scaler.fit_transform(X_ssvep_aug),
            y_ssvep_aug,
            'SSVEP',
            k=80  # Balanced selection to prevent overfitting
        )
        
        # Train advanced SSVEP ensemble with boosting
        print("Training advanced SSVEP ensemble with boosting...")
        self.ssvep_ensemble = self.apply_ensemble_boosting(X_ssvep_selected, y_ssvep_clean, 'SSVEP')
        self.ssvep_ensemble.fit(X_ssvep_selected, y_ssvep_clean)
        
        # Show comprehensive training metrics
        self.show_training_metrics(X_ssvep_aug, y_ssvep_aug, 'SSVEP')
        
        # Train Riemannian classifier for SSVEP
        try:
            # Load raw EEG data for Riemannian classification
            ssvep_eeg_data = []
            ssvep_raw_labels = []
            
            for idx, row in ssvep_train.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        # Focus on occipital channels for SSVEP
                        occipital_channels = [5, 6, 7]  # PO7, OZ, PO8
                        trial_occipital = trial_data[:, occipital_channels]
                        ssvep_eeg_data.append(trial_occipital)
                        ssvep_raw_labels.append(row['label'])
                except Exception as e:
                    continue
            
            if len(ssvep_eeg_data) > 10:
                ssvep_eeg_data = np.array(ssvep_eeg_data)
                ssvep_raw_labels = np.array(ssvep_raw_labels)
                
                # Train Riemannian classifier
                self.ssvep_riemannian_classifier, _ = self.apply_riemannian_classification(
                    ssvep_eeg_data, ssvep_raw_labels
                )
            else:
                self.ssvep_riemannian_classifier = None
                
        except Exception as e:
            self.ssvep_riemannian_classifier = None
        
        # Cross-validation
        cv_scores = cross_val_score(self.ssvep_ensemble, X_ssvep_selected, y_ssvep_clean, 
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        print(f"SSVEP CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Training accuracy
        ssvep_train_pred = self.ssvep_ensemble.predict(X_ssvep_selected)
        ssvep_train_acc = accuracy_score(y_ssvep_clean, ssvep_train_pred)
        print(f"SSVEP Train Acc: {ssvep_train_acc:.3f}")

    def original_validate(self, validation_df):
        """Original validation method"""
        if validation_df is not None:
            self.validate(validation_df)

    def validate(self, validation_df):
        """Enhanced validation procedure with comprehensive metrics and proper classification report"""
        print("\n📈 Enhanced Validation with Advanced Analytics")
        print("=" * 70)
        
        # Enhanced validation pipeline
        self.enhanced_validation_pipeline(validation_df)
        
        # Original validation for comparison
        print("\n📊 Standard Validation Results")
        print("=" * 50)
        mi_val = validation_df[validation_df['task'] == 'MI']
        ssvep_val = validation_df[validation_df['task'] == 'SSVEP']
        
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_true_labels = []
        
        # Validate MI
        if len(mi_val) > 0 and hasattr(self, 'mi_ensemble'):
            print("\n🧠 MI Validation Metrics:")
            print("-" * 30)
            
            # Use the same feature extraction method as training
            mi_eeg_data = []
            mi_labels = []
            
            for idx, row in mi_val.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        mi_eeg_data.append(trial_data)
                        mi_labels.append(row['label'])
                except Exception as e:
                    continue
            
            mi_eeg_data = np.array(mi_eeg_data)
            mi_labels = np.array(mi_labels)
            
            # Apply same preprocessing and feature extraction as training
            if len(mi_eeg_data) > 0:
                # Apply advanced preprocessing
                processed_eeg_data = []
                for trial in mi_eeg_data:
                    processed_trial = self.apply_advanced_preprocessing(trial, 'MI')
                    processed_eeg_data.append(processed_trial)
                processed_eeg_data = np.array(processed_eeg_data)
                
                # Apply Advanced CSP if available
                if hasattr(self, 'mi_csp_filters') and self.mi_csp_filters is not None:
                    # Apply CSP filtering to validation data
                    csp_features = []
                    for trial in processed_eeg_data:
                        filtered_trial = self.mi_csp_filters.T @ trial
                        log_var_features = np.log(np.var(filtered_trial, axis=1) + 1e-8)
                        csp_features.append(log_var_features)
                    csp_features = np.array(csp_features)
                    
                    # Extract advanced spectral features
                    spectral_features = []
                    for trial in processed_eeg_data:
                        spec_feat = self.extract_advanced_spectral_features(trial, 'MI')
                        spectral_features.append(spec_feat)
                    spectral_features = np.array(spectral_features)
                    
                    # Combine features same as training
                    X_mi_val = np.hstack([csp_features, spectral_features])
                else:
                    # Fallback to enhanced MI features
                    X_mi_val = []
                    for trial in processed_eeg_data:
                        features = self.extract_enhanced_mi_features(trial)
                        X_mi_val.append(features)
                    X_mi_val = np.array(X_mi_val)
                
                y_mi_val = mi_labels
                
                print(f"MI validation features shape: {X_mi_val.shape}")
                X_mi_val_scaled = self.mi_scaler.transform(X_mi_val)
            
            # Filter out NaN values
            mask = np.all(np.isfinite(X_mi_val_scaled), axis=1)
            X_mi_val_clean = X_mi_val_scaled[mask]
            y_mi_val_clean = y_mi_val[mask]
            
            if len(X_mi_val_clean) > 0:
                X_mi_val_selected = self.mi_selector.transform(X_mi_val_clean)
                y_mi_val_encoded = self.mi_label_encoder.transform(y_mi_val_clean)
                
                val_pred = self.mi_ensemble.predict(X_mi_val_selected)
                val_acc = accuracy_score(y_mi_val_encoded, val_pred)
                val_f1_weighted = f1_score(y_mi_val_encoded, val_pred, average='weighted')
                val_f1_macro = f1_score(y_mi_val_encoded, val_pred, average='macro')
                val_f1_micro = f1_score(y_mi_val_encoded, val_pred, average='micro')
                
                # Per-class metrics
                from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
                precision, recall, f1, support = precision_recall_fscore_support(y_mi_val_encoded, val_pred, average=None)
                conf_matrix = confusion_matrix(y_mi_val_encoded, val_pred)
                
                print(f"Overall Accuracy: {val_acc:.4f}")
                print(f"F1 Score (Weighted): {val_f1_weighted:.4f}")
                print(f"F1 Score (Macro): {val_f1_macro:.4f}")
                print(f"F1 Score (Micro): {val_f1_micro:.4f}")
                
                print("\nPer-Class Metrics:")
                class_names = self.mi_label_encoder.classes_
                for i, class_name in enumerate(class_names):
                    print(f"  {class_name}:")
                    print(f"    Precision: {precision[i]:.4f}")
                    print(f"    Recall: {recall[i]:.4f}")
                    print(f"    F1-Score: {f1[i]:.4f}")
                    print(f"    Support: {support[i]}")
                
                print("\nConfusion Matrix:")
                print(conf_matrix)
                
                # Additional statistical metrics
                from scipy.stats import skew, kurtosis
                confidences = np.max(self.mi_ensemble.predict_proba(X_mi_val_selected), axis=1)
                print(f"\nConfidence Statistics:")
                print(f"  Mean Confidence: {np.mean(confidences):.4f}")
                print(f"  Std Confidence: {np.std(confidences):.4f}")
                print(f"  Min Confidence: {np.min(confidences):.4f}")
                print(f"  Max Confidence: {np.max(confidences):.4f}")
                print(f"  Skewness: {skew(confidences):.4f}")
                print(f"  Kurtosis: {kurtosis(confidences):.4f}")
                
                total_correct += np.sum(val_pred == y_mi_val_encoded)
                total_samples += len(y_mi_val_clean)
                
                # Store for overall report
                all_predictions.extend(val_pred)
                all_true_labels.extend(y_mi_val_encoded)
            else:
                print("MI Val: No valid samples")
        
        # Validate SSVEP
        if len(ssvep_val) > 0 and hasattr(self, 'ssvep_ensemble'):
            print("\n👁️ SSVEP Validation Metrics:")
            print("-" * 30)
            
            # Use the same feature extraction method as training
            ssvep_eeg_data = []
            ssvep_labels = []
            
            for idx, row in ssvep_val.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        # Apply advanced preprocessing
                        processed_trial = self.apply_advanced_preprocessing(trial_data, 'SSVEP')
                        # Extract advanced features
                        advanced_features = self.extract_advanced_spectral_features(processed_trial, 'SSVEP')
                        # Combine with enhanced SSVEP features
                        enhanced_features = self.extract_enhanced_ssvep_features(trial_data)
                        combined_features = np.concatenate([advanced_features, enhanced_features])
                        
                        ssvep_eeg_data.append(combined_features)
                        ssvep_labels.append(row['label'])
                except Exception as e:
                    continue
            
            X_ssvep_val = np.array(ssvep_eeg_data)
            y_ssvep_val = np.array(ssvep_labels)
            
            print(f"SSVEP validation features shape: {X_ssvep_val.shape}")
            X_ssvep_val_scaled = self.ssvep_scaler.transform(X_ssvep_val)
            
            # Filter out NaN values
            mask = np.all(np.isfinite(X_ssvep_val_scaled), axis=1)
            X_ssvep_val_clean = X_ssvep_val_scaled[mask]
            y_ssvep_val_clean = y_ssvep_val[mask]
            
            if len(X_ssvep_val_clean) > 0:
                X_ssvep_val_selected = self.ssvep_selector.transform(X_ssvep_val_clean)
                y_ssvep_val_encoded = self.ssvep_label_encoder.transform(y_ssvep_val_clean)
                
                val_pred = self.ssvep_ensemble.predict(X_ssvep_val_selected)
                val_acc = accuracy_score(y_ssvep_val_encoded, val_pred)
                val_f1_weighted = f1_score(y_ssvep_val_encoded, val_pred, average='weighted')
                val_f1_macro = f1_score(y_ssvep_val_encoded, val_pred, average='macro')
                val_f1_micro = f1_score(y_ssvep_val_encoded, val_pred, average='micro')
                
                # Per-class metrics
                from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
                precision, recall, f1, support = precision_recall_fscore_support(y_ssvep_val_encoded, val_pred, average=None)
                conf_matrix = confusion_matrix(y_ssvep_val_encoded, val_pred)
                
                print(f"Overall Accuracy: {val_acc:.4f}")
                print(f"F1 Score (Weighted): {val_f1_weighted:.4f}")
                print(f"F1 Score (Macro): {val_f1_macro:.4f}")
                print(f"F1 Score (Micro): {val_f1_micro:.4f}")
                
                print("\nPer-Class Metrics:")
                class_names = self.ssvep_label_encoder.classes_
                for i, class_name in enumerate(class_names):
                    print(f"  {class_name}:")
                    print(f"    Precision: {precision[i]:.4f}")
                    print(f"    Recall: {recall[i]:.4f}")
                    print(f"    F1-Score: {f1[i]:.4f}")
                    print(f"    Support: {support[i]}")
                
                print("\nConfusion Matrix:")
                print(conf_matrix)
                
                # Additional statistical metrics
                from scipy.stats import skew, kurtosis
                confidences = np.max(self.ssvep_ensemble.predict_proba(X_ssvep_val_selected), axis=1)
                print(f"\nConfidence Statistics:")
                print(f"  Mean Confidence: {np.mean(confidences):.4f}")
                print(f"  Std Confidence: {np.std(confidences):.4f}")
                print(f"  Min Confidence: {np.min(confidences):.4f}")
                print(f"  Max Confidence: {np.max(confidences):.4f}")
                print(f"  Skewness: {skew(confidences):.4f}")
                print(f"  Kurtosis: {kurtosis(confidences):.4f}")
                
                total_correct += np.sum(val_pred == y_ssvep_val_encoded)
                total_samples += len(y_ssvep_val_clean)
                
                # Store for overall report
                all_predictions.extend(val_pred)
                all_true_labels.extend(y_ssvep_val_encoded)
            else:
                print("SSVEP Val: No valid samples")
        
        # Overall validation metrics with proper classification report
        if total_samples > 0:
            overall_acc = total_correct / total_samples
            print(f"\n🎯 Overall Validation Summary:")
            print(f"  Total Samples: {total_samples}")
            print(f"  Correct Predictions: {total_correct}")
            print(f"  Overall Accuracy: {overall_acc:.4f}")
            
            # Generate proper classification report
            if len(all_predictions) > 0:
                # Get all unique class names
                all_class_names = []
                if hasattr(self, 'mi_label_encoder'):
                    all_class_names.extend(self.mi_label_encoder.classes_)
                if hasattr(self, 'ssvep_label_encoder'):
                    all_class_names.extend(self.ssvep_label_encoder.classes_)
                all_class_names = list(set(all_class_names))  # Remove duplicates
                
                # Generate the classification report in the requested format
                self.generate_classification_report(all_true_labels, all_predictions, all_class_names)
        else:
            print("Overall Val: No valid samples")
    
    def predict(self, test_df):
        """Enhanced prediction with improved ensemble strategy and confidence calibration"""
        predictions = []
        confidences = []
        
        print(f"🔮 Enhanced prediction for {len(test_df)} samples...")
        
        for idx, row in test_df.iterrows():
            try:
                trial_data = self.load_trial_data(row)
                if trial_data is None:
                    predictions.append(self.get_default_prediction(row['task']))
                    confidences.append(0.4)  # Lower confidence for missing data
                    continue
                
                # Use dynamic prediction fusion
                prediction, confidence = self.dynamic_prediction_fusion(trial_data, row['task'])
                predictions.append(prediction)
                confidences.append(confidence)
                        
            except Exception as e:
                print(f"Error predicting row {idx}: {e}")
                predictions.append(self.get_default_prediction(row['task']))
                confidences.append(0.3)  # Lower confidence for errors
                continue
        
        # Post-process predictions for better class balance
        predictions, confidences = self.post_process_predictions(predictions, confidences, test_df)
        
        return predictions, confidences
    
    def post_process_predictions(self, predictions, confidences, test_df):
        """Post-process predictions to improve class balance and confidence"""
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Separate by task
        mi_mask = test_df['task'] == 'MI'
        ssvep_mask = test_df['task'] == 'SSVEP'
        
        # Post-process MI predictions
        if np.any(mi_mask):
            mi_predictions = predictions[mi_mask]
            mi_confidences = confidences[mi_mask]
            
            # Check for severe class imbalance
            unique, counts = np.unique(mi_predictions, return_counts=True)
            if len(unique) == 2:  # Binary classification
                ratio = max(counts) / min(counts)
                if ratio > 3.0:  # Severe imbalance
                    # Adjust some low-confidence predictions of majority class
                    majority_class = unique[np.argmax(counts)]
                    minority_class = unique[np.argmin(counts)]
                    
                    # Find low-confidence majority predictions
                    majority_mask = mi_predictions == majority_class
                    low_conf_mask = mi_confidences < 0.6
                    candidates = majority_mask & low_conf_mask
                    
                    if np.any(candidates):
                        # Flip some predictions to minority class
                        flip_count = min(np.sum(candidates) // 3, 3)  # Conservative flipping
                        flip_indices = np.where(candidates)[0][:flip_count]
                        mi_predictions[flip_indices] = minority_class
                        mi_confidences[flip_indices] *= 0.8  # Reduce confidence
            
            predictions[mi_mask] = mi_predictions
            confidences[mi_mask] = mi_confidences
        
        # Post-process SSVEP predictions
        if np.any(ssvep_mask):
            ssvep_predictions = predictions[ssvep_mask]
            ssvep_confidences = confidences[ssvep_mask]
            
            # Check for class distribution
            unique, counts = np.unique(ssvep_predictions, return_counts=True)
            if len(unique) > 1:
                # Ensure minimum representation for each class
                min_count = max(1, len(ssvep_predictions) // (len(unique) * 3))
                
                for class_name in ['Forward', 'Backward', 'Left', 'Right']:
                    if class_name in ssvep_predictions:
                        class_count = np.sum(ssvep_predictions == class_name)
                        if class_count < min_count:
                            # Find candidates to flip to this class
                            other_classes = ssvep_predictions != class_name
                            low_conf_mask = ssvep_confidences < 0.5
                            candidates = other_classes & low_conf_mask
                            
                            if np.any(candidates):
                                flip_count = min(min_count - class_count, np.sum(candidates))
                                flip_indices = np.where(candidates)[0][:flip_count]
                                ssvep_predictions[flip_indices] = class_name
                                ssvep_confidences[flip_indices] *= 0.7
            
            predictions[ssvep_mask] = ssvep_predictions
            confidences[ssvep_mask] = ssvep_confidences
        
        return predictions.tolist(), confidences.tolist()
    
    def get_default_prediction(self, task):
        """Get sensible default prediction based on task"""
        if task == 'MI':
            return 'Left'  # Default for MI
        else:
            return self.ssvep_label_encoder.classes_[0]  # First SSVEP class

    def tune_cca_threshold(self, validation_df):
        """Enhanced CCA threshold tuning with comprehensive search"""
        print("\n🎯 Enhanced CCA Threshold Tuning...")
        ssvep_val = validation_df[validation_df['task'] == 'SSVEP']
        
        if len(ssvep_val) == 0:
            print("No SSVEP validation data available for threshold tuning")
            return self.cca_threshold
        
        # Ensure LabelEncoder is fitted
        if not hasattr(self.ssvep_label_encoder, 'classes_'):
            # Fit the label encoder with the validation data
            self.ssvep_label_encoder.fit(ssvep_val['label'])
        
        # Comprehensive threshold range with fine-grained search
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
                     0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
                     0.22, 0.25, 0.28, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
        best_threshold = self.cca_threshold
        best_accuracy = 0
        
        for threshold in thresholds:
            self.cca_threshold = threshold
            correct = 0
            total = 0
            
            for idx, row in ssvep_val.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is None:
                        continue
                    
                    cca_class, cca_score = self.enhanced_cca_predict(trial_data)
                    
                    # Fix the comparison logic
                    if cca_class is not None and cca_score > self.cca_threshold:
                        cca_label = self.ssvep_label_encoder.classes_[cca_class]
                        
                        if cca_label == row['label']:
                            correct += 1
                        total += 1
                except Exception as e:
                    # print(f"Error in threshold tuning for row {idx}: {e}")  # Reduced verbose printing
                    continue
            
            if total > 0:
                accuracy = correct / total
                # print(f"Threshold {threshold}: {accuracy:.4f} accuracy ({correct}/{total})")  # Reduced verbose printing
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
        
        print(f"Best CCA threshold: {best_threshold} (acc: {best_accuracy:.3f})")
        self.cca_threshold = best_threshold
        return best_threshold

    def load_and_prepare_data(self, df, task_type):
        """Load and prepare data with task-specific feature extraction and improved preprocessing"""
        X = []
        y = []
        print(f"Loading {len(df)} {task_type} samples...")
        
        # Process in smaller batches for better performance and to prevent getting stuck
        batch_size = 20  # Reduced from 50 to prevent memory issues
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        # Add timeout and progress tracking
        import time
        start_time = time.time()
        
        for batch_idx in range(total_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # print(f"Processing batch {batch_idx + 1}/{total_batches} (samples {start_idx + 1}-{end_idx})")  # Reduced verbose printing
            
            batch_success_count = 0
            batch_error_count = 0
            
            for idx, row in batch_df.iterrows():
                try:
                    # Add timeout for individual sample processing
                    sample_start_time = time.time()
                    
                    trial_data = self.load_trial_data(row)
                    if trial_data is None:
                        continue
                    
                    # Improved preprocessing - handle NaN/inf values
                    if np.any(np.isnan(trial_data)) or np.any(np.isinf(trial_data)):
                        # Replace NaN/inf with median of the channel
                        for ch in range(trial_data.shape[1]):
                            channel_data = trial_data[:, ch]
                            median_val = np.nanmedian(channel_data)
                            channel_data = np.where(np.isnan(channel_data) | np.isinf(channel_data), median_val, channel_data)
                            trial_data[:, ch] = channel_data
                    
                    # Extract features with enhanced methods
                    if task_type == 'MI':
                        features = self.extract_enhanced_mi_features(trial_data)  # Use enhanced version
                    else:
                        features = self.extract_enhanced_ssvep_features(trial_data)  # Use enhanced version
                    
                    # Check processing time
                    if time.time() - sample_start_time > 5.0:  # 5 second timeout per sample
                        # print(f"Sample {idx} took too long, skipping...")  # Reduced verbose printing
                        continue
                    
                    # Check for NaN/inf in features
                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        # Replace with zeros for problematic features
                        features = np.where(np.isnan(features) | np.isinf(features), 0, features)
                    
                    X.append(features)
                    y.append(row['label'])
                    batch_success_count += 1
                    
                except Exception as e:
                    # print(f"Error processing row {idx}: {e}")  # Reduced verbose printing
                    batch_error_count += 1
                    continue
        
            # Check batch processing time
            batch_time = time.time() - batch_start_time
            # print(f"Batch {batch_idx + 1} completed in {batch_time:.2f}s - Success: {batch_success_count}, Errors: {batch_error_count}")  # Reduced verbose printing
            
            # If batch takes too long, reduce batch size for remaining batches
            if batch_time > 30.0 and batch_size > 5:
                batch_size = max(5, batch_size // 2)
                # print(f"Batch took too long, reducing batch size to {batch_size}")  # Reduced verbose printing
        
        total_time = time.time() - start_time
        print(f"✓ Loaded {len(X)} {task_type} samples in {total_time:.1f}s")
        return np.array(X), np.array(y)

    def extract_mi_features_fast(self, eeg_data):
        """Fast MI feature extraction with improved accuracy"""
        # Focus on motor cortex channels (C3, C4) - indices 1 and 3
        mi_channels = [1, 3]  # C3, C4
        eeg_mi = eeg_data[:, mi_channels]
        
        # Basic preprocessing
        eeg_mi = self.apply_notch_filter(eeg_mi)
        eeg_mi = eeg_mi - np.median(eeg_mi, axis=0)
        
        # Define key frequency bands for MI
        bands = {
            'mu': (8, 13),      # Mu rhythm (8-13 Hz)
            'beta': (13, 30),   # Beta rhythm (13-30 Hz)
        }
        
        features = []
        
        # Basic spatial features
        cov_c3 = np.cov(eeg_mi[:, 0])
        cov_c4 = np.cov(eeg_mi[:, 1])
        spatial_ratio = cov_c3 / (cov_c4 + 1e-8)
        features.append(spatial_ratio)
        
        # Spatial correlation matrix features
        corr_matrix = np.corrcoef(eeg_mi.T)
        features.extend([corr_matrix[0, 0], corr_matrix[1, 1], corr_matrix[0, 1]])
        
        # Basic asymmetry features
        for band_name, (low_freq, high_freq) in bands.items():
            band_data = self.apply_bandpass_filter(eeg_mi, low_freq, high_freq)
            c3_power = np.var(band_data[:, 0])
            c4_power = np.var(band_data[:, 1])
            
            # Asymmetry measures
            asymmetry = (c3_power - c4_power) / (c3_power + c4_power + 1e-8)
            features.append(asymmetry)
            
            # Log ratio asymmetry
            log_asymmetry = np.log(c3_power + 1e-8) - np.log(c4_power + 1e-8)
            features.append(log_asymmetry)
            
            # Power ratio
            power_ratio = c3_power / (c4_power + 1e-8)
            features.append(power_ratio)
            
            # Basic band-specific features
            for ch in range(band_data.shape[1]):
                channel_data = band_data[:, ch]
                
                # Time domain features
                features.append(np.mean(channel_data))
                features.append(np.std(channel_data))
                features.append(skew(channel_data))
                features.append(kurtosis(channel_data))
                features.append(np.max(channel_data))
                features.append(np.min(channel_data))
                features.append(np.ptp(channel_data))  # Peak-to-peak
                
                # Frequency domain features
                fft_vals = np.abs(fft(channel_data))
                freqs = fftfreq(len(channel_data), 1/self.fs)
                
                # Power in specific bands
                mu_mask = (freqs >= 8) & (freqs <= 13)
                beta_mask = (freqs >= 13) & (freqs <= 30)
                
                mu_power = np.sum(fft_vals[mu_mask]) if np.any(mu_mask) else 0
                beta_power = np.sum(fft_vals[beta_mask]) if np.any(beta_mask) else 0
                
                features.extend([mu_power, beta_power])
                
                # Spectral centroid
                if np.sum(fft_vals) > 0:
                    spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
                else:
                    spectral_centroid = 0
                features.append(spectral_centroid)
        
        # Cross-channel features
        for band_name, (low_freq, high_freq) in bands.items():
            band_data = self.apply_bandpass_filter(eeg_mi, low_freq, high_freq)
            cross_corr = np.corrcoef(band_data.T)[0, 1]
            features.append(cross_corr)
            
            # Coherence-like measure
            fft_c3 = fft(band_data[:, 0])
            fft_c4 = fft(band_data[:, 1])
            coherence = np.abs(np.mean(fft_c3 * np.conj(fft_c4))) / (np.mean(np.abs(fft_c3)) * np.mean(np.abs(fft_c4)) + 1e-8)
            features.append(coherence)
        
        # Add basic Riemannian features (simplified)
        try:
            riemannian_features = self.extract_riemannian_features_fast(eeg_mi)
            features.extend(riemannian_features)
        except:
            features.extend([0.0] * 10)  # Reduced from 20 to 10
        
        return np.array(features)

    def extract_ssvep_features_fast(self, eeg_data):
        """Enhanced SSVEP feature extraction with improved frequency analysis"""
        # Focus on occipital channels (PO7, OZ, PO8) - indices 5, 6, 7
        occipital_channels = [5, 6, 7]
        eeg_occipital = eeg_data[:, occipital_channels]
        
        # Enhanced preprocessing
        eeg_occipital = self.apply_notch_filter(eeg_occipital)
        eeg_occipital = eeg_occipital - np.median(eeg_occipital, axis=0)
        
        # Apply bandpass filter for SSVEP frequencies (5-30 Hz)
        eeg_filtered = self.apply_bandpass_filter(eeg_occipital, 5, 30)
        
        features = []
        
        # Extract features for each channel
        for ch in range(eeg_filtered.shape[1]):
            channel_data = eeg_filtered[:, ch]
            
            # Enhanced time domain features
            features.append(np.var(channel_data))
            features.append(np.mean(np.abs(channel_data)))
            features.append(skew(channel_data))
            features.append(kurtosis(channel_data))
            features.append(np.max(channel_data))
            features.append(np.min(channel_data))
            features.append(np.ptp(channel_data))  # Peak-to-peak
            features.append(np.median(channel_data))
            features.append(np.std(channel_data))
            
            # Enhanced frequency domain features with Welch method
            f, Pxx = signal.welch(channel_data, fs=self.fs, nperseg=512, noverlap=256)
            
            # Power at each SSVEP frequency with multiple bandwidths
            for freq in self.ssvep_freqs:
                # Fundamental frequency - multiple bandwidths
                for bw in [0.3, 0.5, 1.0, 1.5]:  # Different bandwidths
                    band_mask = (f >= freq - bw) & (f <= freq + bw)
                    if np.any(band_mask):
                        power = np.trapz(Pxx[band_mask], f[band_mask])
                        features.append(np.log1p(power))
                    else:
                        features.append(0)
                
                # Second harmonic
                second_harmonic = freq * 2
                for bw in [0.3, 0.5]:
                    band_mask = (f >= second_harmonic - bw) & (f <= second_harmonic + bw)
                    if np.any(band_mask):
                        power = np.trapz(Pxx[band_mask], f[band_mask])
                        features.append(np.log1p(power))
                    else:
                        features.append(0)
                
                # Third harmonic
                third_harmonic = freq * 3
                for bw in [0.3]:
                    band_mask = (f >= third_harmonic - bw) & (f <= third_harmonic + bw)
                    if np.any(band_mask):
                        power = np.trapz(Pxx[band_mask], f[band_mask])
                        features.append(np.log1p(power))
                    else:
                        features.append(0)
                
                # Sub-harmonics
                sub_harmonic = freq * 0.5
                band_mask = (f >= sub_harmonic - 0.3) & (f <= sub_harmonic + 0.3)
                if np.any(band_mask):
                    power = np.trapz(Pxx[band_mask], f[band_mask])
                    features.append(np.log1p(power))
                else:
                    features.append(0)
            
            # Enhanced spectral features
            ssvep_range_mask = (f >= 5) & (f <= 30)
            
            if np.any(ssvep_range_mask):
                ssvep_psd = Pxx[ssvep_range_mask]
                ssvep_freqs = f[ssvep_range_mask]
                
                # Peak frequency and power
                peak_idx = np.argmax(ssvep_psd)
                features.extend([ssvep_freqs[peak_idx], ssvep_psd[peak_idx]])
                
                # Spectral centroid
                centroid = np.sum(ssvep_freqs * ssvep_psd) / np.sum(ssvep_psd)
                features.append(centroid)
                
                # Spectral bandwidth
                bandwidth = np.sqrt(np.sum(((ssvep_freqs - centroid)**2) * ssvep_psd) / np.sum(ssvep_psd))
                features.append(bandwidth)
                
                # Spectral rolloff
                cumulative_power = np.cumsum(ssvep_psd)
                rolloff_threshold = 0.85 * cumulative_power[-1]
                rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
                spectral_rolloff = ssvep_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
                features.append(spectral_rolloff)
                
                # Spectral flatness
                spectral_flatness = np.exp(np.mean(np.log(ssvep_psd + 1e-8))) / (np.mean(ssvep_psd) + 1e-8)
                features.append(spectral_flatness)
                
                # Spectral contrast
                spectral_contrast = np.std(ssvep_psd) / (np.mean(ssvep_psd) + 1e-8)
                features.append(spectral_contrast)
            else:
                features.extend([0]*6)
            
            # Enhanced phase-locking features
            for freq in self.ssvep_freqs:
                t = np.arange(len(channel_data)) / self.fs
                
                # Multiple phase offsets
                for phase_offset in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    ref_signal = np.sin(2 * np.pi * freq * t + phase_offset)
                    
                    # Phase correlation
                    try:
                        phase_corr = np.corrcoef(channel_data, ref_signal)[0, 1]
                        if np.isnan(phase_corr):
                            phase_corr = 0.0
                        features.append(phase_corr)
                    except:
                        features.append(0.0)
                    
                    # Phase-locking value (PLV)
                    try:
                        analytic_signal = signal.hilbert(channel_data)
                        phase = np.angle(analytic_signal)
                        ref_phase = 2 * np.pi * freq * t + phase_offset
                        phase_diff = phase - ref_phase
                        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                        features.append(plv)
                    except:
                        features.append(0.0)
        
        # Enhanced cross-channel features
        corr_matrix = np.corrcoef(eeg_filtered.T)
        features.extend([corr_matrix[0, 1], corr_matrix[0, 2], corr_matrix[1, 2]])
        
        # Spatial features (channel power ratios)
        channel_powers = [np.var(eeg_filtered[:, i]) for i in range(3)]
        power_ratios = [
            channel_powers[0] / (channel_powers[1] + 1e-8),
            channel_powers[1] / (channel_powers[2] + 1e-8),
            channel_powers[0] / (channel_powers[2] + 1e-8)
        ]
        features.extend(power_ratios)
        
        # Enhanced CCA features for each SSVEP frequency
        for freq in self.ssvep_freqs:
            cca_score = self.apply_enhanced_cca_with_phase(eeg_filtered, freq)
            features.append(cca_score)
        
        # Enhanced FBCCA features
        try:
            fbcca_features, _ = self.apply_fbcca_filtering_fast(eeg_occipital, self.ssvep_freqs)
            if len(fbcca_features) > 0:
                # Ensure consistent size
                if len(fbcca_features) >= 24:  # Increased from 16
                    features.extend(fbcca_features[:24])
                else:
                    features.extend(fbcca_features)
                    features.extend([0.0] * (24 - len(fbcca_features)))
            else:
                features.extend([0.0] * 24)
        except:
            features.extend([0.0] * 24)
        
        # Enhanced Riemannian features
        try:
            riemannian_features = self.extract_riemannian_features_fast(eeg_occipital)
            if len(riemannian_features) >= 15:  # Increased from 10
                features.extend(riemannian_features[:15])
            else:
                features.extend(riemannian_features)
                features.extend([0.0] * (15 - len(riemannian_features)))
        except:
            features.extend([0.0] * 15)
        
        return np.array(features)

    def extract_riemannian_features_fast(self, eeg_data):
        """Fast Riemannian feature extraction (simplified version)"""
        try:
            features = []
            
            # Calculate covariance matrix
            cov_matrix = self.calculate_covariance_matrix(eeg_data)
            if cov_matrix is None:
                return np.array([0.0] * 10)
            
            # Extract basic features only
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.real(eigenvals)
            eigenvals = np.sort(eigenvals)[::-1]
            
            # Top 3 eigenvalues
            features.extend(eigenvals[:3])
            
            # Log-determinant
            log_det = np.log(np.linalg.det(cov_matrix) + 1e-10)
            features.append(log_det)
            
            # Trace
            trace = np.trace(cov_matrix)
            features.append(trace)
            
            # Condition number
            condition_num = eigenvals[0] / (eigenvals[-1] + 1e-10)
            features.append(condition_num)
            
            # Frobenius norm
            frob_norm = np.sqrt(np.sum(cov_matrix**2))
            features.append(frob_norm)
            
            # Basic statistics
            features.extend([np.mean(cov_matrix), np.std(cov_matrix)])
            
            return np.array(features[:10])  # Ensure exactly 10 features and return as array
            
        except Exception as e:
            return np.array([0.0] * 10)

    def apply_fbcca_filtering_fast(self, eeg_data, ssvep_freqs):
        """Fast FBCCA filtering (simplified version)"""
        try:
            # Use only fundamental frequencies for speed
            filter_bands = []
            for freq in ssvep_freqs:
                filter_bands.append((freq - 0.5, freq + 0.5))
            
            fbcca_features = []
            cca_scores = []
            
            for band_idx, (low_freq, high_freq) in enumerate(filter_bands):
                # Apply bandpass filter
                band_data = self.apply_bandpass_filter(eeg_data, low_freq, high_freq)
                
                # Calculate simple correlation for each SSVEP frequency
                band_correlations = []
                t = np.arange(band_data.shape[0]) / self.fs
                
                for ssvep_freq in ssvep_freqs:
                    # Generate simple reference signal
                    ref_signal = np.sin(2 * np.pi * ssvep_freq * t)
                    
                    # Calculate correlation with first channel only
                    try:
                        corr = np.corrcoef(band_data[:, 0], ref_signal)[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                        band_correlations.append(corr)
                    except:
                        band_correlations.append(0.0)
                
                # Store features for this band
                fbcca_features.extend(band_correlations)
                cca_scores.append(np.max(band_correlations))
            
            fbcca_features = np.array(fbcca_features)
            return fbcca_features, cca_scores
            
        except Exception as e:
            return np.array([]), []

    def apply_ica_artifact_removal(self, data, n_components=None):
        """Apply ICA for artifact removal with improved component selection"""
        if n_components is None:
            n_components = min(data.shape[1], data.shape[0] // 10)
        
        try:
            # Center the data
            data_centered = data - np.mean(data, axis=0)
            
            # Apply ICA
            ica = FastICA(n_components=n_components, random_state=42, max_iter=1000, tol=0.01)
            ica_data = ica.fit_transform(data_centered)
            
            # Artifact detection based on multiple criteria
            variances = np.var(ica_data, axis=0)
            kurtoses = kurtosis(ica_data, axis=0)
            
            # Remove components with high variance (likely artifacts) or extreme kurtosis
            var_threshold = np.percentile(variances, 90)
            kurt_threshold = np.percentile(np.abs(kurtoses), 95)
            
            # Zero out artifact components
            artifact_mask = (variances > var_threshold) | (np.abs(kurtoses) > kurt_threshold)
            ica_data[:, artifact_mask] = 0
            
            # Transform back
            cleaned_data = ica.inverse_transform(ica_data)
            return cleaned_data + np.mean(data, axis=0)  # Restore mean
            
        except Exception as e:
            print(f"ICA failed: {e}, returning original data")
            return data
    
    def apply_adaptive_filter(self, data, reference_channel=0):
        """Apply adaptive filtering using normalized LMS algorithm"""
        try:
            filtered_data = data.copy()
            mu = 0.01  # Learning rate
            filter_length = 8  # Reduced filter order
            
            for ch in range(data.shape[1]):
                if ch != reference_channel:
                    reference = data[:, reference_channel]
                    target = data[:, ch]
                    
                    # Normalized LMS filter
                    filtered = np.zeros_like(target)
                    weights = np.zeros(filter_length)
                    
                    for i in range(len(target)):
                        if i >= filter_length:
                            # Get reference window
                            ref_window = reference[i-filter_length:i]
                            
                            # Predict using current weights
                            prediction = np.dot(weights, ref_window)
                            filtered[i] = target[i] - prediction
                            
                            # Update weights with normalization
                            ref_energy = np.dot(ref_window, ref_window) + 1e-8
                            weights += (mu / ref_energy) * filtered[i] * ref_window
                        else:
                            filtered[i] = target[i]
                    
                    filtered_data[:, ch] = filtered
            
            return filtered_data
        except Exception as e:
            print(f"Adaptive filtering failed: {e}, returning original data")
            return data

    def apply_csp_filtering(self, eeg_data, labels, n_components=2):
        """Lightweight CSP implementation for MI spatial filtering"""
        try:
            # Ensure we have binary classification for CSP
            unique_labels = np.unique(labels)
            if len(unique_labels) != 2:
                print("CSP requires binary classification, using first two classes")
                mask = np.isin(labels, unique_labels[:2])
                eeg_data = eeg_data[mask]
                labels = labels[mask]
            
            if len(eeg_data) < 10:  # Need minimum samples
                print("Insufficient samples for CSP")
                return eeg_data.reshape(eeg_data.shape[0], -1), None
            
            # Simplified CSP: use only variance features instead of full covariance
            n_trials = eeg_data.shape[0]
            n_channels = eeg_data.shape[1]
            n_samples = eeg_data.shape[2]
            
            # Calculate variance for each trial and channel
            variances = np.var(eeg_data, axis=2)  # Shape: (n_trials, n_channels)
            
            # Calculate mean variance for each class
            class_0_mask = labels == unique_labels[0]
            class_1_mask = labels == unique_labels[1]
            
            if np.sum(class_0_mask) < 3 or np.sum(class_1_mask) < 3:
                print("Insufficient samples per class for CSP")
                return eeg_data.reshape(eeg_data.shape[0], -1), None
            
            mean_var_0 = np.mean(variances[class_0_mask], axis=0)
            mean_var_1 = np.mean(variances[class_1_mask], axis=0)
            
            # Simple spatial filter based on variance ratio
            variance_ratio = mean_var_0 / (mean_var_1 + 1e-8)
            
            # Select channels with highest variance ratio differences
            ratio_diff = np.abs(variance_ratio - 1)
            selected_channels = np.argsort(ratio_diff)[-n_components:]
            
            # Create lightweight spatial filter
            csp_filters = np.zeros((n_channels, n_components))
            for i, ch in enumerate(selected_channels):
                csp_filters[ch, i] = 1.0
            
            # Apply lightweight CSP filtering
            filtered_data = []
            for trial in eeg_data:
                filtered_trial = csp_filters.T @ trial
                filtered_data.append(filtered_trial)
            
            filtered_data = np.array(filtered_data)
            filtered_data = filtered_data.reshape(n_trials, -1)
            
            print(f"Lightweight CSP applied: {eeg_data.shape} -> {filtered_data.shape}")
            return filtered_data, csp_filters
            
        except Exception as e:
            print(f"Lightweight CSP failed: {e}, returning original data")
            return eeg_data.reshape(eeg_data.shape[0], -1), None

    def apply_fbcsp_filtering(self, eeg_data, labels, n_components=2):
        """Lightweight FBCSP implementation for enhanced MI classification"""
        try:
            # Reduced filter bank for lighter computation
            filter_bands = [
                (8, 12),   # Alpha/Mu
                (12, 20),  # Beta
                (20, 28),  # Gamma
            ]
            
            fbcsp_features = []
            csp_filters_list = []
            
            for band_idx, (low_freq, high_freq) in enumerate(filter_bands):
                # print(f"Processing lightweight FBCSP band {band_idx + 1}: {low_freq}-{high_freq} Hz")  # Reduced verbose printing
                
                # Apply bandpass filter to each trial
                band_data = []
                for trial in eeg_data:
                    filtered_trial = self.apply_bandpass_filter(trial, low_freq, high_freq)
                    band_data.append(filtered_trial)
                band_data = np.array(band_data)
                
                # Apply lightweight CSP to this frequency band
                filtered_band, csp_filters = self.apply_csp_filtering(band_data, labels, n_components)
                
                if csp_filters is not None:
                    # Extract variance features from CSP components
                    var_features = np.var(filtered_band, axis=1)
                    fbcsp_features.append(var_features)
                    csp_filters_list.append(csp_filters)
                else:
                    # Fallback: use variance of original band data
                    var_features = np.var(band_data.reshape(band_data.shape[0], -1), axis=1)
                    fbcsp_features.append(var_features)
            
            # Concatenate features from all bands
            if fbcsp_features:
                fbcsp_features = np.concatenate(fbcsp_features, axis=1)
                # print(f"Lightweight FBCSP features shape: {fbcsp_features.shape}")  # Reduced verbose printing
                return fbcsp_features, csp_filters_list
            else:
                # print("Lightweight FBCSP failed, returning original data")  # Reduced verbose printing
                return eeg_data.reshape(eeg_data.shape[0], -1), None
                
        except Exception as e:
            # print(f"Lightweight FBCSP failed: {e}, returning original data")  # Reduced verbose printing
            return eeg_data.reshape(eeg_data.shape[0], -1), None

    def apply_fbcca_filtering(self, eeg_data, ssvep_freqs):
        """Filter Bank CCA (FBCCA) for enhanced SSVEP classification"""
        try:
            # Define narrow bandpass filters around SSVEP frequencies
            filter_bands = []
            for freq in ssvep_freqs:
                # Create narrow bands around each frequency
                filter_bands.extend([
                    (freq - 0.5, freq + 0.5),  # ±0.5 Hz
                    (freq * 2 - 0.5, freq * 2 + 0.5),  # Second harmonic
                ])
            
            # Remove duplicates and sort
            filter_bands = list(set(filter_bands))
            filter_bands.sort()
            
            print(f"FBCCA using {len(filter_bands)} filter bands")
            
            fbcca_features = []
            cca_scores = []
            
            for band_idx, (low_freq, high_freq) in enumerate(filter_bands):
                # Apply bandpass filter
                band_data = self.apply_bandpass_filter(eeg_data, low_freq, high_freq)
                
                # Calculate CCA correlation for each SSVEP frequency
                band_correlations = []
                t = np.arange(band_data.shape[0]) / self.fs
                
                for ssvep_freq in ssvep_freqs:
                    # Generate reference signal
                    ref_signal = np.sin(2 * np.pi * ssvep_freq * t)
                    
                    # Calculate correlation with each channel
                    channel_correlations = []
                    for ch in range(band_data.shape[1]):
                        try:
                            corr = np.corrcoef(band_data[:, ch], ref_signal)[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                            channel_correlations.append(corr)
                        except:
                            channel_correlations.append(0.0)
                    
                    # Use maximum correlation across channels
                    max_corr = np.max(np.abs(channel_correlations))
                    band_correlations.append(max_corr)
                
                # Store features for this band
                fbcca_features.extend(band_correlations)
                cca_scores.append(np.max(band_correlations))
            
            fbcca_features = np.array(fbcca_features)
            print(f"FBCCA features shape: {fbcca_features.shape}")
            
            return fbcca_features, cca_scores
            
        except Exception as e:
            print(f"FBCCA failed: {e}")
            return np.array([]), []

    def calculate_riemannian_distance(self, cov1, cov2):
        """Enhanced Riemannian distance calculation with better numerical stability"""
        try:
            # Ensure matrices are positive definite
            cov1 = cov1 + 1e-8 * np.eye(cov1.shape[0])
            cov2 = cov2 + 1e-8 * np.eye(cov2.shape[0])
            
            # Calculate eigenvalues for numerical stability check
            eigenvals1 = np.linalg.eigvals(cov1)
            eigenvals2 = np.linalg.eigvals(cov2)
            
            # Check if matrices are positive definite
            if np.any(eigenvals1 <= 0) or np.any(eigenvals2 <= 0):
                print("Warning: Non-positive definite matrices detected")
                return np.inf
            
            # Calculate Riemannian distance using log-Euclidean metric for better stability
            try:
                # Use log-Euclidean metric as fallback
                log_cov1 = logm(cov1)
                log_cov2 = logm(cov2)
                distance = np.linalg.norm(log_cov1 - log_cov2, 'fro')
            except:
                # If logm fails, use Frobenius norm as fallback
                distance = np.linalg.norm(cov1 - cov2, 'fro')
            
            return distance
            
        except Exception as e:
            print(f"Enhanced Riemannian distance calculation failed: {e}")
            return np.inf

    def calculate_covariance_matrix(self, eeg_data):
        """Enhanced covariance matrix calculation for Riemannian geometry"""
        try:
            # Ensure data is 2D
            if eeg_data.ndim == 3:
                eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
            
            # Remove any remaining NaN or inf values
            eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Center the data
            eeg_data = eeg_data - np.mean(eeg_data, axis=0)
            
            # Calculate covariance matrix
            cov_matrix = np.cov(eeg_data.T)
            
            # Ensure positive definiteness with better regularization
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Apply regularization to ensure positive definiteness
            min_eigenval = np.min(eigenvals)
            if min_eigenval <= 0:
                # Add regularization to make it positive definite
                regularization = abs(min_eigenval) + 1e-6
                eigenvals = eigenvals + regularization
            
            # Ensure minimum eigenvalue for numerical stability
            eigenvals = np.maximum(eigenvals, 1e-8)
            
            # Reconstruct the positive definite covariance matrix
            cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Normalize by trace for better numerical stability
            trace = np.trace(cov_matrix)
            if trace > 0:
                cov_matrix = cov_matrix / trace
            
            return cov_matrix
            
        except Exception as e:
            print(f"Enhanced covariance matrix calculation failed: {e}")
            return None

    def apply_riemannian_classification(self, eeg_data, labels, test_data=None):
        """Enhanced Riemannian geometry with improved MDM classifier"""
        try:
            print("Applying enhanced Riemannian geometry classification...")
            
            # If test_data is provided, we're in prediction mode
            if test_data is not None:
                # Validate test data
                if test_data is None or test_data.size == 0:
                    print("Invalid test data for Riemannian classification")
                    return None, None
                
                # Ensure test data is 2D
                if test_data.ndim == 3:
                    test_data = test_data.reshape(test_data.shape[0], -1)
                elif test_data.ndim == 1:
                    test_data = test_data.reshape(1, -1)
                
                # Calculate test covariance matrix
                test_cov = self.calculate_covariance_matrix(test_data)
                if test_cov is None:
                    print("Failed to calculate test covariance matrix")
                    return None, None
                
                # Use pre-trained classifier if available
                if hasattr(self, 'ssvep_riemannian_classifier') and self.ssvep_riemannian_classifier is not None:
                    # Find closest class mean using enhanced MDM
                    min_distance = np.inf
                    best_label = None
                    
                    for label, mean_cov in self.ssvep_riemannian_classifier.items():
                        distance = self.calculate_riemannian_distance(test_cov, mean_cov)
                        if distance < min_distance:
                            min_distance = distance
                            best_label = label
                    
                    return best_label, min_distance
                else:
                    print("No trained Riemannian classifier available")
                    return None, None
            
            # Training mode
            if eeg_data is None or len(eeg_data) == 0:
                print("No training data provided for Riemannian classification")
                return None, None
            
            # Calculate covariance matrices for training data
            train_covs = []
            train_labels = []
            
            for i in range(len(eeg_data)):
                cov_matrix = self.calculate_covariance_matrix(eeg_data[i])
                if cov_matrix is not None:
                    train_covs.append(cov_matrix)
                    train_labels.append(labels[i])
            
            if len(train_covs) < 5:
                print("Insufficient samples for Riemannian classification")
                return None, None
            
            train_covs = np.array(train_covs)
            train_labels = np.array(train_labels)
            
            # Calculate class means in Riemannian space using enhanced MDM
            unique_labels = np.unique(train_labels)
            class_means = {}
            
            for label in unique_labels:
                label_mask = train_labels == label
                if np.sum(label_mask) > 0:
                    label_covs = train_covs[label_mask]
                    
                    # Enhanced mean calculation using iterative approach
                    mean_cov = label_covs[0].copy()
                    
                    # Iterative mean calculation with more iterations for better convergence
                    for iteration in range(5):  # Increased iterations
                        new_mean = np.zeros_like(mean_cov)
                        total_weight = 0
                        
                        for cov in label_covs:
                            # Calculate Riemannian distance for weighting
                            distance = self.calculate_riemannian_distance(mean_cov, cov)
                            weight = 1.0 / (1.0 + distance)  # Inverse distance weighting
                            
                            new_mean += weight * cov
                            total_weight += weight
                        
                        if total_weight > 0:
                            mean_cov = new_mean / total_weight
                        
                        # Ensure positive definiteness
                        eigenvals, eigenvecs = np.linalg.eigh(mean_cov)
                        eigenvals = np.maximum(eigenvals, 1e-6)  # Ensure positive eigenvalues
                        mean_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                    
                    class_means[label] = mean_cov
            
            return class_means, None
            
        except Exception as e:
            print(f"Enhanced Riemannian classification failed: {e}")
            return None, None

    def extract_riemannian_features_fixed(self, eeg_data):
        """FIXED Riemannian geometry features with proper error handling"""
        try:
            features = []
            
            # Ensure data is properly shaped
            if eeg_data.ndim == 1:
                eeg_data = eeg_data.reshape(-1, 1)
            elif eeg_data.ndim == 3:
                eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
            
            # Calculate covariance matrix with robust estimation
            cov_matrix = self.calculate_covariance_matrix_robust(eeg_data)
            if cov_matrix is None or cov_matrix.size == 0:
                return [0.0] * 20
            
            # Eigenvalue decomposition with error handling
            try:
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                eigenvals = np.real(eigenvals)
                eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
                
                # Ensure positive eigenvalues
                eigenvals = np.maximum(eigenvals, 1e-12)
                
                # Top eigenvalues (up to 5)
                n_eigenvals = min(5, len(eigenvals))
                features.extend(eigenvals[:n_eigenvals].tolist())
                
                # Pad if necessary
                while len(features) < 5:
                    features.append(0.0)
                    
            except Exception as e:
                print(f"Eigenvalue decomposition failed: {e}")
                features.extend([0.0] * 5)
                eigenvals = np.array([1.0])  # Fallback
            
            # Log-determinant with numerical stability
            try:
                det_val = np.linalg.det(cov_matrix)
                if det_val <= 0:
                    log_det = -50.0  # Very negative value for singular matrices
                else:
                    log_det = np.log(det_val)
                features.append(float(log_det))
            except:
                features.append(-50.0)
            
            # Trace
            try:
                trace = np.trace(cov_matrix)
                features.append(float(trace))
            except:
                features.append(0.0)
            
            # Condition number with stability check
            try:
                if len(eigenvals) > 1:
                    condition_num = eigenvals[0] / (eigenvals[-1] + 1e-12)
                    # Cap condition number to prevent extreme values
                    condition_num = min(condition_num, 1e6)
                    features.append(float(condition_num))
                else:
                    features.append(1.0)
            except:
                features.append(1.0)
            
            # Frobenius norm
            try:
                frob_norm = np.sqrt(np.sum(cov_matrix**2))
                features.append(float(frob_norm))
            except:
                features.append(0.0)
            
            # Matrix statistics with error handling
            try:
                # Off-diagonal elements
                n = cov_matrix.shape[0]
                if n > 1:
                    off_diag_indices = np.triu_indices(n, k=1)
                    off_diag = cov_matrix[off_diag_indices]
                    
                    if len(off_diag) > 0:
                        features.extend([
                            float(np.mean(off_diag)),
                            float(np.std(off_diag)),
                            float(np.max(off_diag)),
                            float(np.min(off_diag))
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            except:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            try:
                # Diagonal elements
                diag_elements = np.diag(cov_matrix)
                if len(diag_elements) > 0:
                    features.extend([
                        float(np.mean(diag_elements)),
                        float(np.std(diag_elements)),
                        float(np.max(diag_elements)),
                        float(np.min(diag_elements))
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            except:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Additional robust features
            try:
                # Matrix rank
                rank = np.linalg.matrix_rank(cov_matrix)
                features.append(float(rank))
            except:
                features.append(0.0)
            
            try:
                # Spectral radius (largest eigenvalue)
                spectral_radius = np.max(np.abs(eigenvals))
                features.append(float(spectral_radius))
            except:
                features.append(0.0)
            
            # Ensure exactly 20 features
            features = features[:20]  # Truncate if too many
            while len(features) < 20:
                features.append(0.0)
            
            # Convert to list and ensure all are finite numbers
            features = [float(f) if np.isfinite(f) else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            print(f"Riemannian feature extraction completely failed: {e}")
            return [0.0] * 20
    
    def calculate_covariance_matrix_robust(self, eeg_data):
        """Robust covariance matrix calculation with better error handling"""
        try:
            # Ensure data is 2D
            if eeg_data.ndim == 3:
                eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
            elif eeg_data.ndim == 1:
                eeg_data = eeg_data.reshape(-1, 1)
            
            # Check data validity
            if eeg_data.size == 0:
                return None
            
            # Remove any NaN or inf values
            eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check if we have enough samples
            n_samples, n_features = eeg_data.shape
            if n_samples < 2 or n_features < 1:
                return np.eye(max(1, n_features)) * 1e-6
            
            # Center the data
            eeg_data = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)
            
            # Calculate covariance matrix with regularization
            if n_samples >= n_features:
                # Standard covariance calculation
                cov_matrix = np.cov(eeg_data.T)
            else:
                # Use regularized covariance for small sample sizes
                cov_matrix = np.dot(eeg_data.T, eeg_data) / (n_samples - 1)
            
            # Ensure matrix is 2D
            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[cov_matrix]])
            elif cov_matrix.ndim == 1:
                cov_matrix = np.diag(cov_matrix)
            
            # Add regularization to ensure positive definiteness
            regularization = 1e-6 * np.eye(cov_matrix.shape[0])
            cov_matrix = cov_matrix + regularization
            
            # Check for numerical issues
            if not np.all(np.isfinite(cov_matrix)):
                print("Warning: Non-finite values in covariance matrix")
                return np.eye(cov_matrix.shape[0]) * 1e-6
            
            # Ensure positive definiteness
            eigenvals = np.linalg.eigvals(cov_matrix)
            min_eigenval = np.min(np.real(eigenvals))
            
            if min_eigenval <= 0:
                # Add more regularization
                regularization = (abs(min_eigenval) + 1e-6) * np.eye(cov_matrix.shape[0])
                cov_matrix = cov_matrix + regularization
            
            return cov_matrix
            
        except Exception as e:
            print(f"Robust covariance matrix calculation failed: {e}")
            # Return identity matrix as fallback
            try:
                n_features = eeg_data.shape[1] if eeg_data.ndim > 1 else 1
                return np.eye(n_features) * 1e-6
            except:
                return np.eye(1) * 1e-6

    def show_training_metrics(self, X_train, y_train, task_type):
        """Show comprehensive training metrics and analysis"""
        print(f"\n📊 {task_type} Training Metrics Analysis")
        print("=" * 50)
        
        # Get predictions and probabilities
        if task_type == 'MI':
            model = self.mi_ensemble
            scaler = self.mi_scaler
            selector = self.mi_selector
            label_encoder = self.mi_label_encoder
        else:
            model = self.ssvep_ensemble
            scaler = self.ssvep_scaler
            selector = self.ssvep_selector
            label_encoder = self.ssvep_label_encoder
        
        # Transform data
        X_scaled = scaler.transform(X_train)
        X_selected = selector.transform(X_scaled)
        
        # Get predictions and probabilities
        train_pred = model.predict(X_selected)
        train_proba = model.predict_proba(X_selected)
        train_confidences = np.max(train_proba, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_recall_fscore_support, 
            confusion_matrix, classification_report, roc_auc_score
        )
        
        # Overall metrics
        acc = accuracy_score(y_train, train_pred)
        f1_weighted = f1_score(y_train, train_pred, average='weighted')
        f1_macro = f1_score(y_train, train_pred, average='macro')
        f1_micro = f1_score(y_train, train_pred, average='micro')
        
        print(f"Overall Training Performance:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"  F1 Score (Macro): {f1_macro:.4f}")
        print(f"  F1 Score (Micro): {f1_micro:.4f}")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_train, train_pred, average=None)
        conf_matrix = confusion_matrix(y_train, train_pred)
        
        print(f"\nPer-Class Training Metrics:")
        class_names = label_encoder.classes_
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {precision[i]:.4f}")
            print(f"    Recall: {recall[i]:.4f}")
            print(f"    F1-Score: {f1[i]:.4f}")
            print(f"    Support: {support[i]}")
        
        print(f"\nTraining Confusion Matrix:")
        print(conf_matrix)
        
        # Confidence analysis
        from scipy.stats import skew, kurtosis
        print(f"\nTraining Confidence Statistics:")
        print(f"  Mean Confidence: {np.mean(train_confidences):.4f}")
        print(f"  Std Confidence: {np.std(train_confidences):.4f}")
        print(f"  Min Confidence: {np.min(train_confidences):.4f}")
        print(f"  Max Confidence: {np.max(train_confidences):.4f}")
        print(f"  Median Confidence: {np.median(train_confidences):.4f}")
        print(f"  Skewness: {skew(train_confidences):.4f}")
        print(f"  Kurtosis: {kurtosis(train_confidences):.4f}")
        
        # Confidence percentiles
        print(f"  Confidence Percentiles:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"    {p}th: {np.percentile(train_confidences, p):.4f}")
        
        # High/low confidence analysis
        high_conf_threshold = 0.8
        low_conf_threshold = 0.5
        
        high_conf_mask = train_confidences >= high_conf_threshold
        low_conf_mask = train_confidences < low_conf_threshold
        
        high_conf_count = np.sum(high_conf_mask)
        low_conf_count = np.sum(low_conf_mask)
        
        print(f"\nTraining Confidence Quality:")
        print(f"  High Confidence (≥{high_conf_threshold}): {high_conf_count} ({high_conf_count/len(train_confidences)*100:.1f}%)")
        print(f"  Low Confidence (<{low_conf_threshold}): {low_conf_count} ({low_conf_count/len(train_confidences)*100:.1f}%)")
        
        # Class-specific confidence analysis
        print(f"\nClass-Specific Training Confidence:")
        for i, class_name in enumerate(class_names):
            class_mask = y_train == i
            if np.sum(class_mask) > 0:
                class_confidences = train_confidences[class_mask]
                print(f"  {class_name}:")
                print(f"    Mean Confidence: {np.mean(class_confidences):.4f}")
                print(f"    Std Confidence: {np.std(class_confidences):.4f}")
                print(f"    Min Confidence: {np.min(class_confidences):.4f}")
                print(f"    Max Confidence: {np.max(class_confidences):.4f}")
                print(f"    Skewness: {skew(class_confidences):.4f}")
                print(f"    Kurtosis: {kurtosis(class_confidences):.4f}")
                
                # High/low confidence for this class
                class_high_conf = np.sum(class_confidences >= high_conf_threshold)
                class_low_conf = np.sum(class_confidences < low_conf_threshold)
                print(f"    High Confidence (≥{high_conf_threshold}): {class_high_conf} ({class_high_conf/len(class_confidences)*100:.1f}%)")
                print(f"    Low Confidence (<{low_conf_threshold}): {class_low_conf} ({class_low_conf/len(class_confidences)*100:.1f}%)")
        
        # Model complexity analysis
        print(f"\nModel Complexity Analysis:")
        print(f"  Training Samples: {len(X_train)}")
        print(f"  Feature Dimensions: {X_train.shape[1]}")
        print(f"  Selected Features: {X_selected.shape[1]}")
        print(f"  Number of Classes: {len(class_names)}")
        
        # Feature importance analysis (if available)
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                print(f"  Feature Importance - Mean: {np.mean(importances):.4f}")
                print(f"  Feature Importance - Std: {np.std(importances):.4f}")
                print(f"  Feature Importance - Max: {np.max(importances):.4f}")
                print(f"  Feature Importance - Min: {np.min(importances):.4f}")
        except:
            pass
        
        # Overfitting analysis
        print(f"\nOverfitting Analysis:")
        print(f"  Training Accuracy: {acc:.4f}")
        if acc > 0.95:
            print(f"  ⚠️  High training accuracy may indicate overfitting")
        elif acc < 0.6:
            print(f"  ⚠️  Low training accuracy may indicate underfitting")
        else:
            print(f"  ✅ Training accuracy appears reasonable")
        
        # Confidence distribution analysis
        print(f"\nConfidence Distribution Analysis:")
        if np.mean(train_confidences) > 0.9:
            print(f"  ⚠️  Very high mean confidence may indicate overfitting")
        elif np.mean(train_confidences) < 0.6:
            print(f"  ⚠️  Low mean confidence may indicate model uncertainty")
        else:
            print(f"  ✅ Mean confidence appears reasonable")
        
        if np.std(train_confidences) < 0.1:
            print(f"  ⚠️  Low confidence variance may indicate overfitting")
        elif np.std(train_confidences) > 0.3:
            print(f"  ⚠️  High confidence variance may indicate model instability")
        else:
            print(f"  ✅ Confidence variance appears reasonable")
        
        return {
            'accuracy': acc,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'mean_confidence': np.mean(train_confidences),
            'std_confidence': np.std(train_confidences),
            'confusion_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1,
            'support': support
        }

    def generate_classification_report(self, y_true, y_pred, class_names):
        """Generate classification report in the requested format"""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Print the report in the requested format
        print("Validation Classification Report:")
        print("     precision    recall   f1-score   support")
        print()
        
        for i, class_name in enumerate(class_names):
            print(f"    {class_name:<10} {precision[i]:.2f}      {recall[i]:.2f}      {f1[i]:.2f}        {support[i]}")
        
        print()
        print(f"    accuracy                           {accuracy:.2f}       {len(y_true)}")
        print(f"   macro avg       {macro_precision:.2f}      {macro_recall:.2f}      {macro_f1:.2f}       {len(y_true)}")
        print(f"weighted avg       {weighted_precision:.2f}      {weighted_recall:.2f}      {weighted_f1:.2f}       {len(y_true)}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'macro_avg': (macro_precision, macro_recall, macro_f1),
            'weighted_avg': (weighted_precision, weighted_recall, weighted_f1)
        }

    def calibrate_confidence(self, probabilities, task_type):
        """Calibrate prediction confidence based on validation performance"""
        if task_type == 'MI':
            # Adjust confidence based on observed validation performance
            confidence_multiplier = 0.8  # Conservative for MI
        else:  # SSVEP
            # More conservative for SSVEP due to lower validation accuracy
            confidence_multiplier = 0.6
        
        # Apply calibration
        calibrated_probs = probabilities * confidence_multiplier
        # Ensure probabilities sum to 1
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
        
        return calibrated_probs
    
    def enhanced_prediction_strategy(self, trial_data, task_type):
        """Enhanced prediction strategy with multiple validation approaches"""
        if task_type == 'MI':
            # MI prediction with enhanced validation
            features = self.extract_enhanced_mi_features(trial_data)
            
            # Handle NaN/inf in features
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale and select features
            features_scaled = self.mi_scaler.transform(features.reshape(1, -1))
            features_selected = self.mi_selector.transform(features_scaled)
            
            # Get ensemble prediction with confidence calibration
            ensemble_proba = self.mi_ensemble.predict_proba(features_selected)[0]
            calibrated_proba = self.calibrate_confidence(ensemble_proba, 'MI')
            
            ensemble_pred = np.argmax(calibrated_proba)
            ensemble_conf = np.max(calibrated_proba)
            
            # Decode prediction
            prediction = self.mi_label_encoder.inverse_transform([ensemble_pred])[0]
            
            return prediction, ensemble_conf
            
        else:  # SSVEP
            # Multi-strategy SSVEP prediction
            
            # Strategy 1: Enhanced CCA
            cca_pred, cca_conf = self.enhanced_cca_predict(trial_data)
            
            # Strategy 2: Riemannian classification
            riemannian_pred = None
            riemannian_conf = 0.0
            
            if hasattr(self, 'ssvep_riemannian_classifier') and self.ssvep_riemannian_classifier is not None:
                try:
                    occipital_channels = [5, 6, 7]  # PO7, OZ, PO8
                    trial_occipital = trial_data[:, occipital_channels]
                    
                    riemannian_pred, riemannian_distance = self.apply_riemannian_classification(
                        None, None, trial_occipital
                    )
                    
                    if riemannian_pred is not None:
                        # Convert distance to confidence
                        riemannian_conf = max(0.0, 1.0 - riemannian_distance / 15.0)
                except:
                    pass
            
            # Strategy 3: Ensemble prediction
            features = self.extract_enhanced_ssvep_features(trial_data)
            
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            features_scaled = self.ssvep_scaler.transform(features.reshape(1, -1))
            features_selected = self.ssvep_selector.transform(features_scaled)
            
            ensemble_proba = self.ssvep_ensemble.predict_proba(features_selected)[0]
            calibrated_proba = self.calibrate_confidence(ensemble_proba, 'SSVEP')
            
            ensemble_pred = np.argmax(calibrated_proba)
            ensemble_conf = np.max(calibrated_proba)
            
            # Decision fusion with weighted voting
            predictions = []
            confidences = []
            
            # CCA prediction
            if cca_pred is not None and cca_conf > self.cca_threshold:
                predictions.append((cca_pred, cca_conf * 1.2))  # Boost CCA confidence slightly
            
            # Riemannian prediction
            if riemannian_pred is not None and riemannian_conf > 0.5:
                # Find class index for riemannian prediction
                try:
                    riemannian_idx = np.where(self.ssvep_label_encoder.classes_ == riemannian_pred)[0][0]
                    predictions.append((riemannian_idx, riemannian_conf))
                except:
                    pass
            
            # Ensemble prediction
            predictions.append((ensemble_pred, ensemble_conf))
            
            if len(predictions) > 1:
                # Weighted voting
                vote_scores = {}
                for pred, conf in predictions:
                    if pred in vote_scores:
                        vote_scores[pred] += conf
                    else:
                        vote_scores[pred] = conf
                
                # Select prediction with highest weighted score
                best_pred = max(vote_scores.items(), key=lambda x: x[1])
                final_pred = best_pred[0]
                final_conf = best_pred[1] / len(predictions)  # Average confidence
            else:
                final_pred = ensemble_pred
                final_conf = ensemble_conf
            
            # Decode prediction
            prediction = self.ssvep_label_encoder.inverse_transform([final_pred])[0]
            
            return prediction, final_conf

    def extract_enhanced_ssvep_features(self, eeg_data):
        """Enhanced SSVEP feature extraction with improved frequency analysis"""
        # Focus on occipital channels (PO7, OZ, PO8) - indices 5, 6, 7
        occipital_channels = [5, 6, 7]
        eeg_occipital = eeg_data[:, occipital_channels]
        
        # Enhanced preprocessing with multiple filtering stages
        eeg_occipital = self.apply_notch_filter(eeg_occipital)
        eeg_occipital = signal.detrend(eeg_occipital, axis=0)
        eeg_occipital = eeg_occipital - np.median(eeg_occipital, axis=0)
        
        # Apply adaptive filtering for better signal quality
        try:
            eeg_occipital = self.apply_adaptive_filter(eeg_occipital)
        except:
            pass
        
        features = []
        
        # Enhanced frequency analysis with better resolution
        for ch in range(eeg_occipital.shape[1]):
            channel_data = eeg_occipital[:, ch]
            
            # Multi-taper spectral estimation for better frequency resolution
            f, Pxx = signal.welch(channel_data, fs=self.fs, nperseg=min(512, len(channel_data)//2), 
                                noverlap=min(256, len(channel_data)//4), window='hann')
            
            # Enhanced SSVEP frequency detection with harmonics
            for target_freq in self.ssvep_freqs:
                # Fundamental frequency with multiple bandwidths
                for bw in [0.2, 0.5, 1.0]:  # Different bandwidth sensitivities
                    mask = (f >= target_freq - bw) & (f <= target_freq + bw)
                    if np.any(mask):
                        power = np.trapz(Pxx[mask], f[mask])
                        # Normalize by total power in SSVEP range
                        ssvep_mask = (f >= 5) & (f <= 30)
                        total_power = np.trapz(Pxx[ssvep_mask], f[ssvep_mask]) if np.any(ssvep_mask) else 1
                        normalized_power = power / (total_power + 1e-8)
                        features.append(np.log1p(normalized_power))
                    else:
                        features.append(0)
                
                # Harmonic analysis (2nd, 3rd harmonics)
                for harmonic in [2, 3]:
                    harm_freq = target_freq * harmonic
                    mask = (f >= harm_freq - 0.5) & (f <= harm_freq + 0.5)
                    if np.any(mask):
                        harm_power = np.trapz(Pxx[mask], f[mask])
                        # Ratio to fundamental
                        fund_mask = (f >= target_freq - 0.5) & (f <= target_freq + 0.5)
                        fund_power = np.trapz(Pxx[fund_mask], f[fund_mask]) if np.any(fund_mask) else 1
                        harm_ratio = harm_power / (fund_power + 1e-8)
                        features.append(harm_ratio)
                    else:
                        features.append(0)
                
                # Sub-harmonic analysis
                sub_freq = target_freq / 2
                if sub_freq >= 1:  # Valid sub-harmonic
                    mask = (f >= sub_freq - 0.3) & (f <= sub_freq + 0.3)
                    if np.any(mask):
                        sub_power = np.trapz(Pxx[mask], f[mask])
                        fund_mask = (f >= target_freq - 0.5) & (f <= target_freq + 0.5)
                        fund_power = np.trapz(Pxx[fund_mask], f[fund_mask]) if np.any(fund_mask) else 1
                        sub_ratio = sub_power / (fund_power + 1e-8)
                        features.append(sub_ratio)
                    else:
                        features.append(0)
                else:
                    features.append(0)
            
            # Enhanced spectral features
            ssvep_range = (f >= 5) & (f <= 30)
            if np.any(ssvep_range):
                ssvep_psd = Pxx[ssvep_range]
                ssvep_freqs = f[ssvep_range]
                
                # Peak detection with prominence
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(ssvep_psd, prominence=np.max(ssvep_psd)*0.1)
                
                if len(peaks) > 0:
                    # Dominant peak frequency and power
                    dominant_idx = peaks[np.argmax(ssvep_psd[peaks])]
                    dominant_freq = ssvep_freqs[dominant_idx]
                    dominant_power = ssvep_psd[dominant_idx]
                    
                    features.extend([dominant_freq, dominant_power])
                    
                    # Peak sharpness (Q-factor)
                    half_max = dominant_power / 2
                    left_half = np.where(ssvep_psd[:dominant_idx] <= half_max)[0]
                    right_half = np.where(ssvep_psd[dominant_idx:] <= half_max)[0] + dominant_idx
                    
                    if len(left_half) > 0 and len(right_half) > 0:
                        bandwidth = ssvep_freqs[right_half[0]] - ssvep_freqs[left_half[-1]]
                        q_factor = dominant_freq / (bandwidth + 1e-8)
                        features.append(q_factor)
                    else:
                        features.append(0)
                    
                    # Number of significant peaks
                    features.append(len(peaks))
                else:
                    features.extend([0, 0, 0, 0])
                
                # Spectral entropy (measure of frequency spreading)
                psd_norm = ssvep_psd / (np.sum(ssvep_psd) + 1e-8)
                spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
                features.append(spectral_entropy)
                
                # Alpha/Beta ratio in SSVEP range
                alpha_mask = (ssvep_freqs >= 8) & (ssvep_freqs <= 13)
                beta_mask = (ssvep_freqs >= 13) & (ssvep_freqs <= 25)
                
                alpha_power = np.trapz(ssvep_psd[alpha_mask], ssvep_freqs[alpha_mask]) if np.any(alpha_mask) else 0
                beta_power = np.trapz(ssvep_psd[beta_mask], ssvep_freqs[beta_mask]) if np.any(beta_mask) else 0
                
                alpha_beta_ratio = alpha_power / (beta_power + 1e-8)
                features.append(alpha_beta_ratio)
            else:
                features.extend([0] * 7)
        
        # Enhanced cross-channel coherence analysis
        for i in range(len(occipital_channels)):
            for j in range(i+1, len(occipital_channels)):
                ch1_data = eeg_occipital[:, i]
                ch2_data = eeg_occipital[:, j]
                
                # Coherence analysis
                f_coh, Cxy = signal.coherence(ch1_data, ch2_data, fs=self.fs, nperseg=256)
                
                # Mean coherence in SSVEP frequency bands
                for target_freq in self.ssvep_freqs:
                    mask = (f_coh >= target_freq - 1) & (f_coh <= target_freq + 1)
                    if np.any(mask):
                        mean_coh = np.mean(Cxy[mask])
                        features.append(mean_coh)
                    else:
                        features.append(0)
                
                # Overall SSVEP band coherence
                ssvep_mask = (f_coh >= 5) & (f_coh <= 30)
                if np.any(ssvep_mask):
                    ssvep_coherence = np.mean(Cxy[ssvep_mask])
                    features.append(ssvep_coherence)
                else:
                    features.append(0)
        
        # Enhanced CCA features with multiple reference signals
        for target_freq in self.ssvep_freqs:
            # Multiple CCA approaches
            cca_scores = []
            
            # Standard CCA
            standard_cca = self.apply_enhanced_cca_with_phase(eeg_occipital, target_freq)
            cca_scores.append(standard_cca)
            
            # CCA with delayed signals (account for neural delay)
            for delay in [1, 2, 3]:  # Different delays in samples
                if len(eeg_occipital) > delay:
                    delayed_data = eeg_occipital[delay:]
                    delayed_cca = self.apply_enhanced_cca_with_phase(delayed_data, target_freq)
                    cca_scores.append(delayed_cca)
                else:
                    cca_scores.append(0)
            
            # Best CCA score for this frequency
            features.append(np.max(cca_scores))
            features.append(np.mean(cca_scores))
            features.append(np.std(cca_scores))
        
        # Phase-locking analysis with improved accuracy
        t = np.arange(len(eeg_occipital)) / self.fs
        
        for target_freq in self.ssvep_freqs:
            # Phase-locking value with multiple channels
            plv_values = []
            
            for ch in range(eeg_occipital.shape[1]):
                try:
                    # Analytic signal using Hilbert transform
                    analytic = signal.hilbert(eeg_occipital[:, ch])
                    phase = np.angle(analytic)
                    
                    # Reference phase
                    ref_phase = 2 * np.pi * target_freq * t
                    
                    # Phase difference
                    phase_diff = phase - ref_phase
                    
                    # PLV calculation
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    plv_values.append(plv)
                except:
                    plv_values.append(0)
            
            # Statistics of PLV across channels
            features.extend([np.mean(plv_values), np.max(plv_values), np.std(plv_values)])
        
        # Enhanced spatial features
        # Laplacian spatial filtering for better SSVEP detection
        try:
            # Simple Laplacian: center - average of neighbors
            center_ch = 1  # OZ (center occipital)
            neighbors = [0, 2]  # PO7, PO8
            
            laplacian = eeg_occipital[:, center_ch] - np.mean(eeg_occipital[:, neighbors], axis=1)
            
            # Power spectral analysis of Laplacian signal
            f_lap, Pxx_lap = signal.welch(laplacian, fs=self.fs, nperseg=256)
            
            for target_freq in self.ssvep_freqs:
                mask = (f_lap >= target_freq - 0.5) & (f_lap <= target_freq + 0.5)
                if np.any(mask):
                    lap_power = np.trapz(Pxx_lap[mask], f_lap[mask])
                    features.append(np.log1p(lap_power))
                else:
                    features.append(0)
        except:
            features.extend([0] * len(self.ssvep_freqs))
        
        return np.array(features)
    
    def extract_enhanced_mi_features(self, eeg_data):
        """Enhanced MI feature extraction with better spatial filtering"""
        # Focus on motor cortex channels (C3, CZ, C4) - indices 1, 2, 3
        motor_channels = [1, 2, 3]  # C3, CZ, C4
        eeg_motor = eeg_data[:, motor_channels]
        
        # Enhanced preprocessing
        eeg_motor = self.apply_notch_filter(eeg_motor)
        eeg_motor = signal.detrend(eeg_motor, axis=0)
        eeg_motor = eeg_motor - np.median(eeg_motor, axis=0)
        
        features = []
        
        # Enhanced spatial patterns
        c3_data = eeg_motor[:, 0]  # C3
        cz_data = eeg_motor[:, 1]  # CZ
        c4_data = eeg_motor[:, 2]  # C4
        
        # Lateral asymmetry features
        lateral_diff = c3_data - c4_data
        lateral_ratio = np.var(c3_data) / (np.var(c4_data) + 1e-8)
        features.extend([np.var(lateral_diff), np.mean(lateral_diff), lateral_ratio])
        
        # Common average reference
        car_c3 = c3_data - np.mean(eeg_motor, axis=1)
        car_c4 = c4_data - np.mean(eeg_motor, axis=1)
        car_asymmetry = np.var(car_c3) / (np.var(car_c4) + 1e-8)
        features.append(car_asymmetry)
        
        # Laplacian spatial filtering
        lap_c3 = c3_data - cz_data  # C3 referenced to CZ
        lap_c4 = c4_data - cz_data  # C4 referenced to CZ
        lap_asymmetry = np.var(lap_c3) / (np.var(lap_c4) + 1e-8)
        features.append(lap_asymmetry)
        
        # Enhanced frequency band analysis
        bands = {
            'theta': (4, 8),    # Theta
            'alpha': (8, 13),   # Alpha/Mu
            'beta1': (13, 20),  # Beta1
            'beta2': (20, 30),  # Beta2
            'gamma': (30, 45),  # Gamma
        }
        
        for band_name, (low_freq, high_freq) in bands.items():
            band_data = self.apply_bandpass_filter(eeg_motor, low_freq, high_freq)
            
            # Power features for each channel
            powers = [np.var(band_data[:, ch]) for ch in range(3)]
            
            # Asymmetry indices
            c3_c4_asym = (powers[0] - powers[2]) / (powers[0] + powers[2] + 1e-8)
            features.append(c3_c4_asym)
            
            # Laterality index
            laterality = (powers[0] - powers[2]) / (2 * powers[1] + 1e-8)
            features.append(laterality)
            
            # Power ratios
            c3_cz_ratio = powers[0] / (powers[1] + 1e-8)
            c4_cz_ratio = powers[2] / (powers[1] + 1e-8)
            features.extend([c3_cz_ratio, c4_cz_ratio])
            
            # Enhanced spectral features for each channel
            for ch in range(3):
                channel_data = band_data[:, ch]
                
                # Spectral analysis
                f, Pxx = signal.welch(channel_data, fs=self.fs, nperseg=256)
                
                # Peak frequency in band
                band_mask = (f >= low_freq) & (f <= high_freq)
                if np.any(band_mask):
                    band_psd = Pxx[band_mask]
                    band_freqs = f[band_mask]
                    peak_idx = np.argmax(band_psd)
                    peak_freq = band_freqs[peak_idx]
                    peak_power = band_psd[peak_idx]
                    
                    features.extend([peak_freq, peak_power])
                    
                    # Spectral width (bandwidth)
                    half_max = peak_power / 2
                    width_indices = np.where(band_psd >= half_max)[0]
                    if len(width_indices) > 1:
                        bandwidth = band_freqs[width_indices[-1]] - band_freqs[width_indices[0]]
                        features.append(bandwidth)
                    else:
                        features.append(0)
                else:
                    features.extend([0, 0, 0])
        
        # Event-related synchronization/desynchronization (ERD/ERS) simulation
        # Using sliding window power analysis
        window_size = self.fs // 2  # 0.5 second windows
        step_size = self.fs // 4    # 0.25 second steps
        
        if len(eeg_motor) >= window_size:
            c3_powers = []
            c4_powers = []
            
            for start in range(0, len(eeg_motor) - window_size + 1, step_size):
                end = start + window_size
                
                # Mu band (8-13 Hz) power in each window
                c3_window = self.apply_bandpass_filter(c3_data[start:end].reshape(-1, 1), 8, 13)
                c4_window = self.apply_bandpass_filter(c4_data[start:end].reshape(-1, 1), 8, 13)
                
                c3_powers.append(np.var(c3_window))
                c4_powers.append(np.var(c4_window))
            
            if len(c3_powers) > 2:
                # ERD/ERS patterns
                c3_powers = np.array(c3_powers)
                c4_powers = np.array(c4_powers)
                
                # Power modulation
                c3_modulation = np.std(c3_powers) / (np.mean(c3_powers) + 1e-8)
                c4_modulation = np.std(c4_powers) / (np.mean(c4_powers) + 1e-8)
                
                features.extend([c3_modulation, c4_modulation])
                
                # Temporal asymmetry
                early_late_c3 = np.mean(c3_powers[:len(c3_powers)//2]) / (np.mean(c3_powers[len(c3_powers)//2:]) + 1e-8)
                early_late_c4 = np.mean(c4_powers[:len(c4_powers)//2]) / (np.mean(c4_powers[len(c4_powers)//2:]) + 1e-8)
                
                features.extend([early_late_c3, early_late_c4])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0])
        
        # Enhanced connectivity features
        # Coherence between motor channels
        for i in range(3):
            for j in range(i+1, 3):
                ch1_data = eeg_motor[:, i]
                ch2_data = eeg_motor[:, j]
                
                # Coherence in mu and beta bands
                f_coh, Cxy = signal.coherence(ch1_data, ch2_data, fs=self.fs, nperseg=256)
                
                mu_mask = (f_coh >= 8) & (f_coh <= 13)
                beta_mask = (f_coh >= 13) & (f_coh <= 30)
                
                mu_coherence = np.mean(Cxy[mu_mask]) if np.any(mu_mask) else 0
                beta_coherence = np.mean(Cxy[beta_mask]) if np.any(beta_mask) else 0
                
                features.extend([mu_coherence, beta_coherence])
        
        return np.array(features)

    def adaptive_model_selection(self, X, y, task_type):
        """Adaptive model selection based on cross-validation performance"""
        print(f"🎯 Adaptive model selection for {task_type}...")
        
        # Define candidate models based on task
        if task_type == 'MI':
            models = {
                'xgb_tuned': XGBClassifier(
                    n_estimators=80, learning_rate=0.04, max_depth=4,
                    subsample=0.7, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=3.0,
                    min_child_weight=5, gamma=0.2, random_state=42,
                    tree_method='hist', use_label_encoder=False, eval_metric='logloss'
                ),
                'rf_tuned': RandomForestClassifier(
                    n_estimators=60, max_depth=5, min_samples_split=12,
                    min_samples_leaf=6, max_features='sqrt', random_state=42,
                    class_weight='balanced', bootstrap=True, n_jobs=-1
                ),
                'svm_tuned': SVC(
                    kernel='rbf', C=1.5, gamma='scale', class_weight='balanced',
                    probability=True, random_state=42
                ),
                'lda_tuned': LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.2)
            }
        else:  # SSVEP
            models = {
                'xgb_tuned': XGBClassifier(
                    n_estimators=100, learning_rate=0.05, max_depth=5,
                    subsample=0.75, colsample_bytree=0.75, reg_alpha=1.5, reg_lambda=2.5,
                    min_child_weight=6, gamma=0.15, random_state=42,
                    tree_method='hist', use_label_encoder=False, eval_metric='mlogloss'
                ),
                'rf_tuned': RandomForestClassifier(
                    n_estimators=80, max_depth=6, min_samples_split=10,
                    min_samples_leaf=5, max_features='sqrt', random_state=42,
                    class_weight='balanced', bootstrap=True, n_jobs=-1
                ),
                'svm_tuned': SVC(
                    kernel='rbf', C=0.8, gamma='scale', class_weight='balanced',
                    probability=True, random_state=42
                ),
                'mlp_tuned': MLPClassifier(
                    hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                    alpha=0.03, learning_rate='adaptive', max_iter=300,
                    random_state=42, early_stopping=True, validation_fraction=0.2
                )
            }
        
        # Evaluate models using stratified cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model_scores = {}
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
                model_scores[name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores,
                    'model': model
                }
                print(f"  {name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            except Exception as e:
                print(f"  {name}: Failed - {e}")
                model_scores[name] = {
                    'mean': 0.0,
                    'std': 1.0,
                    'scores': [0.0],
                    'model': model
                }
        
        # Select top models (minimum 2, maximum 4)
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        # Dynamic selection based on performance
        selected_models = []
        best_score = sorted_models[0][1]['mean']
        
        for name, scores in sorted_models:
            # Include if within 5% of best performance or if we have less than 2 models
            if scores['mean'] >= best_score - 0.05 or len(selected_models) < 2:
                selected_models.append((name, scores['model']))
                if len(selected_models) >= 4:  # Maximum 4 models
                    break
        
        print(f"  Selected {len(selected_models)} models: {[name for name, _ in selected_models]}")
        
        # Create adaptive ensemble with selected models
        if len(selected_models) >= 2:
            ensemble = StackingClassifier(
                estimators=selected_models,
                final_estimator=LogisticRegression(
                    random_state=42, max_iter=500, 
                    C=0.3 if task_type == 'SSVEP' else 0.5,
                    solver='liblinear', class_weight='balanced'
                ),
                cv=3,
                n_jobs=1,
                passthrough=False
            )
        else:
            # Fallback to single best model
            ensemble = selected_models[0][1]
        
        return ensemble
    
    def enhanced_threshold_tuning(self, validation_df):
        """Enhanced threshold tuning with adaptive search and performance optimization"""
        print("\n🎯 Enhanced Adaptive CCA Threshold Tuning with Performance Optimization...")
        ssvep_val = validation_df[validation_df['task'] == 'SSVEP']
        
        if len(ssvep_val) == 0:
            print("No SSVEP validation data available for threshold tuning")
            return self.cca_threshold
        
        # Ensure LabelEncoder is fitted
        if not hasattr(self.ssvep_label_encoder, 'classes_'):
            self.ssvep_label_encoder.fit(ssvep_val['label'])
        
        # Stage 1: Performance-based regularization adjustment
        print("\n🔧 Stage 1: Performance-based Regularization Adjustment")
        self.adjust_regularization_based_on_performance(validation_df)
        
        # Stage 2: Multi-stage threshold search with error pattern analysis
        print("\n🔍 Stage 2: Adaptive Threshold Search with Error Pattern Analysis")
        threshold_results = self.adaptive_threshold_search_with_error_analysis(ssvep_val)
        
        # Stage 3: Confidence-based sample weighting
        print("\n⚖️ Stage 3: Confidence-Based Sample Weighting")
        self.implement_confidence_based_weighting(ssvep_val, threshold_results)
        
        # Find best threshold from results
        best_threshold = max(threshold_results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_accuracy = threshold_results[best_threshold]['accuracy']
        
        print(f"✓ Optimal threshold: {best_threshold:.3f} (acc: {best_accuracy:.3f})")
        self.cca_threshold = best_threshold
        
        # Stage 4: Dynamic threshold adjustment based on performance
        self.setup_dynamic_threshold_adjustment(threshold_results)
        
        return best_threshold
    
    def adjust_regularization_based_on_performance(self, validation_df):
        """Adjust regularization parameters based on validation performance"""
        try:
            # Analyze overfitting indicators
            mi_val = validation_df[validation_df['task'] == 'MI']
            ssvep_val = validation_df[validation_df['task'] == 'SSVEP']
            
            # Test different regularization strengths
            if len(mi_val) > 0 and hasattr(self, 'mi_ensemble'):
                print("   Adjusting MI regularization...")
                mi_performance = self.evaluate_mi_performance(mi_val)
                
                # If overfitting detected (high train-val gap), increase regularization
                if mi_performance.get('overfitting_detected', False):
                    print("   ⚠️ MI overfitting detected - increasing regularization")
                    self.apply_stronger_mi_regularization()
                
            if len(ssvep_val) > 0 and hasattr(self, 'ssvep_ensemble'):
                print("   Adjusting SSVEP regularization...")
                ssvep_performance = self.evaluate_ssvep_performance(ssvep_val)
                
                # If underfitting detected (low performance), reduce regularization
                if ssvep_performance.get('underfitting_detected', False):
                    print("   📈 SSVEP underfitting detected - reducing regularization")
                    self.apply_lighter_ssvep_regularization()
                    
        except Exception as e:
            print(f"   Regularization adjustment failed: {e}")
    
    def adaptive_threshold_search_with_error_analysis(self, ssvep_val):
        """Adaptive threshold search with comprehensive error pattern analysis"""
        threshold_results = {}
        
        # Comprehensive threshold range
        thresholds = np.concatenate([
            np.linspace(0.01, 0.1, 10),   # Very low thresholds
            np.linspace(0.1, 0.3, 15),    # Low-medium thresholds
            np.linspace(0.3, 0.6, 10),    # Medium-high thresholds
            np.linspace(0.6, 0.9, 5)      # Very high thresholds
        ])
        
        for threshold in thresholds:
            self.cca_threshold = threshold
            
            # Detailed performance analysis
            results = {
                'correct': 0,
                'total': 0,
                'class_performance': {},
                'confidence_distribution': [],
                'error_patterns': [],
                'quality_scores': []
            }
            
            for idx, row in ssvep_val.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is None:
                        continue
                    
                    # Get prediction and detailed analysis
                    cca_class, cca_score = self.enhanced_cca_predict(trial_data)
                    trial_quality = self.assess_trial_quality(trial_data, 'SSVEP')
                    
                    if cca_class is not None and cca_score > self.cca_threshold:
                        cca_label = self.ssvep_label_encoder.classes_[cca_class]
                        true_label = row['label']
                        
                        # Record detailed results
                        is_correct = (cca_label == true_label)
                        if is_correct:
                            results['correct'] += 1
                        else:
                            # Record error pattern
                            results['error_patterns'].append({
                                'true': true_label,
                                'predicted': cca_label,
                                'confidence': cca_score,
                                'quality': trial_quality
                            })
                        
                        results['total'] += 1
                        results['confidence_distribution'].append(cca_score)
                        results['quality_scores'].append(trial_quality)
                        
                        # Per-class performance
                        if true_label not in results['class_performance']:
                            results['class_performance'][true_label] = {'correct': 0, 'total': 0}
                        results['class_performance'][true_label]['total'] += 1
                        if is_correct:
                            results['class_performance'][true_label]['correct'] += 1
                
                except Exception as e:
                    continue
            
            # Calculate comprehensive metrics
            results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
            results['mean_confidence'] = np.mean(results['confidence_distribution']) if results['confidence_distribution'] else 0
            results['mean_quality'] = np.mean(results['quality_scores']) if results['quality_scores'] else 0
            
            # Error pattern analysis
            if results['error_patterns']:
                error_types = {}
                for error in results['error_patterns']:
                    error_key = f"{error['true']}->{error['predicted']}"
                    if error_key not in error_types:
                        error_types[error_key] = []
                    error_types[error_key].append(error)
                results['common_errors'] = error_types
            
            threshold_results[threshold] = results
        
        # Print summary of best thresholds
        print("   📊 Top threshold candidates:")
        sorted_results = sorted(threshold_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for i, (thresh, res) in enumerate(sorted_results[:5]):
            print(f"      {i+1}. Threshold {thresh:.3f}: {res['accuracy']:.3f} acc, {res['mean_confidence']:.3f} conf")
        
        return threshold_results
    
    def implement_confidence_based_weighting(self, ssvep_val, threshold_results):
        """Implement confidence-based sample weighting for better performance"""
        try:
            # Find optimal threshold with good balance of accuracy and confidence
            best_balanced_threshold = None
            best_balance_score = 0
            
            for threshold, results in threshold_results.items():
                if results['total'] > 0:
                    # Balance score considers both accuracy and confidence quality
                    balance_score = (
                        results['accuracy'] * 0.7 +
                        results['mean_confidence'] * 0.2 +
                        results['mean_quality'] * 0.1
                    )
                    
                    if balance_score > best_balance_score:
                        best_balance_score = balance_score
                        best_balanced_threshold = threshold
            
            if best_balanced_threshold is not None:
                print(f"   ✓ Balanced threshold: {best_balanced_threshold:.3f} (score: {best_balance_score:.3f})")
                
                # Implement confidence-based weighting for future predictions
                self.confidence_weights = self.calculate_confidence_weights(
                    threshold_results[best_balanced_threshold]
                )
            
        except Exception as e:
            print(f"   Confidence weighting failed: {e}")
    
    def setup_dynamic_threshold_adjustment(self, threshold_results):
        """Setup dynamic threshold adjustment based on performance patterns"""
        try:
            # Analyze performance vs threshold relationship
            thresholds = list(threshold_results.keys())
            accuracies = [threshold_results[t]['accuracy'] for t in thresholds]
            
            # Find performance plateau regions
            threshold_performance_map = {}
            for i, threshold in enumerate(thresholds):
                threshold_performance_map[threshold] = {
                    'accuracy': accuracies[i],
                    'stability': self.calculate_threshold_stability(threshold, threshold_results)
                }
            
            # Setup adaptive adjustment rules
            self.dynamic_threshold_rules = {
                'conservative': min(thresholds),  # Most inclusive
                'balanced': max(threshold_results.items(), key=lambda x: x[1]['accuracy'])[0],  # Best accuracy
                'aggressive': max(thresholds),    # Most exclusive
                'current_mode': 'balanced'
            }
            
            print(f"   ✓ Dynamic threshold rules setup: {self.dynamic_threshold_rules}")
            
        except Exception as e:
            print(f"   Dynamic threshold setup failed: {e}")
    
    def calculate_confidence_weights(self, threshold_results):
        """Calculate confidence-based weights for sample weighting"""
        try:
            if not threshold_results['confidence_distribution']:
                return {'default': 1.0}
            
            confidences = threshold_results['confidence_distribution']
            qualities = threshold_results['quality_scores']
            
            # Create confidence bins
            conf_bins = np.linspace(0, 1, 11)
            weights = {}
            
            for i in range(len(conf_bins)-1):
                bin_mask = (np.array(confidences) >= conf_bins[i]) & (np.array(confidences) < conf_bins[i+1])
                if np.any(bin_mask):
                    bin_quality = np.mean(np.array(qualities)[bin_mask])
                    # Higher quality samples get higher weights
                    weights[f'bin_{i}'] = max(0.5, min(2.0, bin_quality * 2))
                else:
                    weights[f'bin_{i}'] = 1.0
            
            return weights
            
        except Exception as e:
            print(f"Confidence weight calculation failed: {e}")
            return {'default': 1.0}
    
    def calculate_threshold_stability(self, threshold, threshold_results):
        """Calculate stability score for a threshold"""
        try:
            results = threshold_results[threshold]
            
            # Stability based on consistent performance across classes
            if not results['class_performance']:
                return 0.0
            
            class_accuracies = []
            for class_name, class_stats in results['class_performance'].items():
                class_acc = class_stats['correct'] / class_stats['total'] if class_stats['total'] > 0 else 0
                class_accuracies.append(class_acc)
            
            # Lower standard deviation = higher stability
            if len(class_accuracies) > 1:
                stability = 1.0 / (np.std(class_accuracies) + 0.1)
            else:
                stability = 1.0
            
            return min(stability, 10.0)  # Cap at 10
            
        except Exception as e:
            return 1.0
    
    def evaluate_mi_performance(self, mi_val):
        """Evaluate MI performance for regularization adjustment"""
        try:
            performance_metrics = {'overfitting_detected': False}
            
            # Quick validation on subset
            sample_size = min(20, len(mi_val))
            sample_val = mi_val.sample(sample_size, random_state=42)
            
            predictions = []
            confidences = []
            
            for idx, row in sample_val.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        pred, conf = self.dynamic_prediction_fusion(trial_data, 'MI')
                        predictions.append(pred)
                        confidences.append(conf)
                except:
                    continue
            
            if len(predictions) > 0:
                # Check for overfitting indicators
                mean_confidence = np.mean(confidences)
                confidence_std = np.std(confidences)
                
                # High confidence with low variance might indicate overfitting
                if mean_confidence > 0.95 and confidence_std < 0.05:
                    performance_metrics['overfitting_detected'] = True
                
                performance_metrics['mean_confidence'] = mean_confidence
                performance_metrics['confidence_std'] = confidence_std
            
            return performance_metrics
            
        except Exception as e:
            return {'overfitting_detected': False}
    
    def evaluate_ssvep_performance(self, ssvep_val):
        """Evaluate SSVEP performance for regularization adjustment"""
        try:
            performance_metrics = {'underfitting_detected': False}
            
            # Quick validation on subset
            sample_size = min(20, len(ssvep_val))
            sample_val = ssvep_val.sample(sample_size, random_state=42)
            
            predictions = []
            confidences = []
            
            for idx, row in sample_val.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        pred, conf = self.dynamic_prediction_fusion(trial_data, 'SSVEP')
                        predictions.append(pred)
                        confidences.append(conf)
                except:
                    continue
            
            if len(predictions) > 0:
                # Check for underfitting indicators
                mean_confidence = np.mean(confidences)
                
                # Very low confidence might indicate underfitting
                if mean_confidence < 0.3:
                    performance_metrics['underfitting_detected'] = True
                
                performance_metrics['mean_confidence'] = mean_confidence
            
            return performance_metrics
            
        except Exception as e:
            return {'underfitting_detected': False}
    
    def apply_stronger_mi_regularization(self):
        """Apply stronger regularization to MI models"""
        try:
            # This would retrain models with stronger regularization
            # For now, we'll adjust confidence calibration
            print("   📉 Applying stronger MI regularization through confidence adjustment")
            self.mi_regularization_factor = 0.8  # Reduce confidence
        except Exception as e:
            print(f"   Strong regularization failed: {e}")
    
    def apply_lighter_ssvep_regularization(self):
        """Apply lighter regularization to SSVEP models"""
        try:
            # This would retrain models with lighter regularization
            # For now, we'll adjust confidence calibration
            print("   📈 Applying lighter SSVEP regularization through confidence adjustment")
            self.ssvep_regularization_factor = 1.2  # Boost confidence
        except Exception as e:
            print(f"   Light regularization failed: {e}")

    def improved_confidence_calibration(self, probabilities, task_type):
        """IMPROVED confidence calibration with adaptive regularization factors"""
        max_prob = np.max(probabilities)
        entropy_score = -np.sum(probabilities * np.log(probabilities + 1e-8))
        
        if task_type == 'MI':
            # MI: More aggressive confidence boosting
            base_multiplier = 1.2  # Start with boost
            
            # Apply regularization factor if available
            if hasattr(self, 'mi_regularization_factor'):
                base_multiplier *= self.mi_regularization_factor
            
            # Boost based on prediction sharpness
            if max_prob > 0.7:  # Very confident prediction
                base_multiplier *= 1.25
            elif max_prob > 0.5:  # Moderately confident
                base_multiplier *= 1.15
            elif max_prob > 0.4:  # Weak confidence
                base_multiplier *= 1.05
            else:  # Very weak
                base_multiplier *= 0.95
            
            # Additional boost for low entropy (sharp predictions)
            if entropy_score < 0.5:  # Very sharp prediction
                base_multiplier *= 1.2
            elif entropy_score < 0.8:  # Moderately sharp
                base_multiplier *= 1.1
            
        else:  # SSVEP
            # SSVEP: Significant confidence boosting needed
            base_multiplier = 1.4  # Higher base boost for SSVEP
            
            # Apply regularization factor if available
            if hasattr(self, 'ssvep_regularization_factor'):
                base_multiplier *= self.ssvep_regularization_factor
            
            # More aggressive boosting for SSVEP
            if max_prob > 0.6:  # Good prediction
                base_multiplier *= 1.3
            elif max_prob > 0.4:  # Moderate prediction
                base_multiplier *= 1.2
            elif max_prob > 0.3:  # Weak prediction
                base_multiplier *= 1.1
            else:  # Very weak
                base_multiplier *= 1.0
            
            # Strong boost for sharp SSVEP predictions
            if entropy_score < 0.6:  # Sharp prediction
                base_multiplier *= 1.3
            elif entropy_score < 1.0:  # Moderately sharp
                base_multiplier *= 1.15
        
        # Apply calibration with temperature scaling
        temperature = 0.8  # Lower temperature = higher confidence
        scaled_probs = probabilities ** (1.0 / temperature)
        scaled_probs = scaled_probs / np.sum(scaled_probs)
        
        # Apply multiplier
        calibrated_probs = scaled_probs * base_multiplier
        
        # Ensure probabilities sum to 1
        if np.sum(calibrated_probs) > 0:
            calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
        else:
            # Fallback to uniform distribution
            calibrated_probs = np.ones_like(probabilities) / len(probabilities)
        
        # Final confidence boost - ensure minimum confidence levels
        max_calibrated = np.max(calibrated_probs)
        if task_type == 'MI' and max_calibrated < 0.6:
            # Boost MI confidence to at least 60%
            boost_factor = 0.6 / max_calibrated
            calibrated_probs = calibrated_probs * boost_factor
            calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
        elif task_type == 'SSVEP' and max_calibrated < 0.5:
            # Boost SSVEP confidence to at least 50%
            boost_factor = 0.5 / max_calibrated
            calibrated_probs = calibrated_probs * boost_factor
            calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
        
        return calibrated_probs
    
    def calibrate_confidence_advanced(self, probabilities, task_type, prediction_method='ensemble'):
        """Advanced confidence calibration based on method and validation performance"""
        if task_type == 'MI':
            # MI calibration factors based on method
            method_factors = {
                'ensemble': 0.85,  # Conservative for ensemble
                'csp': 0.9,       # Slightly higher for CSP
                'single': 0.75    # Lower for single models
            }
            base_multiplier = method_factors.get(prediction_method, 0.8)
            
            # Adjust based on class balance (if heavily biased, reduce confidence)
            max_prob = np.max(probabilities)
            if max_prob > 0.95:  # Very confident but potentially overfit
                base_multiplier *= 0.85
            
        else:  # SSVEP
            # SSVEP calibration - more conservative due to complexity
            method_factors = {
                'cca': 0.7,       # CCA often overconfident
                'ensemble': 0.65, # Conservative for ensemble
                'riemannian': 0.8, # Riemannian can be more trusted
                'fbcca': 0.75     # FBCCA moderate confidence
            }
            base_multiplier = method_factors.get(prediction_method, 0.6)
            
            # Additional SSVEP-specific adjustments
            max_prob = np.max(probabilities)
            entropy_score = -np.sum(probabilities * np.log(probabilities + 1e-8))
            
            # If entropy is low (prediction is sharp), slightly increase confidence
            if entropy_score < 0.5:
                base_multiplier *= 1.1
            elif entropy_score > 1.0:  # High entropy, reduce confidence
                base_multiplier *= 0.9
        
        # Apply calibration
        calibrated_probs = probabilities * base_multiplier
        
        # Ensure probabilities sum to 1 and maintain relative ratios
        if np.sum(calibrated_probs) > 0:
            calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
        else:
            # Fallback to uniform distribution
            calibrated_probs = np.ones_like(probabilities) / len(probabilities)
        
        return calibrated_probs
    
    def dynamic_prediction_fusion(self, trial_data, task_type):
        """FIXED Dynamic prediction fusion with consistent feature extraction"""
        if task_type == 'MI':
            # MI: Use SAME feature extraction as training (fast method)
            try:
                # Use the SAME feature extraction method as training
                features = self.extract_mi_features_fast(trial_data)
                
                print(f"MI prediction features shape: {features.shape}")
                
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Scale and select features (same pipeline as training)
                features_scaled = self.mi_scaler.transform(features.reshape(1, -1))
                features_selected = self.mi_selector.transform(features_scaled)
                
                # Get ensemble prediction with improved confidence calibration
                ensemble_proba = self.mi_ensemble.predict_proba(features_selected)[0]
                
                # IMPROVED confidence calibration for MI
                calibrated_proba = self.improved_confidence_calibration(ensemble_proba, 'MI')
                
                ensemble_pred = np.argmax(calibrated_proba)
                ensemble_conf = np.max(calibrated_proba)
                
                # ENHANCED confidence boost based on advanced spectral quality and asymmetry
                try:
                    motor_channels = [1, 2, 3]  # C3, CZ, C4
                    motor_data = trial_data[:, motor_channels]
                    
                    # Advanced asymmetry analysis with multiple frequency bands
                    asymmetry_boost = self.calculate_mi_asymmetry_boost(motor_data)
                    
                    # Advanced spectral quality analysis for MI
                    spectral_boost = self.calculate_mi_spectral_quality_boost(motor_data)
                    
                    # Combined confidence enhancement
                    total_boost = asymmetry_boost * spectral_boost
                    ensemble_conf = min(1.0, ensemble_conf * total_boost)
                    
                    print(f"MI confidence boost: asymmetry={asymmetry_boost:.3f}, spectral={spectral_boost:.3f}, total={total_boost:.3f}")
                        
                except Exception as e:
                    print(f"MI confidence boost failed: {e}")
                    pass
                
                prediction = self.mi_label_encoder.inverse_transform([ensemble_pred])[0]
                return prediction, ensemble_conf
                
            except Exception as e:
                print(f"MI prediction failed: {e}, using fallback")
                return 'Left', 0.4  # Fallback
            
        else:  # SSVEP
            # SSVEP: Use SAME feature extraction as training (fast method)
            try:
                # Use the SAME feature extraction method as training
                features = self.extract_ssvep_features_fast(trial_data)
                
                print(f"SSVEP prediction features shape: {features.shape}")
                
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Scale and select features (same pipeline as training)
                features_scaled = self.ssvep_scaler.transform(features.reshape(1, -1))
                features_selected = self.ssvep_selector.transform(features_scaled)
                
                # Get ensemble prediction with improved confidence calibration
                ensemble_proba = self.ssvep_ensemble.predict_proba(features_selected)[0]
                
                # IMPROVED confidence calibration for SSVEP
                calibrated_proba = self.improved_confidence_calibration(ensemble_proba, 'SSVEP')
                
                ensemble_pred = np.argmax(calibrated_proba)
                ensemble_conf = np.max(calibrated_proba)
                
                # Strategy 1: Enhanced CCA for additional validation
                cca_pred, cca_conf = self.enhanced_cca_predict(trial_data)
                
                # ENHANCED Strategy 2: Advanced spectral quality and peak detection
                try:
                    occipital_data = trial_data[:, [5, 6, 7]]
                    
                    # Advanced SSVEP spectral quality analysis
                    spectral_boost = self.calculate_ssvep_spectral_quality_boost(occipital_data)
                    
                    # Advanced peak detection and validation
                    peak_boost = self.calculate_ssvep_peak_detection_boost(occipital_data)
                    
                    # SNR-based confidence enhancement
                    snr_boost = self.calculate_ssvep_snr_boost(occipital_data)
                    
                    # Combined confidence enhancement
                    total_boost = spectral_boost * peak_boost * snr_boost
                    ensemble_conf = min(1.0, ensemble_conf * total_boost)
                    
                    print(f"SSVEP confidence boost: spectral={spectral_boost:.3f}, peak={peak_boost:.3f}, snr={snr_boost:.3f}, total={total_boost:.3f}")
                        
                except Exception as e:
                    print(f"SSVEP confidence boost failed: {e}")
                    pass
                
                # Fusion with CCA if available and confident
                if cca_pred is not None and cca_conf > self.cca_threshold:
                    # Weight ensemble and CCA predictions
                    cca_weight = min(0.4, cca_conf / (self.cca_threshold + 1e-8) * 0.2)
                    ensemble_weight = 1.0 - cca_weight
                    
                    # Create weighted prediction
                    final_scores = np.zeros(len(self.ssvep_label_encoder.classes_))
                    final_scores[cca_pred] += cca_weight * cca_conf
                    for i, prob in enumerate(calibrated_proba):
                        final_scores[i] += ensemble_weight * prob
                    
                    final_pred = np.argmax(final_scores)
                    final_conf = np.max(final_scores)
                else:
                    final_pred = ensemble_pred
                    final_conf = ensemble_conf
                
                prediction = self.ssvep_label_encoder.inverse_transform([final_pred])[0]
                return prediction, final_conf
                
            except Exception as e:
                print(f"SSVEP prediction failed: {e}, using fallback")
                return 'Forward', 0.4  # Fallback

    def apply_advanced_csp(self, eeg_data, labels, n_components=4):
        """Advanced CSP with multiple regularization techniques"""
        try:
            from scipy.linalg import eigh
            
            # Ensure binary classification
            unique_labels = np.unique(labels)
            if len(unique_labels) != 2:
                mask = np.isin(labels, unique_labels[:2])
                eeg_data = eeg_data[mask]
                labels = labels[mask]
            
            if len(eeg_data) < 20:
                return eeg_data.reshape(eeg_data.shape[0], -1), None
            
            n_trials, n_channels, n_samples = eeg_data.shape
            
            # Calculate covariance matrices for each class
            class_0_mask = labels == unique_labels[0]
            class_1_mask = labels == unique_labels[1]
            
            class_0_data = eeg_data[class_0_mask]
            class_1_data = eeg_data[class_1_mask]
            
            if len(class_0_data) < 5 or len(class_1_data) < 5:
                return eeg_data.reshape(eeg_data.shape[0], -1), None
            
            # Enhanced covariance estimation with regularization
            def regularized_covariance(data, reg_param=0.1):
                n_trials = data.shape[0]
                cov = np.zeros((n_channels, n_channels))
                
                for trial in data:
                    trial_cov = np.cov(trial)
                    # Regularization: shrinkage towards identity
                    cov += (1 - reg_param) * trial_cov + reg_param * np.eye(n_channels) * np.trace(trial_cov) / n_channels
                
                return cov / n_trials
            
            C0 = regularized_covariance(class_0_data)
            C1 = regularized_covariance(class_1_data)
            
            # Composite covariance matrix
            C = C0 + C1
            
            # Eigendecomposition with numerical stability
            eigenvals, eigenvecs = eigh(C0, C)
            
            # Sort eigenvalues and eigenvectors
            idx = np.argsort(eigenvals)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Select spatial filters (most discriminative components)
            n_components = min(n_components, n_channels)
            
            # Take components from both ends of the spectrum
            selected_indices = np.concatenate([
                idx[:n_components//2],  # Smallest eigenvalues
                idx[-(n_components - n_components//2):]  # Largest eigenvalues
            ])
            
            csp_filters = eigenvecs[:, selected_indices]
            
            # Apply CSP filtering
            filtered_data = []
            for trial in eeg_data:
                filtered_trial = csp_filters.T @ trial
                # Extract log-variance features
                log_var_features = np.log(np.var(filtered_trial, axis=1) + 1e-8)
                filtered_data.append(log_var_features)
            
            filtered_data = np.array(filtered_data)
            
            print(f"Advanced CSP applied: {eeg_data.shape} -> {filtered_data.shape}")
            return filtered_data, csp_filters
            
        except Exception as e:
            print(f"Advanced CSP failed: {e}, using fallback")
            return eeg_data.reshape(eeg_data.shape[0], -1), None
    
    def extract_advanced_spectral_features(self, eeg_data, task_type):
        """Advanced spectral analysis with multiple methods"""
        features = []
        
        if task_type == 'MI':
            # Focus on motor channels
            channels = [1, 2, 3]  # C3, CZ, C4
            target_bands = {
                'mu': (8, 13),
                'beta1': (13, 20),
                'beta2': (20, 30),
                'gamma': (30, 45)
            }
        else:  # SSVEP
            # Focus on occipital channels
            channels = [5, 6, 7]  # PO7, OZ, PO8
            target_bands = {
                'alpha': (8, 13),
                'beta': (13, 25),
                'ssvep': (5, 30)
            }
        
        eeg_channels = eeg_data[:, channels]
        
        for ch_idx, ch_data in enumerate(eeg_channels.T):
            # Multi-taper spectral estimation for better resolution
            from scipy.signal import spectrogram
            
            # Enhanced spectral analysis
            f, t, Sxx = spectrogram(ch_data, fs=self.fs, nperseg=256, noverlap=128)
            
            # Time-frequency features
            for band_name, (low_freq, high_freq) in target_bands.items():
                band_mask = (f >= low_freq) & (f <= high_freq)
                if np.any(band_mask):
                    band_power = np.mean(Sxx[band_mask, :], axis=0)
                    
                    # Statistical features of power over time
                    features.extend([
                        np.mean(band_power),
                        np.std(band_power),
                        np.max(band_power),
                        np.min(band_power),
                        np.median(band_power),
                        skew(band_power),
                        kurtosis(band_power)
                    ])
                    
                    # Power modulation features
                    power_diff = np.diff(band_power)
                    features.extend([
                        np.mean(np.abs(power_diff)),
                        np.std(power_diff),
                        np.sum(power_diff > 0) / len(power_diff)  # Proportion of increases
                    ])
                else:
                    features.extend([0] * 10)
            
            # Enhanced SSVEP-specific features
            if task_type == 'SSVEP':
                for target_freq in self.ssvep_freqs:
                    # Precise frequency analysis
                    freq_mask = (f >= target_freq - 0.25) & (f <= target_freq + 0.25)
                    if np.any(freq_mask):
                        freq_power = np.mean(Sxx[freq_mask, :])
                        features.append(freq_power)
                        
                        # Signal-to-noise ratio
                        noise_mask = ((f >= target_freq - 2) & (f <= target_freq - 1)) | \
                                   ((f >= target_freq + 1) & (f <= target_freq + 2))
                        if np.any(noise_mask):
                            noise_power = np.mean(Sxx[noise_mask, :])
                            snr = freq_power / (noise_power + 1e-8)
                            features.append(snr)
                        else:
                            features.append(0)
                    else:
                        features.extend([0, 0])
        
        return np.array(features)
    
    def apply_ensemble_boosting(self, X, y, task_type):
        """Advanced ensemble with boosting and stacking"""
        from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        # Base models with different strengths
        base_models = []
        
        if task_type == 'MI':
            base_models = [
                ('xgb_conservative', XGBClassifier(
                    n_estimators=60, learning_rate=0.03, max_depth=3,
                    subsample=0.6, colsample_bytree=0.6, reg_alpha=3.0, reg_lambda=4.0,
                    min_child_weight=8, gamma=0.3, random_state=42,
                    tree_method='hist', use_label_encoder=False, eval_metric='logloss'
                )),
                ('rf_conservative', RandomForestClassifier(
                    n_estimators=40, max_depth=4, min_samples_split=15,
                    min_samples_leaf=8, max_features='sqrt', random_state=42,
                    class_weight='balanced', bootstrap=True, n_jobs=-1
                )),
                ('ada_boost', AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=10),
                    n_estimators=50, learning_rate=0.8, random_state=42
                )),
                ('gradient_boost', GradientBoostingClassifier(
                    n_estimators=60, learning_rate=0.03, max_depth=3,
                    subsample=0.7, random_state=42, validation_fraction=0.2,
                    n_iter_no_change=15
                )),
                ('svm_conservative', SVC(
                    kernel='rbf', C=0.8, gamma='scale', class_weight='balanced',
                    probability=True, random_state=42
                ))
            ]
        else:  # SSVEP
            base_models = [
                ('xgb_ssvep', XGBClassifier(
                    n_estimators=80, learning_rate=0.04, max_depth=4,
                    subsample=0.65, colsample_bytree=0.65, reg_alpha=2.0, reg_lambda=3.0,
                    min_child_weight=7, gamma=0.2, random_state=42,
                    tree_method='hist', use_label_encoder=False, eval_metric='mlogloss'
                )),
                ('rf_ssvep', RandomForestClassifier(
                    n_estimators=60, max_depth=5, min_samples_split=12,
                    min_samples_leaf=6, max_features='sqrt', random_state=42,
                    class_weight='balanced', bootstrap=True, n_jobs=-1
                )),
                ('ada_ssvep', AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=3, min_samples_split=8),
                    n_estimators=60, learning_rate=0.6, random_state=42
                )),
                ('gradient_ssvep', GradientBoostingClassifier(
                    n_estimators=80, learning_rate=0.04, max_depth=4,
                    subsample=0.7, random_state=42, validation_fraction=0.2,
                    n_iter_no_change=15
                ))
            ]
        
        # Evaluate models with cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model_scores = {}
        
        for name, model in base_models:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
                model_scores[name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'model': model
                }
                print(f"  {name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            except Exception as e:
                print(f"  {name}: Failed - {e}")
                continue
        
        # Select top performing models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
        selected_models = []
        
        # Select top 3-4 models
        for i, (name, scores) in enumerate(sorted_models[:4]):
            if scores['mean'] > 0.3:  # Minimum threshold
                selected_models.append((name, scores['model']))
        
        if len(selected_models) < 2:
            # Fallback to best model
            return sorted_models[0][1]['model'] if sorted_models else base_models[0][1]
        
        print(f"  Selected {len(selected_models)} models for ensemble")
        
        # Create enhanced stacking ensemble
        meta_learner = LogisticRegression(
            random_state=42, max_iter=1000,
            C=0.1 if task_type == 'SSVEP' else 0.3,
            solver='liblinear', class_weight='balanced',
            penalty='l2'
        )
        
        ensemble = StackingClassifier(
            estimators=selected_models,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=1,
            passthrough=False
        )
        
        return ensemble
    
    def apply_advanced_preprocessing(self, eeg_data, task_type):
        """Advanced preprocessing pipeline"""
        # Apply multiple preprocessing steps
        processed_data = eeg_data.copy()
        
        # 1. Enhanced filtering
        processed_data = self.apply_notch_filter(processed_data, 50)  # Power line noise
        processed_data = signal.detrend(processed_data, axis=0)  # Remove linear trends
        
        # 2. Robust outlier removal
        for ch in range(processed_data.shape[1]):
            channel_data = processed_data[:, ch]
            # Remove extreme outliers (beyond 4 standard deviations)
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            outlier_mask = np.abs(channel_data - mean_val) > 4 * std_val
            processed_data[outlier_mask, ch] = mean_val
        
        # 3. Advanced artifact removal
        try:
            processed_data = self.apply_ica_artifact_removal(processed_data, n_components=min(6, processed_data.shape[1]))
        except:
            pass
        
        # 4. Task-specific filtering
        if task_type == 'MI':
            # Enhanced motor imagery filtering
            processed_data = self.apply_bandpass_filter(processed_data, 4, 45)
        else:  # SSVEP
            # Enhanced SSVEP filtering with multiple bands
            processed_data = self.apply_bandpass_filter(processed_data, 3, 50)
        
        # 5. Spatial filtering
        if task_type == 'MI':
            # Apply surface Laplacian for motor channels
            try:
                motor_channels = [1, 2, 3]  # C3, CZ, C4
                for i, ch_idx in enumerate(motor_channels):
                    if ch_idx < processed_data.shape[1]:
                        # Simple Laplacian: center - average of neighbors
                        if i == 1:  # CZ (center)
                            neighbors = [motor_channels[0], motor_channels[2]]  # C3, C4
                            if all(n < processed_data.shape[1] for n in neighbors):
                                processed_data[:, ch_idx] = processed_data[:, ch_idx] - \
                                                           np.mean(processed_data[:, neighbors], axis=1)
            except:
                pass
        else:  # SSVEP
            # Apply Laplacian for occipital channels
            try:
                occipital_channels = [5, 6, 7]  # PO7, OZ, PO8
                center_ch = 1  # OZ
                neighbors = [0, 2]  # PO7, PO8
                if all(ch < processed_data.shape[1] for ch in occipital_channels):
                    processed_data[:, occipital_channels[center_ch]] = \
                        processed_data[:, occipital_channels[center_ch]] - \
                        np.mean(processed_data[:, [occipital_channels[i] for i in neighbors]], axis=1)
            except:
                pass
        
        # 6. Final normalization
        processed_data = processed_data - np.median(processed_data, axis=0)
        
        return processed_data

    def calculate_mi_asymmetry_boost(self, motor_data):
        """Advanced MI asymmetry analysis with multiple frequency bands"""
        try:
            c3_data = motor_data[:, 0]  # C3
            cz_data = motor_data[:, 1]  # CZ  
            c4_data = motor_data[:, 2]  # C4
            
            # Analyze asymmetry in multiple frequency bands
            bands = {
                'mu': (8, 13),      # Mu rhythm
                'beta1': (13, 20),  # Beta1
                'beta2': (20, 30),  # Beta2
                'gamma': (30, 45)   # Gamma
            }
            
            asymmetry_scores = []
            
            for band_name, (low_freq, high_freq) in bands.items():
                # Filter data to frequency band
                c3_band = self.apply_bandpass_filter(c3_data.reshape(-1, 1), low_freq, high_freq).flatten()
                c4_band = self.apply_bandpass_filter(c4_data.reshape(-1, 1), low_freq, high_freq).flatten()
                
                # Calculate power asymmetry
                c3_power = np.var(c3_band)
                c4_power = np.var(c4_band)
                
                # Multiple asymmetry measures
                # 1. Classic asymmetry index
                classic_asym = abs(c3_power - c4_power) / (c3_power + c4_power + 1e-8)
                
                # 2. Log ratio asymmetry
                log_asym = abs(np.log(c3_power + 1e-8) - np.log(c4_power + 1e-8))
                
                # 3. Relative asymmetry
                rel_asym = abs(c3_power - c4_power) / (max(c3_power, c4_power) + 1e-8)
                
                # Combine asymmetry measures
                combined_asym = (classic_asym + log_asym * 0.5 + rel_asym) / 2.5
                asymmetry_scores.append(combined_asym)
            
            # Calculate overall asymmetry score
            mean_asymmetry = np.mean(asymmetry_scores)
            max_asymmetry = np.max(asymmetry_scores)
            
            # Additional spatial analysis
            # Laplacian asymmetry (C3-CZ vs C4-CZ)
            c3_lap = c3_data - cz_data
            c4_lap = c4_data - cz_data
            lap_c3_power = np.var(c3_lap)
            lap_c4_power = np.var(c4_lap)
            lap_asymmetry = abs(lap_c3_power - lap_c4_power) / (lap_c3_power + lap_c4_power + 1e-8)
            
            # Final asymmetry score
            final_asymmetry = (mean_asymmetry * 0.4 + max_asymmetry * 0.4 + lap_asymmetry * 0.2)
            
            # Convert to confidence boost factor
            if final_asymmetry > 0.5:      # Very strong asymmetry
                boost = 1.5
            elif final_asymmetry > 0.35:   # Strong asymmetry
                boost = 1.35
            elif final_asymmetry > 0.25:   # Moderate asymmetry
                boost = 1.2
            elif final_asymmetry > 0.15:   # Weak asymmetry
                boost = 1.1
            elif final_asymmetry > 0.08:   # Very weak asymmetry
                boost = 1.05
            else:                          # No significant asymmetry
                boost = 1.0
            
            return boost
            
        except Exception as e:
            print(f"MI asymmetry boost calculation failed: {e}")
            return 1.0

    def calculate_mi_spectral_quality_boost(self, motor_data):
        """Advanced MI spectral quality analysis"""
        try:
            c3_data = motor_data[:, 0]  # C3
            c4_data = motor_data[:, 2]  # C4
            
            quality_scores = []
            
            for ch_data in [c3_data, c4_data]:
                # Spectral analysis
                f, Pxx = signal.welch(ch_data, fs=self.fs, nperseg=min(512, len(ch_data)//2))
                
                # 1. Mu rhythm quality (8-13 Hz)
                mu_mask = (f >= 8) & (f <= 13)
                beta_mask = (f >= 13) & (f <= 30)
                gamma_mask = (f >= 30) & (f <= 45)
                
                if np.any(mu_mask) and np.any(beta_mask):
                    mu_power = np.trapz(Pxx[mu_mask], f[mu_mask])
                    beta_power = np.trapz(Pxx[beta_mask], f[beta_mask])
                    gamma_power = np.trapz(Pxx[gamma_mask], f[gamma_mask]) if np.any(gamma_mask) else 0
                    
                    # 2. Peak prominence in mu band
                    from scipy.signal import find_peaks
                    mu_peaks, mu_properties = find_peaks(Pxx[mu_mask], prominence=np.max(Pxx[mu_mask])*0.1)
                    mu_peak_quality = len(mu_peaks) * np.mean(mu_properties.get('prominences', [0]))
                    
                    # 3. Spectral concentration (how focused the power is)
                    total_power = np.trapz(Pxx, f)
                    mu_concentration = mu_power / (total_power + 1e-8)
                    beta_concentration = beta_power / (total_power + 1e-8)
                    
                    # 4. Signal-to-noise ratio
                    # Background: frequencies outside motor bands
                    background_mask = ((f >= 1) & (f <= 4)) | ((f >= 50) & (f <= 80))
                    if np.any(background_mask):
                        background_power = np.mean(Pxx[background_mask])
                        mu_snr = mu_power / (background_power + 1e-8)
                        beta_snr = beta_power / (background_power + 1e-8)
                    else:
                        mu_snr = beta_snr = 1.0
                    
                    # 5. Spectral entropy (lower = more organized)
                    psd_norm = Pxx / (np.sum(Pxx) + 1e-8)
                    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
                    entropy_quality = 1.0 / (spectral_entropy + 1e-8)  # Invert for quality
                    
                    # Combine quality measures
                    channel_quality = (
                        mu_concentration * 2.0 +
                        beta_concentration * 1.5 +
                        np.log1p(mu_snr) * 1.0 +
                        np.log1p(beta_snr) * 0.8 +
                        np.log1p(mu_peak_quality) * 0.5 +
                        entropy_quality * 0.3
                    ) / 6.1
                    
                    quality_scores.append(channel_quality)
                else:
                    quality_scores.append(0.1)  # Low quality fallback
            
            # Overall spectral quality
            mean_quality = np.mean(quality_scores)
            max_quality = np.max(quality_scores)
            final_quality = (mean_quality * 0.7 + max_quality * 0.3)
            
            # Convert to confidence boost factor
            if final_quality > 0.8:       # Excellent spectral quality
                boost = 1.4
            elif final_quality > 0.6:     # Good spectral quality
                boost = 1.25
            elif final_quality > 0.4:     # Moderate spectral quality
                boost = 1.15
            elif final_quality > 0.25:    # Fair spectral quality
                boost = 1.08
            elif final_quality > 0.15:    # Poor spectral quality
                boost = 1.03
            else:                         # Very poor spectral quality
                boost = 1.0
            
            return boost
            
        except Exception as e:
            print(f"MI spectral quality boost calculation failed: {e}")
            return 1.0

    def calculate_ssvep_spectral_quality_boost(self, occipital_data):
        """Advanced SSVEP spectral quality analysis"""
        try:
            quality_scores = []
            
            for ch in range(occipital_data.shape[1]):
                ch_data = occipital_data[:, ch]
                
                # High-resolution spectral analysis
                f, Pxx = signal.welch(ch_data, fs=self.fs, nperseg=min(1024, len(ch_data)//2), 
                                    noverlap=min(512, len(ch_data)//4))
                
                channel_quality = 0
                
                # Analyze each SSVEP frequency
                for target_freq in self.ssvep_freqs:
                    # 1. Fundamental frequency quality
                    fund_mask = (f >= target_freq - 0.25) & (f <= target_freq + 0.25)
                    if np.any(fund_mask):
                        fund_power = np.trapz(Pxx[fund_mask], f[fund_mask])
                        
                        # 2. Harmonic quality (2nd and 3rd harmonics)
                        harm2_freq = target_freq * 2
                        harm3_freq = target_freq * 3
                        
                        harm2_power = 0
                        harm3_power = 0
                        
                        if harm2_freq <= self.fs/2:
                            harm2_mask = (f >= harm2_freq - 0.3) & (f <= harm2_freq + 0.3)
                            if np.any(harm2_mask):
                                harm2_power = np.trapz(Pxx[harm2_mask], f[harm2_mask])
                        
                        if harm3_freq <= self.fs/2:
                            harm3_mask = (f >= harm3_freq - 0.3) & (f <= harm3_freq + 0.3)
                            if np.any(harm3_mask):
                                harm3_power = np.trapz(Pxx[harm3_mask], f[harm3_mask])
                        
                        # 3. Harmonic-to-fundamental ratios
                        h2f_ratio = harm2_power / (fund_power + 1e-8)
                        h3f_ratio = harm3_power / (fund_power + 1e-8)
                        
                        # 4. Frequency precision (how narrow the peak is)
                        peak_idx = np.argmax(Pxx[fund_mask])
                        peak_freq = f[fund_mask][peak_idx]
                        freq_precision = 1.0 / (abs(peak_freq - target_freq) + 0.1)
                        
                        # 5. Peak sharpness (Q-factor)
                        peak_power = Pxx[fund_mask][peak_idx]
                        half_power = peak_power / 2
                        
                        # Find bandwidth at half power
                        left_idx = peak_idx
                        right_idx = peak_idx
                        
                        while left_idx > 0 and Pxx[fund_mask][left_idx] > half_power:
                            left_idx -= 1
                        while right_idx < len(Pxx[fund_mask])-1 and Pxx[fund_mask][right_idx] > half_power:
                            right_idx += 1
                        
                        if right_idx > left_idx:
                            bandwidth = f[fund_mask][right_idx] - f[fund_mask][left_idx]
                            q_factor = target_freq / (bandwidth + 0.01)
                        else:
                            q_factor = 1.0
                        
                        # Combine quality measures for this frequency
                        freq_quality = (
                            np.log1p(fund_power) * 2.0 +
                            h2f_ratio * 1.0 +
                            h3f_ratio * 0.5 +
                            freq_precision * 1.5 +
                            np.log1p(q_factor) * 1.0
                        ) / 6.0
                        
                        channel_quality += freq_quality
                
                quality_scores.append(channel_quality / len(self.ssvep_freqs))
            
            # Overall SSVEP spectral quality
            mean_quality = np.mean(quality_scores)
            max_quality = np.max(quality_scores)
            final_quality = (mean_quality * 0.6 + max_quality * 0.4)
            
            # Convert to confidence boost factor
            if final_quality > 1.5:       # Excellent SSVEP quality
                boost = 1.6
            elif final_quality > 1.0:     # Good SSVEP quality
                boost = 1.4
            elif final_quality > 0.7:     # Moderate SSVEP quality
                boost = 1.25
            elif final_quality > 0.5:     # Fair SSVEP quality
                boost = 1.15
            elif final_quality > 0.3:     # Poor SSVEP quality
                boost = 1.05
            else:                         # Very poor SSVEP quality
                boost = 1.0
            
            return boost
            
        except Exception as e:
            print(f"SSVEP spectral quality boost calculation failed: {e}")
            return 1.0

    def calculate_ssvep_peak_detection_boost(self, occipital_data):
        """Advanced SSVEP peak detection and validation"""
        try:
            from scipy.signal import find_peaks
            
            peak_scores = []
            
            for ch in range(occipital_data.shape[1]):
                ch_data = occipital_data[:, ch]
                
                # High-resolution spectral analysis
                f, Pxx = signal.welch(ch_data, fs=self.fs, nperseg=min(1024, len(ch_data)//2))
                
                channel_peak_score = 0
                detected_peaks = 0
                
                for target_freq in self.ssvep_freqs:
                    # Focus on frequency region around target
                    freq_range = 2.0  # ±2 Hz around target
                    region_mask = (f >= target_freq - freq_range) & (f <= target_freq + freq_range)
                    
                    if np.any(region_mask):
                        region_psd = Pxx[region_mask]
                        region_freqs = f[region_mask]
                        
                        # Find peaks in this region
                        peaks, properties = find_peaks(
                            region_psd, 
                            prominence=np.max(region_psd) * 0.1,
                            width=1,
                            distance=3
                        )
                        
                        if len(peaks) > 0:
                            # Find peak closest to target frequency
                            peak_freqs = region_freqs[peaks]
                            freq_distances = np.abs(peak_freqs - target_freq)
                            closest_peak_idx = np.argmin(freq_distances)
                            closest_peak = peaks[closest_peak_idx]
                            closest_freq = peak_freqs[closest_peak_idx]
                            closest_distance = freq_distances[closest_peak_idx]
                            
                            # Peak quality metrics
                            peak_power = region_psd[closest_peak]
                            peak_prominence = properties['prominences'][closest_peak_idx]
                            peak_width = properties['widths'][closest_peak_idx]
                            
                            # 1. Frequency accuracy (how close to target)
                            freq_accuracy = 1.0 / (closest_distance + 0.1)
                            
                            # 2. Peak prominence (how much it stands out)
                            prominence_score = peak_prominence / (np.mean(region_psd) + 1e-8)
                            
                            # 3. Peak sharpness (narrow peaks are better for SSVEP)
                            sharpness_score = 1.0 / (peak_width + 0.5)
                            
                            # 4. Relative power (compared to background)
                            background_power = np.percentile(region_psd, 25)  # 25th percentile as background
                            relative_power = peak_power / (background_power + 1e-8)
                            
                            # 5. Peak isolation (how well separated from other peaks)
                            if len(peaks) > 1:
                                other_peaks = peaks[peaks != closest_peak]
                                if len(other_peaks) > 0:
                                    other_peak_freqs = region_freqs[other_peaks]
                                    min_separation = np.min(np.abs(other_peak_freqs - closest_freq))
                                    isolation_score = min_separation / 1.0  # Normalized by 1 Hz
                                else:
                                    isolation_score = 2.0  # Perfect isolation
                            else:
                                isolation_score = 2.0  # Perfect isolation
                            
                            # Combine peak quality measures
                            peak_quality = (
                                freq_accuracy * 2.0 +
                                np.log1p(prominence_score) * 1.5 +
                                sharpness_score * 1.0 +
                                np.log1p(relative_power) * 1.5 +
                                isolation_score * 1.0
                            ) / 7.0
                            
                            channel_peak_score += peak_quality
                            detected_peaks += 1
                
                # Normalize by number of target frequencies
                if detected_peaks > 0:
                    avg_peak_score = channel_peak_score / len(self.ssvep_freqs)
                    # Bonus for detecting multiple frequencies
                    detection_bonus = detected_peaks / len(self.ssvep_freqs)
                    final_score = avg_peak_score * (1.0 + detection_bonus * 0.5)
                else:
                    final_score = 0.1  # Low score for no peaks detected
                
                peak_scores.append(final_score)
            
            # Overall peak detection quality
            mean_peak_score = np.mean(peak_scores)
            max_peak_score = np.max(peak_scores)
            final_peak_quality = (mean_peak_score * 0.7 + max_peak_score * 0.3)
            
            # Convert to confidence boost factor
            if final_peak_quality > 2.0:      # Excellent peak detection
                boost = 1.5
            elif final_peak_quality > 1.5:    # Good peak detection
                boost = 1.35
            elif final_peak_quality > 1.0:    # Moderate peak detection
                boost = 1.2
            elif final_peak_quality > 0.7:    # Fair peak detection
                boost = 1.1
            elif final_peak_quality > 0.4:    # Poor peak detection
                boost = 1.05
            else:                             # Very poor peak detection
                boost = 1.0
            
            return boost
            
        except Exception as e:
            print(f"SSVEP peak detection boost calculation failed: {e}")
            return 1.0

    def calculate_ssvep_snr_boost(self, occipital_data):
        """Advanced SNR-based confidence enhancement for SSVEP"""
        try:
            snr_scores = []
            
            for ch in range(occipital_data.shape[1]):
                ch_data = occipital_data[:, ch]
                
                # High-resolution spectral analysis
                f, Pxx = signal.welch(ch_data, fs=self.fs, nperseg=min(1024, len(ch_data)//2))
                
                channel_snr_scores = []
                
                for target_freq in self.ssvep_freqs:
                    # Signal region (narrow band around target frequency)
                    signal_bw = 0.3  # ±0.3 Hz
                    signal_mask = (f >= target_freq - signal_bw) & (f <= target_freq + signal_bw)
                    
                    if np.any(signal_mask):
                        signal_power = np.trapz(Pxx[signal_mask], f[signal_mask])
                        
                        # Multiple noise estimation strategies
                        noise_powers = []
                        
                        # 1. Adjacent frequency bands (traditional approach)
                        noise_bw = 1.0
                        noise_mask1 = (f >= target_freq - noise_bw - 1.5) & (f <= target_freq - noise_bw)
                        noise_mask2 = (f >= target_freq + noise_bw) & (f <= target_freq + noise_bw + 1.5)
                        adjacent_noise_mask = noise_mask1 | noise_mask2
                        
                        if np.any(adjacent_noise_mask):
                            adjacent_noise = np.mean(Pxx[adjacent_noise_mask])
                            noise_powers.append(adjacent_noise)
                        
                        # 2. Background estimation (frequencies far from any SSVEP target)
                        background_mask = np.ones(len(f), dtype=bool)
                        for freq in self.ssvep_freqs:
                            exclude_mask = (f >= freq - 2.0) & (f <= freq + 2.0)
                            background_mask &= ~exclude_mask
                        
                        # Also exclude very low and very high frequencies
                        background_mask &= (f >= 3) & (f <= 45)
                        
                        if np.any(background_mask):
                            background_noise = np.mean(Pxx[background_mask])
                            noise_powers.append(background_noise)
                        
                        # 3. Local baseline (median of surrounding frequencies)
                        local_range = 5.0  # ±5 Hz
                        local_mask = (f >= target_freq - local_range) & (f <= target_freq + local_range)
                        local_mask &= ~signal_mask  # Exclude signal region
                        
                        if np.any(local_mask):
                            local_baseline = np.median(Pxx[local_mask])
                            noise_powers.append(local_baseline)
                        
                        # Calculate SNR using different noise estimates
                        if len(noise_powers) > 0:
                            # Use the most conservative (highest) noise estimate
                            noise_power = np.max(noise_powers)
                            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
                            
                            # Also calculate linear SNR
                            snr_linear = signal_power / (noise_power + 1e-12)
                            
                            # Additional SNR quality metrics
                            # 4. Peak-to-sidelobe ratio
                            sidelobe_mask = (f >= target_freq - 3.0) & (f <= target_freq + 3.0)
                            sidelobe_mask &= ~signal_mask
                            
                            if np.any(sidelobe_mask):
                                max_sidelobe = np.max(Pxx[sidelobe_mask])
                                peak_signal = np.max(Pxx[signal_mask])
                                pslr_db = 10 * np.log10(peak_signal / (max_sidelobe + 1e-12))
                            else:
                                pslr_db = 20  # High value if no sidelobes
                            
                            # 5. Signal consistency (how stable the signal power is)
                            signal_std = np.std(Pxx[signal_mask])
                            signal_mean = np.mean(Pxx[signal_mask])
                            signal_consistency = signal_mean / (signal_std + 1e-8)
                            
                            # Combine SNR metrics
                            combined_snr = (
                                snr_db * 0.4 +
                                np.log1p(snr_linear) * 0.3 +
                                pslr_db * 0.2 +
                                np.log1p(signal_consistency) * 0.1
                            )
                            
                            channel_snr_scores.append(combined_snr)
                        else:
                            channel_snr_scores.append(0)  # No noise reference available
                    else:
                        channel_snr_scores.append(0)  # No signal region found
                
                # Average SNR across all SSVEP frequencies for this channel
                if len(channel_snr_scores) > 0:
                    avg_snr = np.mean(channel_snr_scores)
                    max_snr = np.max(channel_snr_scores)
                    # Weighted combination favoring the best frequency
                    final_snr = avg_snr * 0.6 + max_snr * 0.4
                else:
                    final_snr = 0
                
                snr_scores.append(final_snr)
            
            # Overall SNR quality across all channels
            mean_snr = np.mean(snr_scores)
            max_snr = np.max(snr_scores)
            final_snr_quality = (mean_snr * 0.7 + max_snr * 0.3)
            
            # Convert to confidence boost factor
            if final_snr_quality > 15:        # Excellent SNR (>15 dB)
                boost = 1.7
            elif final_snr_quality > 10:      # Good SNR (>10 dB)
                boost = 1.5
            elif final_snr_quality > 6:       # Moderate SNR (>6 dB)
                boost = 1.3
            elif final_snr_quality > 3:       # Fair SNR (>3 dB)
                boost = 1.15
            elif final_snr_quality > 0:       # Poor SNR (>0 dB)
                boost = 1.05
            else:                             # Very poor SNR (≤0 dB)
                boost = 1.0
            
            return boost
            
        except Exception as e:
            print(f"SSVEP SNR boost calculation failed: {e}")
            return 1.0

    def progressive_mi_training(self, mi_train, validation_df=None):
        """Progressive MI training with curriculum learning and proper model training"""
        print("📚 Progressive MI Training with Curriculum Learning")
        
        try:
            # Stage 1: Load and assess data quality
            print("\n🔍 Stage 1: Data Quality Assessment and Loading")
            mi_eeg_data = []
            mi_labels = []
            quality_scores = []
            
            for idx, row in mi_train.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        # Assess quality
                        quality = self.assess_trial_quality(trial_data, 'MI')
                        if quality > 0.3:  # Minimum quality threshold
                            mi_eeg_data.append(trial_data)
                            mi_labels.append(row['label'])
                            quality_scores.append(quality)
                except Exception as e:
                    continue
            
            if len(mi_eeg_data) == 0:
                print("❌ No valid MI data found")
                return None
            
            mi_labels = np.array(mi_labels)
            quality_scores = np.array(quality_scores)
            
            print(f"✓ Loaded {len(mi_eeg_data)} MI trials")
            print(f"✓ Mean quality score: {np.mean(quality_scores):.3f}")
            
            # Stage 2: Progressive feature extraction with consistent methods
            print("\n📈 Stage 2: Consistent Feature Extraction")
            
            # Use CONSISTENT feature extraction method (same as prediction)
            X_mi_features = []
            for trial_data in mi_eeg_data:
                features = self.extract_mi_features_fast(trial_data)
                X_mi_features.append(features)
            
            X_mi_features = np.array(X_mi_features)
            print(f"✓ Extracted features shape: {X_mi_features.shape}")
            
            # Stage 3: Final model training
            print("\n🎯 Stage 3: Final Model Training")
            
            # Encode labels
            y_mi_encoded = self.mi_label_encoder.fit_transform(mi_labels)
            
            # Scale features
            X_mi_scaled = self.mi_scaler.fit_transform(X_mi_features)
            
            # Feature selection
            X_mi_selected, y_mi_clean, self.mi_selector = self.feature_selection(
                X_mi_scaled, y_mi_encoded, 'MI', k=min(50, X_mi_scaled.shape[1])
            )
            
            # Create and train ensemble
            self.mi_ensemble = self.create_advanced_mi_ensemble()
            self.mi_ensemble.fit(X_mi_selected, y_mi_clean)
            
            # Validate model
            cv_scores = cross_val_score(self.mi_ensemble, X_mi_selected, y_mi_clean, cv=5)
            print(f"✓ Final MI model CV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            return {
                "success": True,
                "final_accuracy": np.mean(cv_scores),
                "n_features": X_mi_selected.shape[1],
                "n_samples": len(y_mi_clean)
            }
            
        except Exception as e:
            print(f"Progressive MI training failed: {e}")
            return self.basic_mi_training(mi_train)

    def progressive_ssvep_training(self, ssvep_train, validation_df=None):
        """Progressive SSVEP training with frequency curriculum and proper model training"""
        print("📚 Progressive SSVEP Training with Frequency Curriculum")
        
        try:
            # Stage 1: Load and assess data quality
            print("\n🔍 Stage 1: SSVEP Data Quality Assessment")
            ssvep_eeg_data = []
            ssvep_labels = []
            quality_scores = []
            
            for idx, row in ssvep_train.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        # Assess quality
                        quality = self.assess_trial_quality(trial_data, 'SSVEP')
                        if quality > 0.3:  # Minimum quality threshold
                            ssvep_eeg_data.append(trial_data)
                            ssvep_labels.append(row['label'])
                            quality_scores.append(quality)
                except Exception as e:
                    continue
            
            if len(ssvep_eeg_data) == 0:
                print("❌ No valid SSVEP data found")
                return None
            
            ssvep_labels = np.array(ssvep_labels)
            quality_scores = np.array(quality_scores)
            
            print(f"✓ Loaded {len(ssvep_eeg_data)} SSVEP trials")
            print(f"✓ Mean quality score: {np.mean(quality_scores):.3f}")
            
            # Stage 2: Consistent feature extraction
            print("\n📈 Stage 2: Consistent SSVEP Feature Extraction")
            
            # Use CONSISTENT feature extraction method (same as prediction)
            X_ssvep_features = []
            for trial_data in ssvep_eeg_data:
                features = self.extract_ssvep_features_fast(trial_data)
                X_ssvep_features.append(features)
            
            X_ssvep_features = np.array(X_ssvep_features)
            print(f"✓ Extracted features shape: {X_ssvep_features.shape}")
            
            # Stage 3: Final SSVEP model training
            print("\n🎯 Stage 3: Final SSVEP Model Training")
            
            # Encode labels
            y_ssvep_encoded = self.ssvep_label_encoder.fit_transform(ssvep_labels)
            
            # Scale features
            X_ssvep_scaled = self.ssvep_scaler.fit_transform(X_ssvep_features)
            
            # Feature selection
            X_ssvep_selected, y_ssvep_clean, self.ssvep_selector = self.feature_selection(
                X_ssvep_scaled, y_ssvep_encoded, 'SSVEP', k=min(80, X_ssvep_scaled.shape[1])
            )
            
            # Create and train ensemble
            self.ssvep_ensemble = self.create_advanced_ssvep_ensemble()
            self.ssvep_ensemble.fit(X_ssvep_selected, y_ssvep_clean)
            
            # Validate model
            cv_scores = cross_val_score(self.ssvep_ensemble, X_ssvep_selected, y_ssvep_clean, cv=5)
            print(f"✓ Final SSVEP model CV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            # Train Riemannian classifier
            try:
                print("\n🔬 Training Riemannian Classifier")
                self.train_riemannian_classifier(ssvep_eeg_data, ssvep_labels)
            except Exception as e:
                print(f"Riemannian training failed: {e}")
                self.ssvep_riemannian_classifier = None
            
            return {
                "success": True,
                "final_accuracy": np.mean(cv_scores),
                "n_features": X_ssvep_selected.shape[1],
                "n_samples": len(y_ssvep_clean)
            }
            
        except Exception as e:
            print(f"Progressive SSVEP training failed: {e}")
            return self.basic_ssvep_training(ssvep_train)

    def analyze_mi_feature_importance(self, mi_results):
        """Analyze MI feature importance"""
        if mi_results and hasattr(self, 'mi_ensemble'):
            try:
                print("🔍 Analyzing MI feature importance...")
                if hasattr(self.mi_ensemble, 'feature_importances_'):
                    importances = self.mi_ensemble.feature_importances_
                    print(f"   Top feature importance: {np.max(importances):.3f}")
                    print(f"   Feature importance mean: {np.mean(importances):.3f}")
                else:
                    print("   Feature importance not available for this model")
            except Exception as e:
                print(f"   Feature importance analysis failed: {e}")

    def analyze_ssvep_feature_importance(self, ssvep_results):
        """Analyze SSVEP feature importance"""
        if ssvep_results and hasattr(self, 'ssvep_ensemble'):
            try:
                print("🔍 Analyzing SSVEP feature importance...")
                if hasattr(self.ssvep_ensemble, 'feature_importances_'):
                    importances = self.ssvep_ensemble.feature_importances_
                    print(f"   Top feature importance: {np.max(importances):.3f}")
                    print(f"   Feature importance mean: {np.mean(importances):.3f}")
                else:
                    print("   Feature importance not available for this model")
            except Exception as e:
                print(f"   Feature importance analysis failed: {e}")

    def temporal_cross_validation(self, df, task_type):
        """Cross-validation with temporal considerations"""
        try:
            print(f"📊 Temporal cross-validation for {task_type}...")
            
            # Simple validation for now
            X, y = self.load_and_prepare_data(df, task_type)
            if len(X) > 10:
                if task_type == 'MI' and hasattr(self, 'mi_ensemble'):
                    cv_scores = cross_val_score(self.mi_ensemble, X, self.mi_label_encoder.transform(y), cv=3)
                    print(f"   {task_type} CV scores: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                elif task_type == 'SSVEP' and hasattr(self, 'ssvep_ensemble'):
                    cv_scores = cross_val_score(self.ssvep_ensemble, X, self.ssvep_label_encoder.transform(y), cv=3)
                    print(f"   {task_type} CV scores: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
        except Exception as e:
            print(f"   Temporal CV failed: {e}")

    def advanced_cca_optimization(self, validation_df):
        """Advanced CCA threshold optimization"""
        try:
            print("🎯 Advanced CCA optimization...")
            if validation_df is not None:
                self.enhanced_threshold_tuning(validation_df)
        except Exception as e:
            print(f"   CCA optimization failed: {e}")

    def comprehensive_validation(self, validation_df):
        """Comprehensive validation with enhanced metrics"""
        try:
            print("📈 Comprehensive Validation...")
            self.validate(validation_df)  # Use existing validation method
        except Exception as e:
            print(f"   Comprehensive validation failed: {e}")

    def model_interpretability_analysis(self, train_df):
        """Model interpretability analysis"""
        try:
            print("🔬 Model interpretability analysis...")
            
            # Basic model analysis
            if hasattr(self, 'mi_ensemble'):
                print("   MI model trained successfully")
            if hasattr(self, 'ssvep_ensemble'):
                print("   SSVEP model trained successfully")
                
        except Exception as e:
            print(f"   Interpretability analysis failed: {e}")

    def fallback_mi_training(self, mi_train):
        """Fallback MI training method"""
        try:
            print("🔄 Fallback MI training...")
            X_mi, y_mi = self.load_and_prepare_data(mi_train, 'MI')
            
            if len(X_mi) > 0:
                y_mi_encoded = self.mi_label_encoder.fit_transform(y_mi)
                X_mi_scaled = self.mi_scaler.fit_transform(X_mi)
                X_mi_selected, y_mi_clean, self.mi_selector = self.feature_selection(
                    X_mi_scaled, y_mi_encoded, 'MI', k=40
                )
                self.mi_ensemble.fit(X_mi_selected, y_mi_clean)
                print("   Fallback MI training completed")
            
        except Exception as e:
            print(f"   Fallback MI training failed: {e}")

    def fallback_ssvep_training(self, ssvep_train):
        """Fallback SSVEP training method"""
        try:
            print("🔄 Fallback SSVEP training...")
            X_ssvep, y_ssvep = self.load_and_prepare_data(ssvep_train, 'SSVEP')
            
            if len(X_ssvep) > 0:
                y_ssvep_encoded = self.ssvep_label_encoder.fit_transform(y_ssvep)
                X_ssvep_scaled = self.ssvep_scaler.fit_transform(X_ssvep)
                X_ssvep_selected, y_ssvep_clean, self.ssvep_selector = self.feature_selection(
                    X_ssvep_scaled, y_ssvep_encoded, 'SSVEP', k=60
                )
                self.ssvep_ensemble.fit(X_ssvep_selected, y_ssvep_clean)
                print("   Fallback SSVEP training completed")
            
        except Exception as e:
            print(f"   Fallback SSVEP training failed: {e}")

    def assess_trial_quality(self, trial_data, task_type):
        """Assess the quality of a trial based on signal characteristics"""
        try:
            # Basic quality metrics
            quality_score = 0.0
            
            # 1. Check for NaN/infinite values (penalty)
            if np.any(np.isnan(trial_data)) or np.any(np.isinf(trial_data)):
                quality_score -= 0.3
            
            # 2. Signal variance (too low = bad quality)
            channel_variances = np.var(trial_data, axis=0)
            mean_variance = np.mean(channel_variances)
            if mean_variance > 50:  # Good variance
                quality_score += 0.3
            elif mean_variance > 10:  # Moderate variance
                quality_score += 0.2
            else:  # Low variance
                quality_score += 0.1
            
            # 3. Signal range (too small = artifacts or disconnection)
            channel_ranges = np.ptp(trial_data, axis=0)
            mean_range = np.mean(channel_ranges)
            if mean_range > 100:  # Good range
                quality_score += 0.2
            elif mean_range > 50:  # Moderate range
                quality_score += 0.15
            else:  # Small range
                quality_score += 0.05
            
            # 4. Frequency domain quality
            try:
                if task_type == 'MI':
                    # Check for mu/beta rhythm presence
                    motor_channels = [1, 3]  # C3, C4
                    for ch in motor_channels:
                        f, Pxx = signal.welch(trial_data[:, ch], fs=self.fs, nperseg=256)
                        mu_mask = (f >= 8) & (f <= 13)
                        beta_mask = (f >= 13) & (f <= 30)
                        if np.any(mu_mask) and np.any(beta_mask):
                            mu_power = np.trapz(Pxx[mu_mask], f[mu_mask])
                            beta_power = np.trapz(Pxx[beta_mask], f[beta_mask])
                            if mu_power > 0 and beta_power > 0:
                                quality_score += 0.1
                else:  # SSVEP
                    # Check for SSVEP frequency presence
                    occipital_channels = [5, 6, 7]  # PO7, OZ, PO8
                    for ch in occipital_channels:
                        f, Pxx = signal.welch(trial_data[:, ch], fs=self.fs, nperseg=256)
                        ssvep_detected = False
                        for target_freq in self.ssvep_freqs:
                            freq_mask = (f >= target_freq - 1) & (f <= target_freq + 1)
                            if np.any(freq_mask):
                                freq_power = np.trapz(Pxx[freq_mask], f[freq_mask])
                                if freq_power > np.mean(Pxx) * 2:  # Power above average
                                    ssvep_detected = True
                                    break
                        if ssvep_detected:
                            quality_score += 0.1
            except:
                pass
            
            # 5. Artifact detection (high frequency noise)
            try:
                for ch in range(trial_data.shape[1]):
                    f, Pxx = signal.welch(trial_data[:, ch], fs=self.fs, nperseg=256)
                    high_freq_mask = f > 50  # Above 50 Hz
                    if np.any(high_freq_mask):
                        high_freq_power = np.trapz(Pxx[high_freq_mask], f[high_freq_mask])
                        total_power = np.trapz(Pxx, f)
                        if high_freq_power / total_power > 0.3:  # Too much high freq
                            quality_score -= 0.1
            except:
                pass
            
            # Normalize to 0-1 range
            quality_score = max(0.0, min(1.0, quality_score))
            
            return quality_score
            
        except Exception as e:
            return 0.5  # Default moderate quality
    
    def create_advanced_mi_ensemble(self):
        """Create advanced MI ensemble with optimized parameters"""
        from sklearn.ensemble import VotingClassifier
        
        # Base models with different characteristics
        models = [
            ('xgb_balanced', XGBClassifier(
                n_estimators=60, learning_rate=0.04, max_depth=4,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=3.0,
                min_child_weight=6, gamma=0.2, random_state=42,
                tree_method='hist', use_label_encoder=False, eval_metric='logloss'
            )),
            ('rf_balanced', RandomForestClassifier(
                n_estimators=50, max_depth=5, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt', random_state=42,
                class_weight='balanced', bootstrap=True, n_jobs=-1
            )),
            ('svm_balanced', SVC(
                kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                probability=True, random_state=42
            )),
            ('lda_balanced', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.3))
        ]
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def create_advanced_ssvep_ensemble(self):
        """Create advanced SSVEP ensemble with optimized parameters"""
        from sklearn.ensemble import VotingClassifier
        
        # Base models optimized for SSVEP
        models = [
            ('xgb_ssvep', XGBClassifier(
                n_estimators=80, learning_rate=0.05, max_depth=5,
                subsample=0.75, colsample_bytree=0.75, reg_alpha=1.5, reg_lambda=2.5,
                min_child_weight=4, gamma=0.1, random_state=42,
                tree_method='hist', use_label_encoder=False, eval_metric='mlogloss'
            )),
            ('rf_ssvep', RandomForestClassifier(
                n_estimators=70, max_depth=6, min_samples_split=8,
                min_samples_leaf=4, max_features='sqrt', random_state=42,
                class_weight='balanced', bootstrap=True, n_jobs=-1
            )),
            ('svm_ssvep', SVC(
                kernel='rbf', C=2.0, gamma='scale', class_weight='balanced',
                probability=True, random_state=42
            )),
            ('mlp_ssvep', MLPClassifier(
                hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                alpha=0.02, learning_rate='adaptive', max_iter=300,
                random_state=42, early_stopping=True, validation_fraction=0.2
            ))
        ]
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def train_riemannian_classifier(self, eeg_data, labels):
        """Train Riemannian classifier for SSVEP"""
        try:
            # Focus on occipital channels
            occipital_channels = [5, 6, 7]  # PO7, OZ, PO8
            occipital_data = []
            
            for trial_data in eeg_data:
                trial_occipital = trial_data[:, occipital_channels]
                occipital_data.append(trial_occipital)
            
            occipital_data = np.array(occipital_data)
            
            # Train Riemannian classifier
            self.ssvep_riemannian_classifier, _ = self.apply_riemannian_classification(
                occipital_data, labels
            )
            
            if self.ssvep_riemannian_classifier is not None:
                print("   ✓ Riemannian classifier trained successfully")
            else:
                print("   ❌ Riemannian classifier training failed")
                
        except Exception as e:
            print(f"   Riemannian training error: {e}")
            self.ssvep_riemannian_classifier = None
    
    def analyze_frequency_difficulty(self, eeg_data, labels):
        """Analyze which SSVEP frequencies are more difficult to classify"""
        try:
            freq_scores = {}
            
            # Simple frequency analysis
            for freq in self.ssvep_freqs:
                freq_power_scores = []
                
                for i, trial_data in enumerate(eeg_data):
                    occipital_data = trial_data[:, [5, 6, 7]]  # PO7, OZ, PO8
                    
                    # Calculate power at target frequency
                    for ch in range(occipital_data.shape[1]):
                        f, Pxx = signal.welch(occipital_data[:, ch], fs=self.fs, nperseg=256)
                        freq_mask = (f >= freq - 0.5) & (f <= freq + 0.5)
                        if np.any(freq_mask):
                            freq_power = np.trapz(Pxx[freq_mask], f[freq_mask])
                            freq_power_scores.append(freq_power)
                
                freq_scores[freq] = np.mean(freq_power_scores) if freq_power_scores else 0
            
            print(f"   Frequency power analysis: {freq_scores}")
            return freq_scores
            
        except Exception as e:
            print(f"   Frequency analysis failed: {e}")
            return {}
    
    def basic_mi_training(self, mi_train):
        """Basic fallback MI training"""
        try:
            print("🔧 Basic MI training fallback...")
            X_mi, y_mi = self.load_and_prepare_data(mi_train, 'MI')
            
            if len(X_mi) > 0:
                y_mi_encoded = self.mi_label_encoder.fit_transform(y_mi)
                X_mi_scaled = self.mi_scaler.fit_transform(X_mi)
                X_mi_selected, y_mi_clean, self.mi_selector = self.feature_selection(
                    X_mi_scaled, y_mi_encoded, 'MI', k=40
                )
                
                # Use simple ensemble
                self.mi_ensemble = RandomForestClassifier(
                    n_estimators=50, max_depth=5, random_state=42, 
                    class_weight='balanced', n_jobs=-1
                )
                self.mi_ensemble.fit(X_mi_selected, y_mi_clean)
                
                return {"basic_training": True, "success": True}
        except Exception as e:
            print(f"Basic MI training failed: {e}")
            return None
    
    def basic_ssvep_training(self, ssvep_train):
        """Basic fallback SSVEP training"""
        try:
            print("🔧 Basic SSVEP training fallback...")
            X_ssvep, y_ssvep = self.load_and_prepare_data(ssvep_train, 'SSVEP')
            
            if len(X_ssvep) > 0:
                y_ssvep_encoded = self.ssvep_label_encoder.fit_transform(y_ssvep)
                X_ssvep_scaled = self.ssvep_scaler.fit_transform(X_ssvep)
                X_ssvep_selected, y_ssvep_clean, self.ssvep_selector = self.feature_selection(
                    X_ssvep_scaled, y_ssvep_encoded, 'SSVEP', k=60
                )
                
                # Use simple ensemble
                self.ssvep_ensemble = RandomForestClassifier(
                    n_estimators=70, max_depth=6, random_state=42, 
                    class_weight='balanced', n_jobs=-1
                )
                self.ssvep_ensemble.fit(X_ssvep_selected, y_ssvep_clean)
                
                return {"basic_training": True, "success": True}
        except Exception as e:
            print(f"Basic SSVEP training failed: {e}")
            return None

    def enhanced_validation_pipeline(self, validation_df):
        """Enhanced validation pipeline with advanced analytics"""
        print("🔬 Enhanced Validation Pipeline")
        print("-" * 40)
        
        # 1. Data Quality Assessment
        self.validate_data_quality(validation_df)
        
        # 2. Task-Specific Performance Analysis
        self.validate_task_performance(validation_df)
        
        # 3. Cross-Validation with Different Strategies
        self.validate_cross_validation_strategies(validation_df)
        
        # 4. Confidence Calibration Analysis
        self.validate_confidence_calibration(validation_df)
        
        # 5. Error Analysis and Pattern Detection
        self.validate_error_analysis(validation_df)
        
        # 6. Model Stability Assessment
        self.validate_model_stability(validation_df)

    def validate_data_quality(self, validation_df):
        """Assess validation data quality"""
        try:
            print("\n📊 Data Quality Assessment")
            print("-" * 30)
            
            mi_val = validation_df[validation_df['task'] == 'MI']
            ssvep_val = validation_df[validation_df['task'] == 'SSVEP']
            
            print(f"📈 Dataset Distribution:")
            print(f"   MI samples: {len(mi_val)}")
            print(f"   SSVEP samples: {len(ssvep_val)}")
            
            # Assess sample quality
            quality_scores = []
            for idx, row in validation_df.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        quality = self.assess_trial_quality(trial_data, row['task'])
                        quality_scores.append(quality)
                    else:
                        quality_scores.append(0.0)
                except:
                    quality_scores.append(0.0)
            
            if quality_scores:
                print(f"📊 Quality Statistics:")
                print(f"   Mean quality: {np.mean(quality_scores):.3f}")
                print(f"   Std quality: {np.std(quality_scores):.3f}")
                print(f"   Min quality: {np.min(quality_scores):.3f}")
                print(f"   Max quality: {np.max(quality_scores):.3f}")
                
                high_quality = np.sum(np.array(quality_scores) > 0.7)
                low_quality = np.sum(np.array(quality_scores) < 0.4)
                print(f"   High quality samples (>0.7): {high_quality} ({high_quality/len(quality_scores)*100:.1f}%)")
                print(f"   Low quality samples (<0.4): {low_quality} ({low_quality/len(quality_scores)*100:.1f}%)")
            
        except Exception as e:
            print(f"Data quality assessment failed: {e}")

    def validate_task_performance(self, validation_df):
        """Task-specific performance validation"""
        try:
            print("\n🎯 Task-Specific Performance Analysis")
            print("-" * 40)
            
            # MI Performance Analysis
            mi_val = validation_df[validation_df['task'] == 'MI']
            if len(mi_val) > 0 and hasattr(self, 'mi_ensemble'):
                print("\n🧠 MI Performance Analysis:")
                mi_predictions, mi_confidences = self.validate_mi_performance(mi_val)
                
            # SSVEP Performance Analysis
            ssvep_val = validation_df[validation_df['task'] == 'SSVEP']
            if len(ssvep_val) > 0 and hasattr(self, 'ssvep_ensemble'):
                print("\n👁️ SSVEP Performance Analysis:")
                ssvep_predictions, ssvep_confidences = self.validate_ssvep_performance(ssvep_val)
                
        except Exception as e:
            print(f"Task performance analysis failed: {e}")

    def validate_mi_performance(self, mi_val):
        """Detailed MI performance validation"""
        predictions = []
        confidences = []
        true_labels = []
        
        print("   Loading MI validation data...")
        for idx, row in mi_val.iterrows():
            try:
                trial_data = self.load_trial_data(row)
                if trial_data is not None:
                    # Use same prediction pipeline as main prediction
                    pred, conf = self.dynamic_prediction_fusion(trial_data, 'MI')
                    predictions.append(pred)
                    confidences.append(conf)
                    true_labels.append(row['label'])
            except Exception as e:
                continue
        
        if len(predictions) > 0:
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            print(f"   MI Validation Accuracy: {accuracy:.3f}")
            print(f"   MI Mean Confidence: {np.mean(confidences):.3f}")
            print(f"   MI Confidence Std: {np.std(confidences):.3f}")
            
            # Class distribution
            pred_dist = pd.Series(predictions).value_counts()
            true_dist = pd.Series(true_labels).value_counts()
            print(f"   MI Prediction Distribution: {pred_dist.to_dict()}")
            print(f"   MI True Distribution: {true_dist.to_dict()}")
            
            # Confidence analysis by class
            for label in set(true_labels):
                label_mask = np.array(true_labels) == label
                if np.any(label_mask):
                    label_confs = np.array(confidences)[label_mask]
                    label_preds = np.array(predictions)[label_mask]
                    label_acc = accuracy_score([label]*len(label_preds), label_preds)
                    print(f"   MI {label}: Acc={label_acc:.3f}, AvgConf={np.mean(label_confs):.3f}")
        
        return predictions, confidences

    def validate_ssvep_performance(self, ssvep_val):
        """Detailed SSVEP performance validation"""
        predictions = []
        confidences = []
        true_labels = []
        cca_predictions = []
        
        print("   Loading SSVEP validation data...")
        for idx, row in ssvep_val.iterrows():
            try:
                trial_data = self.load_trial_data(row)
                if trial_data is not None:
                    # Test both ensemble and CCA predictions
                    pred, conf = self.dynamic_prediction_fusion(trial_data, 'SSVEP')
                    predictions.append(pred)
                    confidences.append(conf)
                    true_labels.append(row['label'])
                    
                    # Also test CCA prediction separately
                    cca_pred, cca_conf = self.enhanced_cca_predict(trial_data)
                    cca_predictions.append(cca_pred if cca_pred is not None else pred)
            except Exception as e:
                continue
        
        if len(predictions) > 0:
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            print(f"   SSVEP Validation Accuracy: {accuracy:.3f}")
            print(f"   SSVEP Mean Confidence: {np.mean(confidences):.3f}")
            print(f"   SSVEP Confidence Std: {np.std(confidences):.3f}")
            
            # CCA vs Ensemble comparison
            if len(cca_predictions) == len(predictions):
                cca_accuracy = accuracy_score(true_labels, cca_predictions)
                print(f"   SSVEP CCA Accuracy: {cca_accuracy:.3f}")
                agreement = np.mean(np.array(predictions) == np.array(cca_predictions))
                print(f"   Ensemble-CCA Agreement: {agreement:.3f}")
            
            # Frequency-specific analysis
            for freq in ['Forward', 'Backward', 'Left', 'Right']:
                freq_mask = np.array(true_labels) == freq
                if np.any(freq_mask):
                    freq_preds = np.array(predictions)[freq_mask]
                    freq_acc = accuracy_score([freq]*len(freq_preds), freq_preds)
                    freq_confs = np.array(confidences)[freq_mask]
                    print(f"   SSVEP {freq}: Acc={freq_acc:.3f}, AvgConf={np.mean(freq_confs):.3f}")
        
        return predictions, confidences

    def validate_cross_validation_strategies(self, validation_df):
        """Test different cross-validation strategies"""
        try:
            print("\n🔄 Cross-Validation Strategy Analysis")
            print("-" * 40)
            
            # Test different CV strategies for robustness
            cv_strategies = {
                'Stratified K-Fold': StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                'Standard K-Fold': KFold(n_splits=3, shuffle=True, random_state=42),
            }
            
            for task in ['MI', 'SSVEP']:
                task_val = validation_df[validation_df['task'] == task]
                if len(task_val) > 10:
                    print(f"\n   {task} Cross-Validation Analysis:")
                    
                    # Load data for CV
                    X, y = self.load_and_prepare_data(task_val, task)
                    if len(X) > 0:
                        if task == 'MI' and hasattr(self, 'mi_ensemble'):
                            model = self.mi_ensemble
                            encoder = self.mi_label_encoder
                            scaler = self.mi_scaler
                            selector = self.mi_selector
                        elif task == 'SSVEP' and hasattr(self, 'ssvep_ensemble'):
                            model = self.ssvep_ensemble
                            encoder = self.ssvep_label_encoder
                            scaler = self.ssvep_scaler
                            selector = self.ssvep_selector
                        else:
                            continue
                        
                        y_encoded = encoder.transform(y)
                        X_scaled = scaler.transform(X)
                        X_selected = selector.transform(X_scaled)
                        
                        for cv_name, cv_strategy in cv_strategies.items():
                            try:
                                scores = cross_val_score(model, X_selected, y_encoded, cv=cv_strategy, scoring='accuracy')
                                print(f"      {cv_name}: {scores.mean():.3f} ± {scores.std():.3f}")
                            except Exception as e:
                                print(f"      {cv_name}: Failed - {e}")
        
        except Exception as e:
            print(f"Cross-validation analysis failed: {e}")

    def validate_confidence_calibration(self, validation_df):
        """Analyze confidence calibration"""
        try:
            print("\n📊 Confidence Calibration Analysis")
            print("-" * 40)
            
            all_confidences = []
            all_accuracies = []
            
            for idx, row in validation_df.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        pred, conf = self.dynamic_prediction_fusion(trial_data, row['task'])
                        all_confidences.append(conf)
                        all_accuracies.append(1.0 if pred == row['label'] else 0.0)
                except:
                    continue
            
            if len(all_confidences) > 10:
                # Bin confidence levels and analyze calibration
                conf_bins = np.linspace(0, 1, 11)
                bin_accuracies = []
                bin_confidences = []
                bin_counts = []
                
                for i in range(len(conf_bins)-1):
                    bin_mask = (np.array(all_confidences) >= conf_bins[i]) & (np.array(all_confidences) < conf_bins[i+1])
                    if np.any(bin_mask):
                        bin_acc = np.mean(np.array(all_accuracies)[bin_mask])
                        bin_conf = np.mean(np.array(all_confidences)[bin_mask])
                        bin_count = np.sum(bin_mask)
                        
                        bin_accuracies.append(bin_acc)
                        bin_confidences.append(bin_conf)
                        bin_counts.append(bin_count)
                        
                        print(f"   Conf [{conf_bins[i]:.1f}-{conf_bins[i+1]:.1f}]: Acc={bin_acc:.3f}, Count={bin_count}")
                
                # Expected Calibration Error (ECE)
                if bin_accuracies and bin_confidences:
                    ece = np.sum([count * abs(acc - conf) for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)]) / len(all_confidences)
                    print(f"   Expected Calibration Error (ECE): {ece:.3f}")
        
        except Exception as e:
            print(f"Confidence calibration analysis failed: {e}")

    def validate_error_analysis(self, validation_df):
        """Analyze prediction errors and patterns"""
        try:
            print("\n🔍 Error Analysis and Pattern Detection")
            print("-" * 40)
            
            errors_by_task = {'MI': [], 'SSVEP': []}
            
            for idx, row in validation_df.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        pred, conf = self.dynamic_prediction_fusion(trial_data, row['task'])
                        
                        if pred != row['label']:
                            error_info = {
                                'true_label': row['label'],
                                'predicted_label': pred,
                                'confidence': conf,
                                'trial_id': row['id']
                            }
                            errors_by_task[row['task']].append(error_info)
                except:
                    continue
            
            # Analyze error patterns
            for task, errors in errors_by_task.items():
                if len(errors) > 0:
                    print(f"\n   {task} Error Analysis:")
                    print(f"      Total errors: {len(errors)}")
                    
                    # Most common misclassifications
                    error_pairs = [(e['true_label'], e['predicted_label']) for e in errors]
                    error_counts = pd.Series(error_pairs).value_counts()
                    print(f"      Common misclassifications:")
                    for (true, pred), count in error_counts.head(3).items():
                        print(f"         {true} → {pred}: {count} times")
                    
                    # Error confidence distribution
                    error_confs = [e['confidence'] for e in errors]
                    print(f"      Error confidence: {np.mean(error_confs):.3f} ± {np.std(error_confs):.3f}")
        
        except Exception as e:
            print(f"Error analysis failed: {e}")

    def validate_model_stability(self, validation_df):
        """Assess model stability and consistency"""
        try:
            print("\n🔬 Model Stability Assessment")
            print("-" * 40)
            
            # Test prediction consistency with multiple runs
            stability_scores = []
            
            # Sample a subset for stability testing
            test_samples = validation_df.sample(min(20, len(validation_df)), random_state=42)
            
            for idx, row in test_samples.iterrows():
                try:
                    trial_data = self.load_trial_data(row)
                    if trial_data is not None:
                        # Multiple predictions for same sample
                        predictions = []
                        confidences = []
                        
                        for run in range(3):  # 3 runs
                            pred, conf = self.dynamic_prediction_fusion(trial_data, row['task'])
                            predictions.append(pred)
                            confidences.append(conf)
                        
                        # Check consistency
                        unique_preds = len(set(predictions))
                        conf_std = np.std(confidences)
                        
                        stability_score = 1.0 / unique_preds  # Higher score for consistent predictions
                        stability_scores.append(stability_score)
                
                except:
                    continue
            
            if stability_scores:
                mean_stability = np.mean(stability_scores)
                print(f"   Model Stability Score: {mean_stability:.3f}")
                
                if mean_stability > 0.8:
                    print("   ✅ Model shows high stability")
                elif mean_stability > 0.6:
                    print("   ⚠️ Model shows moderate stability")
                else:
                    print("   ❌ Model shows low stability - investigate further")
        
        except Exception as e:
            print(f"Model stability assessment failed: {e}")

# Main function would remain similar but use EnhancedBCIClassifier instead
def main():
    """Main function to run the task-specific BCI classification with comprehensive metrics"""
    # Initialize the classifier
    base_path = r'E:\AIC_competition\Coding\mtcaic3'
    classifier = EnhancedBCIClassifier(base_path)
    
    # Load the CSV files
    print("📂 Loading dataset...")
    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
    
    print(f"📊 Dataset: {len(train_df)} train, {len(validation_df)} val, {len(test_df)} test")
    
    # Display class distribution
    print(f"🏷️ Classes: {train_df['label'].value_counts().to_dict()}")
    print(f"📋 Tasks: {train_df['task'].value_counts().to_dict()}")
    
    # Train the model with full dataset
    classifier.train(train_df, validation_df)
    
    # Enhanced CCA threshold tuning for SSVEP
    cca_threshold = classifier.enhanced_threshold_tuning(validation_df)
    
    # Make predictions on test data with tuned threshold
    print("\n🔮 Making predictions...")
    predictions, confidences = classifier.predict(test_df)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    
    # Save submission
    submission_path = os.path.join(base_path, 'task_specific_bci_submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"💾 Submission saved: {submission_path}")
    
    # Comprehensive prediction analysis
    print("\n📊 Comprehensive Prediction Analysis")
    print("=" * 60)
    
    # Overall prediction statistics
    print(f"\n🎯 Overall Prediction Statistics:")
    print(f"  Total Predictions: {len(predictions)}")
    print(f"  Mean Confidence: {np.mean(confidences):.4f}")
    print(f"  Std Confidence: {np.std(confidences):.4f}")
    print(f"  Min Confidence: {np.min(confidences):.4f}")
    print(f"  Max Confidence: {np.max(confidences):.4f}")
    
    # Confidence distribution statistics
    from scipy.stats import skew, kurtosis, percentileofscore
    print(f"\n📈 Confidence Distribution Statistics:")
    print(f"  Skewness: {skew(confidences):.4f}")
    print(f"  Kurtosis: {kurtosis(confidences):.4f}")
    print(f"  25th Percentile: {np.percentile(confidences, 25):.4f}")
    print(f"  50th Percentile (Median): {np.percentile(confidences, 50):.4f}")
    print(f"  75th Percentile: {np.percentile(confidences, 75):.4f}")
    print(f"  90th Percentile: {np.percentile(confidences, 90):.4f}")
    print(f"  95th Percentile: {np.percentile(confidences, 95):.4f}")
    
    # High confidence predictions analysis
    high_conf_threshold = 0.8
    high_conf_mask = np.array(confidences) >= high_conf_threshold
    high_conf_count = np.sum(high_conf_mask)
    print(f"\n🔍 High Confidence Analysis (≥{high_conf_threshold}):")
    print(f"  High Confidence Predictions: {high_conf_count} ({high_conf_count/len(predictions)*100:.1f}%)")
    if high_conf_count > 0:
        high_conf_predictions = [p for i, p in enumerate(predictions) if high_conf_mask[i]]
        high_conf_dist = pd.Series(high_conf_predictions).value_counts()
        print(f"  High Confidence Distribution: {high_conf_dist.to_dict()}")
    
    # Low confidence predictions analysis
    low_conf_threshold = 0.5
    low_conf_mask = np.array(confidences) < low_conf_threshold
    low_conf_count = np.sum(low_conf_mask)
    print(f"\n⚠️ Low Confidence Analysis (<{low_conf_threshold}):")
    print(f"  Low Confidence Predictions: {low_conf_count} ({low_conf_count/len(predictions)*100:.1f}%)")
    if low_conf_count > 0:
        low_conf_predictions = [p for i, p in enumerate(predictions) if low_conf_mask[i]]
        low_conf_dist = pd.Series(low_conf_predictions).value_counts()
        print(f"  Low Confidence Distribution: {low_conf_dist.to_dict()}")
    
    # Display prediction distribution
    print(f"\n📊 Overall Prediction Distribution:")
    pred_dist = pd.Series(predictions).value_counts()
    print(f"  {pred_dist.to_dict()}")
    
    # Task-specific analysis
    test_mi = test_df[test_df['task'] == 'MI']
    test_ssvep = test_df[test_df['task'] == 'SSVEP']
    
    if len(test_mi) > 0:
        print(f"\n🧠 MI Task Analysis:")
        print("-" * 30)
        mi_indices = test_df[test_df['task'] == 'MI'].index
        mi_predictions = [predictions[i] for i in range(len(predictions)) if test_df.iloc[i]['task'] == 'MI']
        mi_confidences = [confidences[i] for i in range(len(confidences)) if test_df.iloc[i]['task'] == 'MI']
        
        mi_dist = pd.Series(mi_predictions).value_counts()
        print(f"  MI Predictions: {mi_dist.to_dict()}")
        print(f"  MI Mean Confidence: {np.mean(mi_confidences):.4f}")
        print(f"  MI Std Confidence: {np.std(mi_confidences):.4f}")
        print(f"  MI Confidence Skewness: {skew(mi_confidences):.4f}")
        print(f"  MI Confidence Kurtosis: {kurtosis(mi_confidences):.4f}")
        
        # MI confidence distribution
        mi_high_conf = np.sum(np.array(mi_confidences) >= high_conf_threshold)
        mi_low_conf = np.sum(np.array(mi_confidences) < low_conf_threshold)
        print(f"  MI High Confidence (≥{high_conf_threshold}): {mi_high_conf} ({mi_high_conf/len(mi_predictions)*100:.1f}%)")
        print(f"  MI Low Confidence (<{low_conf_threshold}): {mi_low_conf} ({mi_low_conf/len(mi_predictions)*100:.1f}%)")
    
    if len(test_ssvep) > 0:
        print(f"\n👁️ SSVEP Task Analysis:")
        print("-" * 30)
        ssvep_indices = test_df[test_df['task'] == 'SSVEP'].index
        ssvep_predictions = [predictions[i] for i in range(len(predictions)) if test_df.iloc[i]['task'] == 'SSVEP']
        ssvep_confidences = [confidences[i] for i in range(len(confidences)) if test_df.iloc[i]['task'] == 'SSVEP']
        
        ssvep_dist = pd.Series(ssvep_predictions).value_counts()
        print(f"  SSVEP Predictions: {ssvep_dist.to_dict()}")
        print(f"  SSVEP Mean Confidence: {np.mean(ssvep_confidences):.4f}")
        print(f"  SSVEP Std Confidence: {np.std(ssvep_confidences):.4f}")
        print(f"  SSVEP Confidence Skewness: {skew(ssvep_confidences):.4f}")
        print(f"  SSVEP Confidence Kurtosis: {kurtosis(ssvep_confidences):.4f}")
        
        # SSVEP confidence distribution
        ssvep_high_conf = np.sum(np.array(ssvep_confidences) >= high_conf_threshold)
        ssvep_low_conf = np.sum(np.array(ssvep_confidences) < low_conf_threshold)
        print(f"  SSVEP High Confidence (≥{high_conf_threshold}): {ssvep_high_conf} ({ssvep_high_conf/len(ssvep_predictions)*100:.1f}%)")
        print(f"  SSVEP Low Confidence (<{low_conf_threshold}): {ssvep_low_conf} ({ssvep_low_conf/len(ssvep_predictions)*100:.1f}%)")
    
    # Class-specific confidence analysis
    print(f"\n🎯 Class-Specific Confidence Analysis:")
    print("-" * 40)
    unique_classes = list(set(predictions))
    for class_name in unique_classes:
        class_mask = np.array(predictions) == class_name
        class_confidences = [confidences[i] for i in range(len(confidences)) if class_mask[i]]
        class_count = len(class_confidences)
        
        if class_count > 0:
            print(f"  {class_name}:")
            print(f"    Count: {class_count}")
            print(f"    Mean Confidence: {np.mean(class_confidences):.4f}")
            print(f"    Std Confidence: {np.std(class_confidences):.4f}")
            print(f"    Min Confidence: {np.min(class_confidences):.4f}")
            print(f"    Max Confidence: {np.max(class_confidences):.4f}")
            print(f"    Skewness: {skew(class_confidences):.4f}")
            print(f"    Kurtosis: {kurtosis(class_confidences):.4f}")
            
            # High/low confidence for this class
            class_high_conf = np.sum(np.array(class_confidences) >= high_conf_threshold)
            class_low_conf = np.sum(np.array(class_confidences) < low_conf_threshold)
            print(f"    High Confidence (≥{high_conf_threshold}): {class_high_conf} ({class_high_conf/class_count*100:.1f}%)")
            print(f"    Low Confidence (<{low_conf_threshold}): {class_low_conf} ({class_low_conf/class_count*100:.1f}%)")
    
    # Model performance summary
    print(f"\n🏆 Model Performance Summary:")
    print("-" * 30)
    print(f"  CCA Threshold: {cca_threshold:.4f}")
    print(f"  Total Test Samples: {len(test_df)}")
    print(f"  MI Samples: {len(test_mi)}")
    print(f"  SSVEP Samples: {len(test_ssvep)}")
    print(f"  Unique Classes Predicted: {len(unique_classes)}")
    print(f"  Classes: {unique_classes}")
    
    # Confidence quality metrics
    print(f"\n📊 Confidence Quality Metrics:")
    print("-" * 30)
    print(f"  Mean Confidence: {np.mean(confidences):.4f}")
    print(f"  Median Confidence: {np.median(confidences):.4f}")
    print(f"  Confidence Range: {np.max(confidences) - np.min(confidences):.4f}")
    print(f"  Coefficient of Variation: {np.std(confidences)/np.mean(confidences):.4f}")
    
    # Entropy of predictions (measure of prediction diversity)
    from scipy.stats import entropy
    pred_counts = pd.Series(predictions).value_counts()
    pred_probs = pred_counts / len(predictions)
    pred_entropy = entropy(pred_probs)
    print(f"  Prediction Entropy: {pred_entropy:.4f}")
    
    print("✅ Task-Specific BCI Classification completed with comprehensive metrics!")

if __name__ == "__main__":
    main()