
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
            # early_stopping_rounds=10, # Removed for StackingClassifier
            # validation_fraction=0.2 # Removed for StackingClassifier
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
            # early_stopping_rounds=10, # Removed for StackingClassifier
            # validation_fraction=0.2 # Removed for StackingClassifier
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
            passthrough=False  # Don\'t pass original features
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
            # early_stopping_rounds=10, # Removed for StackingClassifier
            # validation_fraction=0.2 # Removed for StackingClassifier
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
            # early_stopping_rounds=10, # Removed for StackingClassifier
            # validation_fraction=0.2 # Removed for StackingClassifier
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
        # print("Applying dummy ICA artifact removal.") # Removed print to reduce output
        return eeg_data # Return data as is for simplicity in dummy training

    def extract_mi_features(self, eeg_data):
        """Simplified MI feature extraction for dummy data."""
        # Focus on motor cortex channels (C3, C4) - indices 8 and 10 in the full list
        # For dummy data, let\'s just take mean and std of a few channels
        # and some simple band power features
        
        # Ensure eeg_data has enough channels, otherwise use a default
        if eeg_data.shape[1] < 11: # Check if C4 (index 10) is present
            # Fallback for smaller dummy data, take first two channels
            eeg_mi = eeg_data[:, :2]
        else:
            mi_channels = [8, 10]  # C3, C4
            eeg_mi = eeg_data[:, mi_channels]

        eeg_mi = self.apply_notch_filter(eeg_mi)  
        eeg_mi = eeg_mi - np.median(eeg_mi, axis=0)  
        
        features = []
        for ch_data in eeg_mi.T:
            features.append(np.mean(ch_data))
            features.append(np.std(ch_data))
            # Simple band power for mu (8-13 Hz) and beta (13-30 Hz)
            band_mu = self.apply_bandpass_filter(ch_data.reshape(-1,1), 8, 13).flatten()
            band_beta = self.apply_bandpass_filter(ch_data.reshape(-1,1), 13, 30).flatten()
            features.append(np.var(band_mu))
            features.append(np.var(band_beta))

        return np.array(features)

    def apply_cca(self, eeg_data, target_freqs, harmonics=2):
        """Apply CCA for SSVEP feature extraction"""
        cca_features = []
        # Simplified CCA for dummy data - just return some values
        for freq in target_freqs:
            cca_features.append(np.random.rand()) # Dummy CCA value
        return np.array(cca_features)

    def extract_ssvep_features(self, eeg_data):
        """Simplified SSVEP feature extraction for dummy data."""
        eeg_data = self.apply_notch_filter(eeg_data)
        
        features = []
        
        # CCA features (simplified)
        cca_feats = self.apply_cca(eeg_data, self.ssvep_freqs)
        features.extend(cca_feats)
        
        # Frequency domain features for a few channels (simplified)
        # Ensure eeg_data has enough channels, otherwise use a default
        channels_to_process = [17, 18] if eeg_data.shape[1] > 18 else [0, 1] # O1, O2 or first two

        for ch_idx in channels_to_process:
            channel_data = eeg_data[:, ch_idx]
            fft_vals = np.abs(fft(channel_data))
            freqs = fftfreq(len(channel_data), 1/self.fs)
            
            for freq_band in self.ssvep_freqs:
                idx_fund = np.argmin(np.abs(freqs - freq_band))
                features.append(fft_vals[idx_fund])
                
        return np.array(features)

    def train_mi_model(self, X_mi, y_mi):
        print("Training MI model...")
        # Encode labels
        y_mi_encoded = self.mi_label_encoder.fit_transform(y_mi)
        
        # Scale features
        self.mi_scaler.fit(X_mi) # Fit scaler here
        X_mi_scaled = self.mi_scaler.transform(X_mi)
        
        # Train the ensemble model
        self.mi_ensemble.fit(X_mi_scaled, y_mi_encoded)
        self.mi_model = self.mi_ensemble
        print("MI model training complete.")

    def train_ssvep_model(self, X_ssvep, y_ssvep):
        print("Training SSVEP model...")
        # Encode labels
        y_ssvep_encoded = self.ssvep_label_encoder.fit_transform(y_ssvep)
        
        # Scale features
        self.ssvep_scaler.fit(X_ssvep) # Fit scaler here
        X_ssvep_scaled = self.ssvep_scaler.transform(X_ssvep)
        
        # Train the ensemble model
        self.ssvep_ensemble.fit(X_ssvep_scaled, y_ssvep_encoded)
        self.ssvep_model = self.ssvep_ensemble
        print("SSVEP model training complete.")

    def predict_mi(self, eeg_data):
        features = self.extract_mi_features(eeg_data)
        # Ensure features is 2D for scaler.transform
        features_scaled = self.mi_scaler.transform(features.reshape(1, -1))
        prediction_encoded = self.mi_model.predict(features_scaled)
        return self.mi_label_encoder.inverse_transform(prediction_encoded)[0]

    def predict_ssvep(self, eeg_data):
        features = self.extract_ssvep_features(eeg_data)
        # Ensure features is 2D for scaler.transform
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


def generate_dummy_eeg_data(num_samples=1000, num_channels=19, duration_seconds=2, fs=250, task_type="mi"):
    """Generates dummy EEG data for MI or SSVEP tasks."""
    print(f"Generating dummy {task_type} EEG data...")
    n_timepoints = duration_seconds * fs
    eeg_data = np.random.randn(n_timepoints, num_channels) * 10 # Simulate EEG noise

    if task_type == "mi":
        labels = np.random.choice(["Left", "Right"], num_samples)
        all_eeg_data = []
        for i in range(num_samples):
            current_eeg = np.random.randn(n_timepoints, num_channels) * 10
            # Simple simulation: add a distinct value to a channel based on label
            if labels[i] == "Left":
                current_eeg[:, 0] += 50 # Channel 0 for Left
            else:
                current_eeg[:, 1] += 50 # Channel 1 for Right
            all_eeg_data.append(current_eeg)
        return np.array(all_eeg_data), np.array(labels)

    elif task_type == "ssvep":
        ssvep_freqs = [7, 8, 10, 13] 
        labels = np.random.choice(["Forward", "Backward", "Left", "Right"], num_samples)
        all_eeg_data = []
        for i in range(num_samples):
            current_eeg = np.random.randn(n_timepoints, num_channels) * 10
            # Simple simulation: add a distinct value to a channel based on label
            if labels[i] == "Forward":
                current_eeg[:, 0] += 70 # Channel 0 for Forward
            elif labels[i] == "Backward":
                current_eeg[:, 1] += 70 # Channel 1 for Backward
            elif labels[i] == "Left":
                current_eeg[:, 2] += 70 # Channel 2 for Left
            elif labels[i] == "Right":
                current_eeg[:, 3] += 70 # Channel 3 for Right
            all_eeg_data.append(current_eeg)
        return np.array(all_eeg_data), np.array(labels)
    else:
        raise ValueError("task_type must be \"mi\" or \"ssvep\"")

def train_main():
    # Create an instance of the classifier
    bci_classifier = EnhancedBCIClassifier()

    # --- MI Training ---
    print("\n--- Starting MI Training ---")
    # Generate dummy MI data
    mi_eeg_data, mi_labels = generate_dummy_eeg_data(num_samples=50, duration_seconds=1, num_channels=19, task_type="mi")
    
    # Extract features for MI data
    # Reshape mi_eeg_data to be 2D for feature extraction if it\'s 3D (num_samples, timepoints, channels)
    # The extract_mi_features expects (timepoints, channels)
    X_mi = np.array([bci_classifier.extract_mi_features(eeg_sample) for eeg_sample in mi_eeg_data])
    y_mi = mi_labels

    # Train MI model
    bci_classifier.train_mi_model(X_mi, y_mi)

    # --- SSVEP Training ---
    print("\n--- Starting SSVEP Training ---")
    # Generate dummy SSVEP data
    ssvep_eeg_data, ssvep_labels = generate_dummy_eeg_data(num_samples=50, duration_seconds=1, num_channels=19, task_type="ssvep")

    # Extract features for SSVEP data
    X_ssvep = np.array([bci_classifier.extract_ssvep_features(eeg_sample) for eeg_sample in ssvep_eeg_data])
    y_ssvep = ssvep_labels

    # Train SSVEP model
    bci_classifier.train_ssvep_model(X_ssvep, y_ssvep)

    # Save the trained models and preprocessors
    bci_classifier.save_models()
    print("\nTraining complete and models saved.")

if __name__ == "__main__":
    train_main()


