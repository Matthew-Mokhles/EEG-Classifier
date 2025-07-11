# 🧠 Enhanced Brain-Computer Interface (BCI) Classifier

An advanced EEG classification system designed for Brain-Computer Interface (BCI) competitions. This project supports both **Motor Imagery (MI)** and **Steady-State Visual Evoked Potential (SSVEP)** paradigms with robust preprocessing, feature engineering, and ensemble classification techniques.

> 🏆 **Used in an official BCI competition submission to address complex EEG classification tasks with high accuracy and generalization.**

---

## 🚀 Key Features

- **Dual-Paradigm Support:** Fully supports MI and SSVEP EEG signal classification.
- **Aggressive Preprocessing:** Includes notch/bandpass filtering, ICA artifact removal, and robust scaling.
- **Advanced Feature Engineering:**
  - **MI:** Time, frequency, wavelet, and entropy-based features from C3/C4.
  - **SSVEP:** Canonical Correlation Analysis (CCA), PSD, wavelets.
- **Ensemble Learning:**
  - Ultra-regularized StackingClassifier for MI.
  - Balanced StackingClassifier with passthrough for SSVEP.
- **Overfitting Protection:** Regularization, early stopping, cross-validation, and conservative model selection.
- **Battle-Tested:** Applied successfully in a competitive setting with strong performance.

---

## 🧱 Architecture Overview

```
EEG → Preprocessing → Feature Extraction → Classification → Command
```

- Preprocessing: Filters, ICA, RobustScaler
- Feature Engineering: Custom extraction for MI/SSVEP
- Classification: XGBoost, SVM, RF, MLP, LDA, etc.
- Ensemble: Meta-learners for final prediction

---

## 🧪 Paradigm-specific Models

### 🔁 Motor Imagery (MI)
- Features: Asymmetry, spectral bands (delta → ultra-gamma), CSP-like filters
- Classifiers: XGBoost, RandomForest, LDA, SVM, MLP, AdaBoost, ExtraTrees, KNN
- Ensemble: Ultra-conservative stacking with Logistic Regression (C=0.1)

### 🔂 SSVEP
- Features: Canonical Correlation (CCA), PSD, wavelets
- Classifiers: XGBoost, SVM (3 kernels), RF, MLP, Ridge, Naive Bayes, LDA
- Ensemble: Balanced stacking with passthrough (C=2.0)

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Scikit-learn**, **XGBoost**
- **NumPy**, **SciPy**, **PyWavelets**
- **MNE (optional)** for EEG preprocessing
- **Matplotlib / Seaborn** for visualizations

---

## 📂 Project Structure

```
bci_classifier/
├── preprocessing.py       # Filtering, ICA, scaling
├── feature_engineering.py # MI/SSVEP-specific feature extractors
├── models.py              # Model initialization and ensemble logic
├── train_mi.py            # MI training pipeline
├── train_ssvep.py         # SSVEP training pipeline
├── utils/
│   └── metrics.py         # Evaluation helpers
├── README.md
└── requirements.txt
```

---

## 📈 Example Usage

```bash
# Train MI classifier
python train_mi.py --data data/MI_subjects.csv

# Train SSVEP classifier
python train_ssvep.py --data data/SSVEP_trials.csv
```

---

## 🏆 Competition Use

This system was developed and deployed for an official **Brain-Computer Interface Competition**, where it demonstrated:

- 📊 Accurate classification across multiple EEG paradigms  
- ⚙️ Generalization through advanced ensemble learning  
- 💡 Innovation in handling overfitting/underfitting using adaptive strategies  

---

## 🤝 Acknowledgments

Special thanks to the organizers of the competition and mentors who supported the development of this system.

---

## 📜 License

This project is intended for educational and research purposes. Please contact for permission before use in commercial applications.

---

## 📬 Contact

**Matthew Mokhles**  
✉️ [matthewmokhles@gmail.com](mailto:matthewmokhles@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/matthew-mokhles-b31779269/)
