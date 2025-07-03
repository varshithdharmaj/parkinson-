```markdown
# 🧠 Parkinson’s Disease Detection Using Machine Learning

This project uses machine learning techniques to detect Parkinson’s Disease (PD) based on vocal biomarkers. It leverages biomedical voice features like jitter, shimmer, and pitch, enabling early-stage detection through predictive analytics.

> 🚨 This is an academic project inspired by the Parkinson’s dataset from UCI Machine Learning Repository and builds on techniques discussed in [this IEEE reference](https://ieeexplore.ieee.org/document/10863299), but is independently implemented.

---

## 🚀 Overview

Parkinson's Disease often presents early symptoms in a person's voice. This project aims to classify individuals as having Parkinson’s or not, using a range of voice features such as:

- Fundamental frequency (Fo), max frequency (Fhi), min frequency (Flo)
- Frequency instability (Jitter), Amplitude variations (Shimmer)
- Nonlinear measures (PPE), noise-to-harmonics ratio (NHR)

---

## 📁 Project Structure

```
parkinson-disease-detector/
│
├── parkinson_mini.ipynb        # Main ML notebook
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Ignored files
└── data/                       # (Optional) Voice feature data
```

---

## 🔧 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/varshithdharmaj/parkinson-disease-detector.git
cd parkinson-disease-detector
```

2. **Install required packages**

```bash
pip install -r requirements.txt
```

3. **Run the Jupyter Notebook**

```bash
jupyter notebook parkinson_mini.ipynb
```

---

## 🧪 Sample Input (6 Key Features)

The model can be tested using simplified 6-feature vectors:

```
# Non-Parkinson’s
115.856,131.111,111.555,0.00234,0.00176,0.148349

# Parkinson’s
119.992,157.302,74.997,0.00784,0.00370,0.284654
```

Features: Fo, Fhi, Flo, Jitter(%), RAP, PPE

---

## 🧠 ML Pipeline

- Data preprocessing (Standard Scaling, Null checks)
- Feature selection & correlation analysis
- Model training: Random Forest, Logistic Regression, XGBoost
- Model evaluation: Accuracy, Precision, Recall, F1
- Optional: SHAP/LIME for explainability

---

## 🎯 Goals & Extensions

- ✅ Build a Streamlit front-end for live predictions
- ✅ Integrate voice file (.wav) input and extract MFCC features
- ✅ Compare classic ML with CNN-based spectrogram models
- 🔬 Add progression tracking across multiple recordings

---

## 📚 Reference

IEEE Paper: *A Parkinson’s Disease Detection System Based on Machine Learning*  
[IEEE Xplore Link](https://ieeexplore.ieee.org/document/10863299)

---

## 🙋‍♂️ Author

**Varshith Dharmaj**  
🎓 B.Tech CSE (AI & DS), Hyderabad  
📫 GitHub: [@varshithdharmaj](https://github.com/varshithdharmaj)

---

## 🛡 Disclaimer

This is a research-oriented project and **not** a medically certified diagnostic tool. For educational use only.

```

