# Interpretable Music Genre Classification Using Audio Feature Engineering

**Dataset:** [GTZAN Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

## Project Overview

This project explores music genre classification using classical machine learning models
applied to audio features extracted from the GTZAN dataset. Rather than using deep
learning, the focus is on understanding the full ML pipeline — feature preprocessing,
model training, evaluation, and interpretability.

Four models are trained and compared:
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

---

## Repository Structure
```
music-genre-classification/
├── data/
│   ├── raw/                    # Original GTZAN CSV files (not tracked by git)
│   └── processed/              # Train/val/test splits after preprocessing
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_interpretability.ipynb
├── outputs/
│   ├── models/                 # Saved .pkl model files
│   └── figures/                # All generated plots
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AnjelikaBab/music-genre-classification.git
cd music-genre-classification
```

### 2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
and place `features_30_sec.csv` inside `data/raw/`.

### 4. Run notebooks in order
```
01_dataset_exploration.ipynb
02_preprocessing.ipynb
03_model_training.ipynb
04_evaluation.ipynb
05_interpretability.ipynb
```

Each notebook saves its outputs so the next one can load them directly.

---

## Results

| Model               | Train Accuracy | Val Accuracy | Test Accuracy |
|---------------------|---------------|--------------|----------------|
| KNN                 | 0.791         | 0.627        | 0.673          |
| Logistic Regression | 0.910         | 0.780        | 0.733          |
| SVM                 | 0.884         | 0.753        | 0.713          |
| Random Forest       | 0.999         | 0.820        | 0.740          |

Random Forest achieved the highest accuracy. Rock was the hardest genre to classify
(f1=0.40) due to sonic overlap with blues, country, and metal.

---

## Key Findings

- MFCCs were the most important features for classification
- Classical, metal, and pop were easiest to classify due to distinct audio profiles
- Rock was hardest due to broad genre overlap with blues, country, and metal
- Random Forest had the highest accuracy but also the most overfitting
- Logistic Regression showed the best generalization relative to its complexity

---

## Requirements

See `requirements.txt`. Main dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- jupyter

---

## References

- [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- [TensorFlow GTZAN Catalog](https://www.tensorflow.org/datasets/catalog/gtzan)
