# Heart Disease Classification using K-Nearest Neighbors

A machine learning project that predicts the presence of heart disease in patients using the K-Nearest Neighbors (KNN) algorithm, achieving an impressive accuracy of **91.30%**.

## 📋 Overview

This project demonstrates the application of machine learning in medical diagnosis by analyzing patient data to predict heart disease risk. The model processes various clinical parameters and provides binary classification (presence or absence of heart disease).

## 🎯 Objective

The primary goal is to develop a robust classification model that can accurately predict whether a patient has heart disease based on medical indicators. This tool aims to assist healthcare professionals in early detection and risk assessment.

## 📊 Dataset

### Data Source
The `heart.csv` dataset contains **918 patient records** with **12 clinical attributes**.

### Features Description

| Feature | Description | Type |
|---------|-------------|------|
| **Age** | Patient's age in years | Continuous |
| **Sex** | Gender (M/F) | Categorical |
| **ChestPainType** | Type of chest pain (ATA, NAP, ASY, TA) | Categorical |
| **RestingBP** | Resting blood pressure (mm Hg) | Continuous |
| **Cholesterol** | Serum cholesterol level (mg/dl) | Continuous |
| **FastingBS** | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) | Binary |
| **RestingECG** | Resting electrocardiogram results (Normal, ST, LVH) | Categorical |
| **MaxHR** | Maximum heart rate achieved | Continuous |
| **ExerciseAngina** | Exercise induced angina (Y/N) | Binary |
| **Oldpeak** | ST depression induced by exercise | Continuous |
| **ST_Slope** | Slope of peak exercise ST segment (Up, Flat, Down) | Categorical |
| **HeartDisease** | Target variable (1 = disease, 0 = normal) | Binary |

## 🔧 Requirements
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Installation
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Or install all dependencies at once:
```bash
pip install -r requirements.txt
```

## 🚀 Methodology

### 1. Data Loading and Exploration
```python
import pandas as pd
heart_data = pd.read_csv("heart.csv")
```

Initial data exploration includes checking for missing values, duplicates, and understanding the distribution of features.

### 2. Data Preprocessing

#### Handling Missing Values
- **Numerical columns**: Imputed using mean strategy
- **Categorical columns**: Imputed using most frequent value strategy

#### Addressing Class Imbalance
Implemented upsampling technique to balance the dataset between male and female patients, ensuring the model doesn't develop gender bias.

#### Feature Encoding
Applied One-Hot Encoding to transform categorical variables into numerical format suitable for machine learning algorithms.

#### Feature Scaling
Used StandardScaler to normalize numerical features, ensuring all features contribute equally to the model's predictions.

### 3. Data Splitting

The dataset was split using stratified sampling to maintain class distribution:
- **Training set**: 80% (734 samples)
- **Testing set**: 20% (184 samples)

### 4. Model Training
```python
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2
)
classifier.fit(X_train, y_train)
```

The KNN algorithm was chosen for its simplicity and effectiveness in classification tasks. The model considers the 5 nearest neighbors using Euclidean distance (Minkowski with p=2).

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 91.30% |
| **Precision** (Disease class) | 87% |
| **Recall** (Disease class) | 93% |
| **F1-Score** (Disease class) | 90% |

### Confusion Matrix
```
                    Predicted
                No Disease  Disease
Actual  
No Disease         155        17
Disease             9        118
```

**Analysis**:
- **True Negatives (TN)**: 155 - Correctly identified healthy patients
- **False Positives (FP)**: 17 - Healthy patients incorrectly identified as having disease
- **False Negatives (FN)**: 9 - Patients with disease incorrectly identified as healthy
- **True Positives (TP)**: 118 - Correctly identified patients with disease

The model demonstrates strong performance with minimal false negatives, which is crucial in medical diagnosis where missing a disease case could have serious consequences.

## 💻 Usage

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/Saif13m05/Heart-Disease-Classification-KNN.git
cd Heart-Disease-Classification-KNN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook classification_final.ipynb
```

4. Run all cells sequentially to train and evaluate the model.

### Making Predictions

To predict heart disease for a new patient:
```python
# Example patient data
new_patient = {
    'Age': 55,
    'Sex': 'M',
    'ChestPainType': 'NAP',
    'RestingBP': 145,
    'Cholesterol': 333,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 250,
    'ExerciseAngina': 'N',
    'Oldpeak': 0.0,
    'ST_Slope': 'Up'
}

# Preprocess and predict
prediction = model.predict(preprocess(new_patient))
print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
```

## 📁 Project Structure
```
Heart-Disease-Classification-KNN/
│
├── classification_final.ipynb    # Main notebook with complete workflow
├── heart.csv                      # Dataset file
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## 🔍 Key Insights

1. **Data Balancing**: Upsampling proved effective in preventing gender bias in predictions
2. **Feature Scaling**: Standardization significantly improved model performance
3. **Feature Engineering**: One-Hot Encoding properly handled categorical variables
4. **Model Selection**: KNN with k=5 provided optimal balance between bias and variance
5. **Evaluation Strategy**: Stratified split ensured representative testing

## 🎓 Future Improvements

- Experiment with different k values using cross-validation
- Try ensemble methods (Random Forest, XGBoost)
- Implement hyperparameter tuning with GridSearchCV
- Add feature importance analysis

## ⚠️ Disclaimer

**Important**: This project is for educational purposes only and should not be used as a substitute for professional medical advice or diagnosis.

## 📧 Contact

For questions or suggestions, feel free to open an [issue](https://github.com/Saif13m05/Heart-Disease-Classification-KNN/issues).

---

**Built with ❤️ using Machine Learning**
