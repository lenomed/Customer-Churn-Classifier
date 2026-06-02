# Customer-Churn-Classifier
Customer churn prediction system for telecom customers using SMOTE, machine learning model comparison, and SHAP explainability, deployed with Streamlit.
# Customer Churn Classifier 📊

A machine learning project to predict customer churn for telco companies using XGBoost classification with SHAP explainability analysis.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Features](#features)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Model Explainability](#model-explainability)
- [Data Preprocessing](#data-preprocessing)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project builds a predictive model to identify customers at risk of churning from a telco service provider. By understanding which features drive churn decisions, businesses can implement targeted retention strategies to reduce customer attrition.

**Key Technologies:**
- XGBoost for classification
- SMOTE for handling class imbalance
- SHAP for model interpretability
- scikit-learn for evaluation metrics

## Dataset

**Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Dataset Size:** ~7,000+ customer records

**Target Variable:** `Churn` (Yes/No - customer left within last month)

**Features Include:**
- Customer demographics (gender, age, partner, dependents)
- Account information (tenure, contract type, billing method)
- Services subscribed (phone service, internet service, security features)
- Charges (monthly charges, total charges)

## Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Customer-Churn-Classifier.git
cd Customer-Churn-Classifier
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset and place it in the project root:
```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Required Libraries

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn (imblearn)
shap
```

Or install via:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
```

## Project Structure

```
Customer-Churn-Classifier/
├── README.md
├── requirements.txt
├── churn_classifier.py          # Main script
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── outputs/
    ├── shap_summary_plot.png
    └── shap_bar_plot.png
```

## Features

### Data Cleaning & Preprocessing
- ✅ Binary encoding of categorical variables (gender, partner, dependents, etc.)
- ✅ Mapping service features with 3-class encoding (Yes/No/No service)
- ✅ One-hot encoding for multi-class features (Contract, PaymentMethod, InternetService)
- ✅ Handling missing values in `TotalCharges` (filled with median)
- ✅ Removal of non-predictive features (customerID)

### Feature Engineering
- **AvgChargePerTenure:** Average charge normalized by customer tenure
- **ServiceScore:** Aggregate score of security/support services
- **HighRiskPayment:** Flag for electronic check payment method (high churn risk)

### Model Training
- XGBoost Classifier with optimized hyperparameters
- SMOTE resampling to handle class imbalance
- Train-test split: 66% training / 34% testing
- Stratified sampling to maintain class distribution

### Model Evaluation
- Classification report (precision, recall, F1-score)
- Confusion matrix
- ROC AUC score
- SHAP explainability plots

## Usage

Run the classifier:

```bash
python churn_classifier.py
```

### Output
The script generates:
1. **Data Insights:**
   - First 10 rows of cleaned data
   - Dataset statistics and info
   - Correlation with churn target

2. **Model Metrics:**
   - Classification report
   - Confusion matrix
   - ROC AUC score

3. **SHAP Visualizations:**
   - Summary plot (impact of features on predictions)
   - Bar plot (average feature importance)

## Model Performance

### Classification Results
```
Precision: [Metric Value]
Recall:    [Metric Value]
F1-Score:  [Metric Value]
ROC AUC:   [Metric Value]
```

**Confusion Matrix:**
- True Negatives (TN): [Value]
- False Positives (FP): [Value]
- False Negatives (FN): [Value]
- True Positives (TP): [Value]

*Note: Run the script to see exact metrics*

## Model Explainability

### SHAP Analysis
SHAP (SHapley Additive exPlanations) values help interpret individual predictions:

- **Summary Plot:** Shows which features have the largest impact on model predictions
- **Bar Plot:** Displays average absolute SHAP values for feature importance ranking

**Key Insights:**
- Identifies high-risk customers based on contract type, internet service, and payment method
- Explains individual churn probability scores
- Enables business stakeholders to understand model decisions

## Data Preprocessing

### Key Steps:

1. **Binary Encoding:**
   - Gender: Male=1, Female=0
   - Partner/Dependents: Yes=1, No=0
   - PhoneService/PaperlessBilling: Yes=1, No=0
   - Churn: Yes=1, No=0

2. **Service Features Mapping:**
   - Yes=1, No=0, No internet service=2
   - Applied to: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

3. **Categorical Encoding:**
   - One-hot encoding for: MultipleLines, Contract, PaymentMethod, InternetService
   - Dropped first category to avoid multicollinearity

4. **Handling Missing Values:**
   - TotalCharges: Converted to numeric, filled with median value

5. **Feature Removal:**
   - Dropped customerID (non-predictive)
   - Dropped TotalCharges from features (potential data leakage)

## Handling Class Imbalance

**Problem:** Unequal distribution of churned vs. non-churned customers

**Solution:** 
- Applied SMOTE (Synthetic Minority Over-sampling Technique) to training data
- Calculated `scale_pos_weight` for XGBoost to penalize minority class misclassification

## Future Improvements

- [ ] Hyperparameter tuning using GridSearchCV or Bayesian Optimization
- [ ] Cross-validation for robust performance estimation
- [ ] Additional feature engineering (tenure segments, charge ratios)
- [ ] Ensemble methods combining multiple models
- [ ] Deploy model as REST API
- [ ] Create customer retention action plan based on predictions

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed as a machine learning classification project for telco customer retention analysis.

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- Libraries: XGBoost, scikit-learn, SHAP
- Inspiration: Real-world customer churn prediction challenges

## Contact

For questions or suggestions, feel free to open an issue or contact the project maintainer.

---

**Happy predicting! 🎯**