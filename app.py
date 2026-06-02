from numpy.random import RandomState
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# create outputs dir if it do no t exist
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created '{output_dir}' directory")

# save outputs
def save_plot(filename):
    """Save plot to outputs directory"""
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def save_text(filename, content):
    """Save text content to outputs directory"""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✓ Saved: {filepath}")


df = pd.read_csv(r'C:\Users\kelec\source\repos\Customer-Churn-Classifier\WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head(10))
print(df.describe())
print(df.info())
print(df.columns)



# cleaning data
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# The features with customer service equals "Yes" has Nan across the columns for each row it appears 
internet_service_cols = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

mapping = {'Yes': 1, 'No': 0, 'No internet service': 2}  # i had to bring this code to this center to make sure i convert it after filling the kind of missign values
for col in internet_service_cols:
    df[col] = df[col].map(mapping)



df = pd.get_dummies(
    df,
    columns=['MultipleLines'],
    drop_first=True,
    dtype=int
)

df = pd.get_dummies(
    df,
    columns=['Contract'],
    drop_first=True,
    dtype=int
)

df = pd.get_dummies(
    df,
    columns=['PaymentMethod'],
    drop_first=True,
    dtype=int
)

df = pd.get_dummies(df, columns=['InternetService'], drop_first=True, dtype=int)
"""sns.countplot(df['Contract'])
plt.show()"""

#sns.countplot(x='Churn', data=df)
#plt.show()
#sns.countplot(df['PaymentMethod'])
#plt.show()

#print(df.iloc[:,4:])
print(df['Churn'])

print(df.head(10))
print(df.describe())
print(df.info())
print(df.columns)

#print(df['customerID'])
df = df.drop('customerID', axis=1) # the customer ID is usless here because it doesn't determine the diecision or affect the decision making of a customer
#print(df['customerID'])
#print(df.columns)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

print(df.isnull().sum())
print(df.head(10))
print(df.describe())
print(df.info())
print(df.columns)

#sns.countplot(x='InternetService', data=df)
#plt.show()

print(df.isnull().sum())
print(df.head(20))


#sns.pairplot(df, hue='Churn')
#plt.show()

print(df['MonthlyCharges'])
corr_series = df.corr(numeric_only=True)['Churn'].sort_values(ascending=False)
print(corr_series)

# SAVE CORRELATION ANALYSIS
corr_report = f"=== CHURN CORRELATION ANALYSIS ===\n\n{corr_series.to_string()}\n"
save_text('correlation_analysis.txt', corr_report)


"""# spliting my data
cols_to_drop = [
    'Churn',
    'TotalCharges',
    'gender',
    'PhoneService',
    'MultipleLines_No phone service'
]"""
#X = df.drop('Churn', axis=1)
#y=df['Churn']

# 1. FEATURE ENGINEERING TO KNOW WHETHER IT WOULD REDUCE REDUNDANCY

# Avoid division by zero
df["AvgChargePerTenure"] = df["TotalCharges"] / (df["tenure"] + 1)

df["ServiceScore"] = (
    df["OnlineSecurity"] +
    df["TechSupport"] +
    df["DeviceProtection"] +
    df["OnlineBackup"]
)

df["HighRiskPayment"] = (df["PaymentMethod_Electronic check"] == 1).astype(int)


# 2. DROP TARGET + LEAKAGE FEATURES

X = df.drop(["Churn", "TotalCharges"], axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.34, random_state=101, stratify=y)


scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

print(f"\n=== TRAINING INFO ===")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Scale pos weight: {scale_pos_weight}")


smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

modelx = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=100,
    learning_rate=0.03,
    max_depth=2,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)

modelx.fit(X_train, y_train)
pred = modelx.predict(X_test)

print('\n=== XGBOOST RESULTS ===')
classification_rep = classification_report(y_test, pred)
print(classification_rep)

conf_matrix = confusion_matrix(y_test, pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# SAVE CLASSIFICATION REPORT
class_report_text = f"=== CLASSIFICATION REPORT ===\n\n{classification_rep}\n\n=== CONFUSION MATRIX ===\n\n{conf_matrix}\n"
save_text('classification_report.txt', class_report_text)


from sklearn.metrics import roc_auc_score

probs = modelx.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)

print(f"\nROC AUC: {auc}")

# SAVE ROC AUC SCORE
auc_text = f"ROC AUC Score: {auc}\n"
save_text('roc_auc_score.txt', auc_text)


# IMPLEMENTING SHAP
print("\n=== GENERATING SHAP EXPLANATIONS ===")

# Create SHAP explainer for tree-based models
explainer = shap.TreeExplainer(modelx)

# Compute SHAP values for test set
shap_values = explainer.shap_values(X_test)

print("Creating SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
save_plot('shap_summary_plot.png')


print("Creating SHAP bar plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
save_plot('shap_bar_plot.png')

# OPTIONAL: Save SHAP force plot for first 5 predictions
print("Creating SHAP force plots for first 5 predictions...")
for i in range(min(5, len(X_test))):
    plt.figure(figsize=(12, 4))
    shap.force_plot(explainer.expected_value, shap_values[i:i+1], X_test.iloc[i:i+1], matplotlib=True, show=False)
    save_plot(f'shap_force_plot_{i}.png')

