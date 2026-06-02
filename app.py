from numpy.random import RandomState
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv(r'C:\Users\kelec\source\repos\Customer-Churn-Classifier\WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head(10))
print(df.describe())
print(df.info())
print(df.columns)



# Data Cleaning
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

# 1. testing feature engineering to know wether it would reduce redundancy FEATURE ENGINEERING

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
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.34, random_state=101, stratify=y)


scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


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
print('below is XGboost and above is logistic')
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

from sklearn.metrics import roc_auc_score

probs = modelx.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)

print("ROC AUC:", auc)


# implementing shap
# Create SHAP explainer for tree-based models

explainer = shap.TreeExplainer(modelx)

# Compute SHAP values for test set
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")