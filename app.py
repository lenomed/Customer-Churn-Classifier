import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv(r'C:\Users\kelec\source\repos\Customer-Churn-Classifier\WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head(10))
print(df.describe())
print(df.info())
print(df.columns)


"""'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

for fts in df:
    sns.countplot(x=fts, data=df)
    plt.show()
    """
# Data Cleaning
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['InternetService'] = df['InternetService'].map({'DSL': 1, 'Fiber optic': 0})
df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 1, 'No': 0})
df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 1, 'No': 0})
df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 1, 'No': 0})
df['TechSupport'] = df['TechSupport'].map({'Yes': 1, 'No': 0})
df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1, 'No': 0})
df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 1, 'No': 0})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})



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
"""sns.countplot(df['Contract'])
plt.show()"""

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
print(df['TotalCharges'])

print(df.isnull().sum())

# The features with customer service equals "Yes" has Nan across the columns for each row it appears 

internet_related_cols = [
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies'
]

for col in internet_related_cols:
    df[col] = df[col].fillna('No internet service')

print(df.isnull().sum())

sns.pairplot(df)
plt.show()