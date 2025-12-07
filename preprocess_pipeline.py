import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_german_data(df):
    df = df.copy()
    df.fillna('Unknown', inplace=True)
    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    scaler = StandardScaler()
    df[['Age', 'Credit amount', 'Duration']] = scaler.fit_transform(df[['Age', 'Credit amount', 'Duration']])
    return df

def clean_credit_data(df):
    df = df.copy()
    df.fillna(0, inplace=True)  # fill NAs if any

    num_cols = ['rev_util', 'age', 'late_30_59', 'debt_ratio', 'monthly_inc', 
                'open_credit', 'late_90', 'real_estate', 'late_60_89', 
                'dependents', 'dlq_2yrs']

    # Convert all numeric columns safely
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convert invalid entries to NaN
    df.fillna(0, inplace=True)  # replace NaNs with 0

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

if __name__ == '__main__':
    # --- German credit data ---
    german_df = pd.read_csv('german_credit_data.csv')
    clean_german = clean_german_data(german_df)
    clean_german.to_csv('german_credit_cleaned.csv', index=False)
    print("German credit data cleaned and saved as german_credit_cleaned.csv.")

    # --- Credit risk benchmark data ---
    credit_df = pd.read_csv('Credit Risk Benchmark Dataset.csv')
    clean_credit = clean_credit_data(credit_df)
    clean_credit.to_csv('credit_risk_cleaned.csv', index=False)
    print("Credit risk data cleaned and saved as credit_risk_cleaned.csv.")
