import pandas as pd
import numpy as np

df = pd.read_csv("./heart_failure.csv")

null_counts = df.isnull().sum()
print("Valores nulos por coluna:")
print(null_counts)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

df.to_csv("./heart_failure_cleaned.csv", index=False)
print("Tratamento Concluido")
