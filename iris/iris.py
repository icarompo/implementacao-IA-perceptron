# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Dataset
data = pd.read_csv("./iris.csv")

# Separar os campos
X = data.drop('Species', axis=1)
y = data['Species']

# Divisao do conjunto de dados e Normalização
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
start_time = time.time()
mlp.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

# Predição
start_time = time.time()
y_pred = mlp.predict(X_test_scaled)
prediction_time = time.time() - start_time

# Taxa de acerto
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo MLP: {accuracy:.4f}')

# Tempo de processamento
print(f'Tempo de treinamento: {training_time:.4f} segundos')
print(f'Tempo de previsão: {prediction_time:.4f} segundos')

# Plotagem de gráfico
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_pred, palette='Set2', s=100, edgecolor='k', legend='full')
plt.title('Resultado da Classificação do MLP no Dataset Iris')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.show()
