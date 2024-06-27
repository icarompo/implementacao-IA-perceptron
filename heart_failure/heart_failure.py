# Imports
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
data = pd.read_csv("./heart_failure.csv")

# Separar o campo objetivo
X = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]

# Divisao do conjunto de dados e Normalização
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurações para testar
configs = [
    {"hidden_layer_sizes": (10,), "max_iter": 500},
    {"hidden_layer_sizes": (20,), "max_iter": 500},
    {"hidden_layer_sizes": (10, 10), "max_iter": 500},
]

for config in configs:
    mlp = MLPClassifier(hidden_layer_sizes=config["hidden_layer_sizes"], max_iter=config["max_iter"], random_state=42)
    start_time = time.time()
    mlp.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    y_pred = mlp.predict(X_test_scaled)
    prediction_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Camadas ocultas: {config['hidden_layer_sizes']}")
    print(f"Acurácia do modelo MLP: {accuracy:.4f}")
    print(f"Tempo de treinamento: {training_time:.4f} segundos")
    print(f"Tempo de previsão: {prediction_time:.4f} segundos")
    print()
    
    # Plotar gráfico de dispersão
    plt.figure(figsize=(8, 6))
    colors = ['blue' if val == 0 else 'orange' for val in y_test]
    plt.scatter(X_test['age'], X_test['serum_creatinine'], c=colors)
    plt.xlabel('Age')
    plt.ylabel('Serum Creatinine')
    plt.title(f'Scatter plot of Age vs Serum Creatinine (Hidden Layers: {config["hidden_layer_sizes"]})')
    plt.show()
