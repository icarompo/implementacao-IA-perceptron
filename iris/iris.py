import pandas as pd
import numpy as np
import sweetviz as sv

from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score

# Carregue o arquivo CSV para um DataFrame do Pandas
data = pd.read_csv("iris.csv")

#sv.analyze(data).show_html()


#msk = np.random.rand(len(data)) < 0.7
#train = data[msk]
#test = data[~msk]

#print(train.head())


#sv.compare([train, "conjunto de treinamento"], [test, "conjunto de teste"]).show_html()

# Separe os dados em features (X) e rótulos (y)
x = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

# Codifique as classes para números, se necessário
y = LabelEncoder().fit_transform(y)

# Divida os dados em conjuntos de treino e teste
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Inicializa arquitetura da RNA
mlp=MLPClassifier(activation='logistic', max_iter=100, hidden_layer_sizes=(5,), alpha=0.1, solver='lbfgs')

# Conjunto de treinamento
#mlp.fit(x_train, y_train)

# Prevendo a saída para os dados de teste
#predictions=mlp.predict(x_test)

#print("\nSaídas da RNA:", predictions)
#print("\nSaídas esperadas:", y_test, "\n")

'''
# Cria a matriz de confusão (problemas de classificação)
cm = metrics.confusion_matrix(y_test, predictions)
print("Matriz de Confusão:")
print(cm)

# Calcula a acurácia (problemas de classificação)
accuracy = metrics.accuracy_score(y_test, predictions)
print("Acurácia:", accuracy, "\n")
'''

# Implemente a validação cruzada
cross_val_scores = cross_val_score(mlp, x, y, cv=10)  # cv é o número de folds

# Exiba as pontuações de cada fold
print("Pontuações de validação cruzada:", cross_val_scores)

# Exiba a média das pontuações de validação cruzada
print("Média das pontuações de validação cruzada:", cross_val_scores.mean(), "\n")
