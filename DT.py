# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

# Generamos datos simples (variables x1, x2 y la salida y)
X = np.array([[2, 3],
              [4, 2],
              [7, 5],
              [8, 6],
              [6, 4],
              [5, 3],
              [3, 4]])

y = np.array([4, 6, 10, 12, 8, 7, 5])

# Crear el modelo de regresión con un árbol de decisión
tree_regressor = DecisionTreeRegressor(max_depth=2)  # Definimos una profundidad máxima de 2 para la simplicidad

# Ajustamos el modelo a los datos
tree_regressor.fit(X, y)

# Mostrar las reglas del árbol
plt.figure(figsize=(12,8))
tree.plot_tree(tree_regressor, feature_names=["x1", "x2"], filled=True)
plt.show()

# Hacemos predicciones con el modelo
X_new = np.array([[5, 4], [7, 5]])  # Nuevas muestras para predecir
y_pred = tree_regressor.predict(X_new)

print("Predicciones para las nuevas muestras:", y_pred)

