

import pandas as pd


#importo test
test_data = pd.read_csv(
    "C:\\Users\\Usuario\\Documents\\Maestria Nacional\\1.Semestre\\3.Predictiva\\Laboratorio1\\20204-2-LAB-02-prediccion-del-default-usando-logreg-jcastrillonmo\\files\\input\\test_data.csv.zip",
    index_col=False,
    compression="zip",
)
# importo train
train_data = pd.read_csv(
    "C:\\Users\\Usuario\\Documents\\Maestria Nacional\\1.Semestre\\3.Predictiva\\Laboratorio1\\20204-2-LAB-02-prediccion-del-default-usando-logreg-jcastrillonmo\\files\\input\\train_data.csv.zip",
    index_col=False,
    compression="zip",
)

# - Renombre la columna "default payment next month" a "default".

test_data = test_data.rename(columns={'default payment next month': 'default'})
train_data = train_data.rename(columns={'default payment next month': 'default'})
#print(test_data.columns)


# - Remueva la columna "ID".
test_data=test_data.drop(columns=['ID'])
train_data=train_data.drop(columns=['ID'])

# - Elimine los registros con informacion no disponible.

import numpy as np

# Para train_data
train_data = train_data.loc[train_data["MARRIAGE"] != 0]
train_data = train_data.loc[train_data["EDUCATION"] != 0]
#print(train_data["EDUCATION"].value_counts())
#print(train_data["MARRIAGE"].value_counts())

# Para test_data (corregido)
test_data = test_data.loc[test_data["MARRIAGE"] != 0]
test_data = test_data.loc[test_data["EDUCATION"] != 0]
#print(test_data["EDUCATION"].value_counts())
#print(test_data["MARRIAGE"].value_counts())

# - Para la columna EDUCATION, valores > 4 indican niveles superiores
test_data['EDUCATION'] = test_data['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
train_data['EDUCATION'] = train_data['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

#print(train_data["EDUCATION"].value_counts())
#print(test_data["EDUCATION"].value_counts())

#Paso 2
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train=train_data.drop(columns="default")
y_train=train_data["default"]


x_test=test_data.drop(columns="default")
y_test=test_data["default"]

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_regression

# fclassifier

#Columnas categoricas
categorical_features=["SEX","EDUCATION","MARRIAGE"]

#preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder="passthrough"
)
# Preprocesamiento para variables numéricas: Escalado al intervalo [0, 1]
numerical_preprocessor = MinMaxScaler()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

""" Crear un pipeline con regresión logística
pipeline_lr = Pipeline(steps=["scaler",MinMaxScaler
    ("preprocessing", preprocessor),  # Preprocesamiento: codificación y escalado
    ("feature_selection", SelectKBest(score_func=f_classif)),  # Selección de características
    ("classifier", LogisticRegression(max_iter=1000))  # Modelo de regresión logística
])"""

#Alternativa2
pipeline_lr = Pipeline(steps=[
    ("encoder", preprocessor),                       # Transformación de categorías
    ("scaler", MinMaxScaler()),                      # Escalado numérico
    ("feature_selection", SelectKBest(f_classif)),   # Selección de características
    ("classifier", LogisticRegression(max_iter=1000))  # Modelo final
])

# Definir la rejilla de parámetros
param_grid = {
    "feature_selection__k": range(1, len(x_train.columns) + 1),
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"]
}

model = GridSearchCV(
    estimator=pipeline_lr,
    param_grid=param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# Ajustar el modelo con el conjunto de entrenamiento
model.fit(x_train, y_train)

# Obtener los mejores parámetros y puntuación
best_params = model.best_params_
best_score = model.best_score_

# Mostrar resultados
print("Mejores hiperparámetros encontrados:")
print(best_params)
print(f"Mejor puntuación (accuracy): {best_score:.4f}")

# Evaluar el mejor modelo en el conjunto de prueba
best_model = model.best_estimator_
test_accuracy = best_model.score(x_test, y_test)
print(f"Exactitud en el conjunto de prueba: {test_accuracy:.4f}")

# Paso 5.
# Salve el modelo como "files/models/model.pkl".

import pickle
import os

models_dir = 'files/models'
os.makedirs(models_dir, exist_ok=True)

with open("files/models/model.pkl.gz","wb") as file:
    pickle.dump(model,file)

# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
import json
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

def calculate_and_save_metrics(model, x_train, x_test, y_train, y_test):
    # Hacer predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calcular métricas para el conjunto de entrenamiento
    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
    }

    # Calcular métricas para el conjunto de prueba
    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
    }

    # Crear carpeta si no existe
    output_dir = 'files/output'
    os.makedirs(output_dir, exist_ok=True)

    # Guardar las métricas en un archivo JSON
    output_path = os.path.join(output_dir, 'metrics.json')
    with open(output_path, 'w') as f:  # Usar 'w' para comenzar con un archivo limpio
        f.write(json.dumps(metrics_train) + '\n')
        f.write(json.dumps(metrics_test) + '\n')

# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

from sklearn.metrics import confusion_matrix
# Función para calcular las matrices de confusión y guardarlas
def calculate_and_save_confusion_matrices(model, x_train, x_test, y_train, y_test):
    # Hacer predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calcular matrices de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Convertir las matrices de confusión en formato JSON
    def format_confusion_matrix(cm, dataset_type):
        return {
            'type': 'cm_matrix',
            'dataset': dataset_type,
            'true_0': {
                'predicted_0': int(cm[0, 0]),
                'predicted_1': int(cm[0, 1])
            },
            'true_1': {
                'predicted_0': int(cm[1, 0]),
                'predicted_1': int(cm[1, 1])
            }
        }

    metrics = [
        format_confusion_matrix(cm_train, 'train'),
        format_confusion_matrix(cm_test, 'test')
    ]

    # Guardar las matrices de confusión en el mismo archivo JSON
    output_path = 'files/output/metrics.json'
    with open(output_path, 'a') as f:  # Usar 'a' para agregar después de las métricas
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')

# Función principal para ejecutar todo
def main(model, x_train, x_test, y_train, y_test):
    # Crear el directorio de salida si no existe
    import os
    os.makedirs('files/output', exist_ok=True)

    # Calcular y guardar las métricas
    calculate_and_save_metrics(model, x_train, x_test, y_train, y_test)

    # Calcular y guardar las matrices de confusión
    calculate_and_save_confusion_matrices(model, x_train, x_test, y_train, y_test)

# Ejemplo de uso:
main(model, x_train, x_test, y_train, y_test)