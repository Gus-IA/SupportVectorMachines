# Support Vector Machines: Clasificación y Regresión con Python

Este repositorio contiene una colección de ejemplos prácticos usando **Support Vector Machines (SVM)** para resolver problemas tanto de **clasificación** como de **regresión**, usando distintos kernels y configuraciones en Python.

## 📚 Contenidos

Este proyecto abarca:

### 🔹 1. Clasificación con SVM lineales
- Uso del dataset de **Iris**, específicamente con las clases *Virginica* y *Versicolor*.
- Comparación del efecto del hiperparámetro `C` en el margen de decisión y vectores de soporte.
- Visualización de márgenes, fronteras de decisión y vectores de soporte.

### 🔹 2. Clasificación no lineal
- Generación de datos con `make_moons`.
- Aplicación de `PolynomialFeatures` + `LinearSVC` para crear una SVM no lineal.
- Uso de `SVC` con kernel **polinomial** y **RBF** para clasificaciones más complejas.
- Comparación de hiperparámetros: `degree`, `gamma`, `C`, `coef0`.

### 🔹 3. Regresión con SVM
- Ejemplos usando `LinearSVR` y `SVR` con kernel polinomial.
- Visualización de márgenes epsilon y vectores de soporte.
- Comparación del impacto de `epsilon`, `C` y el tipo de kernel.

## 🧠 Lo aprendido

- Diferencias entre **SVC** y **SVR**, así como sus versiones lineales.
- Cómo el hiperparámetro **C** controla el margen y la penalización por errores.
- Uso de diferentes **kernels** (`linear`, `poly`, `rbf`) para adaptar el modelo a los datos.
- Visualización de márgenes de decisión y vectores de soporte.
- Aplicación de `Pipeline` para encadenar transformaciones y modelos.
- Importancia de la **normalización** (`StandardScaler`) al usar SVMs.

## 📊 Ejemplos visuales

Los gráficos generados por este código permiten visualizar claramente:

- Líneas de decisión
- Márgenes
- Vectores de soporte
- Fronteras no lineales
- Curvas de regresión y márgenes epsilon

## Para instalar las dependencias necesarias, puedes usar el siguiente comando:

pip install -r requirements.txt


🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.

