import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Definición de los cursos como características (inputs)
cursos = {
    'Calculo1': [1, 0, 0, 0, 0],
    'Fisica1': [0, 1, 0, 0, 0],
    'Quimica': [0, 0, 1, 0, 0],
    'ContruccionSostenible': [0, 0, 0, 1, 0],
    'Ingles': [0, 0, 0, 0, 1]
}

# Etiquetas (output) para representar un plan de estudios válido (ejemplo)
planes_validos = [
     [1, 1, 0, 1, 1],  # Un plan de estudios válido (Calculo1, Fisica1, ConstruccionSostenible, Ingles)
    [1, 0, 1, 0, 1],  # Otro plan válido (Calculo1, Quimica, Ingles)
    [1, 1, 1, 0, 0],  # Otro plan válido (Calculo1, Fisica1, Quimica)
    [0, 1, 1, 1, 0],  # Otro plan válido (Fisica1, Quimica, ConstruccionSostenible)
    [1, 1, 0, 0, 1]   # Otro plan válido (Calculo1, Fisica1, Ingles).
]

# Convertir los datos a numpy arrays para el entrenamiento
X = np.array(list(cursos.values()))
y = np.array(planes_validos)


#construimos las red neuronal con keras }
model = tf.keras.Sequential([
    layers.Dense(10, input_shape=(5,), activation='relu'),  # Capa oculta
    layers.Dense(5, activation='sigmoid')  # Capa de salida
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X, y, epochs=100, verbose=1)


# Predicción: dado un vector de entrada, la red recomienda un plan de estudios
nuevo_plan = np.array([[1, 0, 1, 0, 0]])  # Ejemplo de entrada (deseo tomar Calculo1 y Fisica1)
prediccion = model.predict(nuevo_plan)

# Interpretación del resultado
print("\nPlan recomendado por la Neurana creada es  (predicción segun Criterios Dados ):")
for idx, curso in enumerate(cursos):
    if prediccion[0][idx] >= 0.5:  # Umbral para considerar que un curso debe ser tomado
        print(curso)
