import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("multi_task_nn.keras")
scaler = joblib.load("scaler.pkl")

new_input = np.array([[1500, 800, 250, 20, 50000, 4, 40, 75000, 500000]])
new_input_scaled = scaler.transform(new_input)

class_pred, reg_pred = model.predict(new_input_scaled)

class_idx = np.argmax(class_pred, axis=1)[0]
performance_score, safety_margin, material_usage = reg_pred[0]

print("Optimal Connection Type (Encoded):", class_idx)
print("Performance Score:", round(performance_score, 2))
print("Safety Margin (%):", round(safety_margin, 2))
print("Material Usage (kg):", round(material_usage, 2))
