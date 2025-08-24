import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

model = keras.models.load_model("multi_task_nn.keras")
scaler = joblib.load("scaler.pkl")

label_map = {0: "Standard", 1: "High-strength"}

new_input = np.array([[1500, 800, 250, 20, 50000, 4, 40, 75000, 500000]])

new_input_scaled = scaler.transform(new_input)

class_pred, reg_pred = model.predict(new_input_scaled)

class_label = np.argmax(class_pred, axis=1)[0]

print("🔹 Predicted Connection Type:", label_map[class_label])
print("🔹 Predicted Regression Outputs:")
print("   - Performance Score:", round(reg_pred[0][0], 2))
print("   - Safety Margin (%):", round(reg_pred[0][1], 2))
print("   - Material Usage (kg):", round(reg_pred[0][2], 2))
