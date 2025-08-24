import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

df = pd.read_csv("fake_full_dataset_ready.csv")

X = df[
    [
        "Axial Load Capacity (kN)",
        "Shear Capacity (kN)",
        "Bending & Torsion Resistance (kN·m)",
        "Ductility (%)",
        "Fatigue Resistance (Cycles)",
        "Constructability (Rating: 1–5)",
        "Fabrication Time (Hours)",
        "Labor & Equipment Cost (₱)",
        "Lifecycle Cost (₱)"
    ]
].values

y_class = df["Optimal Connection Type (Encoded)"].values  # Classification
y_reg = df[["Performance Score", "Safety Margin (%)", "Material Usage (kg)"]].values  # Regression

from sklearn.model_selection import train_test_split
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

inputs = keras.Input(shape=(X_train.shape[1],))
x = layers.Dense(128, activation="relu")(inputs)
x = layers.Dense(64, activation="relu")(x)

class_output = layers.Dense(2, activation="softmax", name="class_output")(x)

reg_output = layers.Dense(3, activation="linear", name="reg_output")(x)

model = keras.Model(inputs=inputs, outputs=[class_output, reg_output])

model.compile(
    optimizer="adam",
    loss={"class_output": "sparse_categorical_crossentropy", "reg_output": "mse"},
    metrics={"class_output": "accuracy", "reg_output": ["mae"]}
)

history = model.fit(
    X_train,
    {"class_output": y_class_train, "reg_output": y_reg_train},
    validation_data=(X_test, {"class_output": y_class_test, "reg_output": y_reg_test}),
    epochs=50,
    batch_size=8,
    verbose=1
)

results = model.evaluate(
    X_test,
    {"class_output": y_class_test, "reg_output": y_reg_test},
    verbose=1
)
print("Test Results:", results)

model.save("multi_task_nn.h5")
print("✅ Model and scaler saved (multi_task_nn.h5, scaler.pkl)")
