import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# DATA
data = load_breast_cancer()
X, y = data.data, data.target
# csv: df = pd.read_csv("data.csv"); X = df.drop("target", axis=1).values; y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),  # L2
    layers.BatchNormalization(),                                                      # BatchNorm
    layers.Dropout(0.3),                                                              # Dropout
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")   # multi-class: Dense(n, "softmax")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",   # multi-class: sparse_categorical_crossentropy
              metrics=["accuracy"])

early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, callbacks=[early_stop])

loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss:", loss, "| Test accuracy:", accuracy)

# PLOTS
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.show()

plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.show()
