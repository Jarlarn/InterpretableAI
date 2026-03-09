import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ============================================================
# Load data
# ============================================================

training_data = np.loadtxt("A1_training.txt")
validation_data = np.loadtxt("A1_validation.txt")
test_data = np.loadtxt("A1_test.txt")

# ============================================================
# Prepare data for LSTM
# ============================================================
# LSTM expects input shape: (samples, timesteps, features)
# We use a sliding window of size `window_size` to predict the next value.


def create_sequences(data, window_size):
    X = []
    y = []
    for t in range(window_size, len(data)):
        X.append(data[t - window_size : t])
        y.append(data[t])
    X = np.array(X)
    y = np.array(y)
    # Reshape X to (samples, timesteps, 1) for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y


window_size = 10  # Number of past values used to predict the next one

X_train, y_train = create_sequences(training_data, window_size)
X_val, y_val = create_sequences(validation_data, window_size)
X_test, y_test = create_sequences(test_data, window_size)

# ============================================================
# Build and train LSTM model
# ============================================================


def build_and_train(num_units, window_size, X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(LSTM(num_units, activation="tanh", input_shape=(window_size, 1)))
    model.add(Dense(1))

    model.compile(optimizer=Adam(), loss="mse")

    # Save the best model based on validation loss
    checkpoint = ModelCheckpoint(
        f"best_lstm_{num_units}.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0,
    )

    # Stop training early if validation loss stops improving
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    return model, history


# ============================================================
# Experiment with different numbers of LSTM units
# ============================================================

unit_options = [8, 16, 32, 64]


best_val_rmse = float("inf")
best_units = None
best_model = None
best_pred = None

for num_units in unit_options:
    print(f"\n--- Training LSTM with {num_units} units ---")
    model, history = build_and_train(
        num_units, window_size, X_train, y_train, X_val, y_val
    )

    # Predict on validation set
    val_pred = model.predict(X_val).flatten()
    val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    print(f"LSTM ({num_units} units) - Validation RMSE: {val_rmse:.4f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_units = num_units
        best_model = model

# After selecting the best model, evaluate on test set
best_pred = best_model.predict(X_test).flatten()
test_rmse = np.sqrt(np.mean((y_test - best_pred) ** 2))
print(
    f"\nBest LSTM: {best_units} units with Validation RMSE: {best_val_rmse:.4f} and Test RMSE: {test_rmse:.4f}"
)

# ============================================================
# Plot: test data vs best LSTM prediction
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(y_test, label="Test Data", color="blue")
plt.plot(
    best_pred,
    label=f"LSTM ({best_units} units) Prediction",
    color="red",
    linestyle="--",
)
plt.title(f"Test Data vs LSTM ({best_units} units)  |  RMSE = {best_rmse:.4f}")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("task_lstm_prediction.png")
plt.show()
