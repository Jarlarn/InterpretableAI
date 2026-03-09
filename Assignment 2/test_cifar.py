import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.models import load_model
import tensorflow as tf

# Load model and data
model = load_model("cifar_cnn_model.keras")
(_, _), (cifar_x_test, cifar_y_test) = tf.keras.datasets.cifar10.load_data()
cifar_x_test = cifar_x_test.astype("float32")

# Normalize test data using training mean and std if available
mean_training = np.mean(cifar_x_test)
std_dev_training = np.std(cifar_x_test)
cifar_x_test = (cifar_x_test - mean_training) / std_dev_training

# Get predictions
pred_probs = model.predict(cifar_x_test)
pred_labels = np.argmax(pred_probs, axis=1)

# Evaluate and print accuracy
test_loss, test_acc = model.evaluate(cifar_x_test, cifar_y_test)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Set up the figure and axis
fig, ax = plt.subplots()
img_display = ax.imshow(cifar_x_test[0].astype(np.float32), cmap="viridis")
title = ax.set_title("")

# CIFAR-10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def update(frame):
    img_display.set_data(cifar_x_test[frame].astype(np.float32))
    true_label = class_names[int(cifar_y_test[frame])]
    pred_label = class_names[int(pred_labels[frame])]
    title.set_text(f"True: {true_label} | Predicted: {pred_label}")
    return img_display, title


ani = animation.FuncAnimation(
    fig, update, frames=100, interval=500, blit=False, repeat=True
)

plt.show()
