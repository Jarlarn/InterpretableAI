import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.models import load_model
import tensorflow as tf

# Load model and data
model = load_model("cifar_cnn_model.h5")
(_, _), (mnist_x_test, mnist_y_test) = tf.keras.datasets.mnist.load_data(
    path="mnist.npz"
)
mnist_x_test = mnist_x_test.astype("float32") / 255.0

# Reshape for model if needed (add channel dimension)
if len(mnist_x_test.shape) == 3:
    mnist_x_test = np.expand_dims(mnist_x_test, -1)

# Get predictions
pred_probs = model.predict(mnist_x_test)
pred_labels = np.argmax(pred_probs, axis=1)

# Set up the figure and axis
fig, ax = plt.subplots()
img_display = ax.imshow(mnist_x_test[0].squeeze(), cmap="gray")
title = ax.set_title("")


def update(frame):
    img_display.set_data(mnist_x_test[frame].squeeze())
    true_label = mnist_y_test[frame]
    pred_label = pred_labels[frame]
    title.set_text(f"True: {true_label} | Predicted: {pred_label}")
    return img_display, title


ani = animation.FuncAnimation(
    fig, update, frames=100, interval=500, blit=False, repeat=True
)

plt.show()
