import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from keras import models, layers

tf.config.list_physical_devices("GPU")

# # Fixing mnist data
(mnist_image_train, mnist_label_train), (mnist_image_test, mnist_label_test) = (
    tf.keras.datasets.mnist.load_data(path="mnist.npz")
)

mnist_image_train, mnist_image_test = (
    mnist_image_train.astype("float32") / 255.0,
    mnist_image_test.astype("float32") / 255.0,
)
# CNN for mnist
model = models.Sequential()
# Layer 1 Conv2d, 32 filters, 3x3
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
# Layer 2 max pool 2x2
model.add(layers.MaxPooling2D(2, 2))
# Layer 3 conv2d 64, 3x3
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
# Layer 4
model.add(layers.MaxPooling2D(2, 2))
# Layer 5
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.35))
# Layer 6
model.add(layers.Dense(10, activation="softmax"))

model.summary()


model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

history = model.fit(
    mnist_image_train,
    mnist_label_train,
    epochs=10,
    validation_data=(mnist_image_test, mnist_label_test),
)

# # Fixing cifar10 data
(cifar_image_train, cifar_label_train), (cifar_image_test, cifar_label_test) = (
    tf.keras.datasets.cifar10.load_data()
)

cifar_image_train = cifar_image_train.astype("float32")
cifar_image_test = cifar_image_test.astype("float32")

mean_training = np.mean(cifar_image_train)
std_dev_training = np.std(cifar_image_train)

cifar_image_train = (cifar_image_train - mean_training) / std_dev_training
cifar_image_test = (cifar_image_test - mean_training) / std_dev_training

model_cifar = models.Sequential()
# Layer 1
model_cifar.add(
    layers.Conv2D(
        64, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same"
    )
)
model_cifar.add(layers.BatchNormalization())

# Layer 2
model_cifar.add(layers.MaxPooling2D(2, 2))

model_cifar.add(
    layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)
    )
)
model_cifar.add(layers.BatchNormalization())
# Layer 3
model_cifar.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
model_cifar.add(layers.BatchNormalization())

# Layer 4
model_cifar.add(layers.MaxPooling2D(2, 2))

# Layer 5
model_cifar.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
model_cifar.add(layers.BatchNormalization())

# Layer 7
model_cifar.add(layers.Conv2D(1024, (3, 3), activation="relu", padding="same"))
model_cifar.add(layers.BatchNormalization())

# Layer 8
model_cifar.add(layers.MaxPooling2D(2, 2))

model_cifar.add(layers.GlobalAveragePooling2D())
model_cifar.add(layers.Dense(256, activation="relu"))
model_cifar.add(layers.Dropout(0.5))
model_cifar.add(layers.Dense(128, activation="relu"))
model_cifar.add(layers.Dropout(0.35))
model_cifar.add(layers.Dense(10, activation="softmax"))


model_cifar.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

history = model_cifar.fit(
    cifar_image_train,
    cifar_label_train,
    epochs=10,
    validation_data=(cifar_image_test, cifar_label_test),
)

# model_cifar.save("cifar_cnn_model.keras")


# --------------------------------------------------------
# KNN MODEL
# --------------------------------------------------------


def compute_distances(train_data, test_data):
    """Compute pairwise Euclidean distances between test and train data."""
    train_sq = np.sum(train_data**2, axis=1)  # (N,)
    test_sq = np.sum(test_data**2, axis=1)  # (M,)
    cross = test_data @ train_data.T  # (M, N)
    dists = np.sqrt(
        np.maximum(test_sq[:, None] + train_sq[None, :] - 2 * cross, 0)
    )  # (M, N)
    return dists


def knn_predict_batch(train_data, train_labels, test_data, k):
    """Vectorized KNN prediction for a batch of test points."""
    dists = compute_distances(train_data, test_data)

    # Get k nearest neighbors for each test point
    k_nearest_idx = np.argpartition(dists, k, axis=1)[:, :k]  # (M, k)
    k_nearest_labels = train_labels[k_nearest_idx]  # (M, k)

    # Majority vote per test point
    predictions = np.array(
        [Counter(row).most_common(1)[0][0] for row in k_nearest_labels]
    )
    return predictions


def evaluate_knn(train_data, train_labels, eval_data, eval_labels, k):
    """Evaluate KNN accuracy for a given k."""
    preds = knn_predict_batch(train_data, train_labels, eval_data, k)
    accuracy = np.mean(preds == eval_labels)
    return accuracy


def run_knn(
    x_train, y_train, x_test, y_test, dataset_name, k_range=range(1, 26)
):
    """Run KNN: split train into train/val, sweep k, report test accuracy."""
    # Flatten images to 1D vectors (pixel-by-pixel Euclidean distance)
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    # Split training data: last 10,000 as validation
    val_size = 10000
    x_val = x_train_flat[-val_size:]
    y_val = y_train_flat[-val_size:]
    x_tr = x_train_flat[:-val_size]
    y_tr = y_train_flat[:-val_size]

    print(f"\n{'=' * 50}")
    print(f"KNN on {dataset_name}")
    print(
        f"Training size: {len(x_tr)}, Validation size: {len(x_val)}, Test size: {len(x_test_flat)}"
    )
    print(f"{'=' * 50}")

    # Sweep k on validation set
    best_k = 1
    best_val_acc = 0.0
    for k in k_range:
        val_acc = evaluate_knn(x_tr, y_tr, x_val, y_val, k)
        print(f"  k={k:2d}  ->  Validation Accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k

    print(f"\nBest k = {best_k} (Validation Accuracy: {best_val_acc:.4f})")

    # Evaluate on test set with best k (using full training data)
    test_acc = evaluate_knn(
        x_train_flat, y_train_flat, x_test_flat, y_test_flat, best_k
    )
    print(f"Test Accuracy with k={best_k}: {test_acc:.4f}")

    return best_k, test_acc


def visualize_knn(
    x_train,
    y_train,
    x_test,
    y_test,
    k,
    dataset_name,
    num_samples=5,
    img_shape=None,
):
    """Show test images alongside their k nearest neighbors for interpretability."""
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    # CIFAR-10 class names for readable labels
    cifar10_classes = [
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

    # Pick random test samples
    indices = np.random.choice(len(x_test_flat), num_samples, replace=False)

    dists = compute_distances(x_train_flat, x_test_flat[indices])

    for sample_idx, test_idx in enumerate(indices):
        # Get k nearest neighbor indices sorted by distance
        nn_idx = np.argsort(dists[sample_idx])[:k]
        nn_labels = y_train_flat[nn_idx]
        nn_dists = dists[sample_idx][nn_idx]

        # Prediction via majority vote
        pred_label = Counter(nn_labels).most_common(1)[0][0]
        true_label = y_test_flat[test_idx]
        correct = pred_label == true_label

        # Create figure: 1 test image + k neighbor images
        fig, axes = plt.subplots(1, k + 1, figsize=(2.5 * (k + 1), 3))
        fig.suptitle(
            f"{dataset_name} — True: {cifar10_classes[true_label] if 'CIFAR' in dataset_name else true_label}, "
            f"Predicted: {cifar10_classes[pred_label] if 'CIFAR' in dataset_name else pred_label} "
            f"({'Correct' if correct else 'WRONG'})",
            fontsize=12,
            color="green" if correct else "red",
            fontweight="bold",
        )

        # Show test image
        test_img = x_test[test_idx]
        if img_shape:
            test_img = test_img.reshape(img_shape)
        cmap = "gray" if test_img.ndim == 2 else None
        # For CIFAR, undo normalization for display
        if test_img.ndim == 3:
            test_img = (test_img - test_img.min()) / (
                test_img.max() - test_img.min()
            )
        axes[0].imshow(test_img, cmap=cmap)
        axes[0].set_title("Test Image", fontsize=10)
        axes[0].axis("off")

        # Show k nearest neighbors
        for j, ni in enumerate(nn_idx):
            neighbor_img = x_train[ni]
            if img_shape:
                neighbor_img = neighbor_img.reshape(img_shape)
            if neighbor_img.ndim == 3:
                neighbor_img = (neighbor_img - neighbor_img.min()) / (
                    neighbor_img.max() - neighbor_img.min()
                )
            lbl = y_train_flat[ni]
            lbl_str = (
                cifar10_classes[lbl] if "CIFAR" in dataset_name else str(lbl)
            )
            axes[j + 1].imshow(neighbor_img, cmap=cmap)
            axes[j + 1].set_title(
                f"NN{j + 1}: {lbl_str}\nd={nn_dists[j]:.2f}", fontsize=9
            )
            axes[j + 1].axis("off")

        plt.tight_layout()
        plt.savefig(f"{dataset_name}_knn_sample_{sample_idx + 1}.png", dpi=150)
        plt.show()


def main():
    # KNN on MNIST
    mnist_best_k, mnist_test_acc = run_knn(
        mnist_image_train, mnist_label_train, mnist_image_test, mnist_label_test, "MNIST"
    )

    # KNN on CIFAR-10
    cifar_best_k, cifar_test_acc = run_knn(
        cifar_image_train,
        cifar_label_train,
        cifar_image_test,
        cifar_label_test,
        "CIFAR-10",
    )

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(
        f"MNIST:    best k={mnist_best_k}, test accuracy={mnist_test_acc:.4f}"
    )
    print(
        f"CIFAR-10: best k={cifar_best_k}, test accuracy={cifar_test_acc:.4f}"
    )

    # Visualize KNN predictions with nearest neighbors
    print("\nGenerating KNN visualizations...")
    visualize_knn(
        mnist_image_train,
        mnist_label_train,
        mnist_image_test,
        mnist_label_test,
        k=mnist_best_k,
        dataset_name="MNIST",
        num_samples=5,
        img_shape=(28, 28),
    )
    visualize_knn(
        cifar_image_train,
        cifar_label_train,
        cifar_image_test,
        cifar_label_test,
        k=cifar_best_k,
        dataset_name="CIFAR-10",
        num_samples=5,
    )


if __name__ == "__main__":
    main()
