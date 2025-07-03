import numpy as np
from PIL import Image
import random
import os
from tensorflow.keras.datasets import mnist, cifar100
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.resnet50 import preprocess_input

def get_random_mnist_sample():
    """
    Returns a random MNIST test sample as (img, label).
    Tries to use datasets/mnist/test.npz if exists, otherwise loads from keras.
    Output img is (28, 28, 1), float32, [0,1].
    """
    try:
        data = np.load("datasets/mnist/test.npz")
        imgs, labels = data["images"], data["labels"]
        idx = random.randint(0, len(imgs) - 1)
        img = imgs[idx].astype("float32") / 255.0
        img = img[..., np.newaxis]
        label = labels[idx]
    except Exception:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        idx = np.random.randint(0, x_test.shape[0])
        img = x_test[idx].astype("float32") / 255.0
        img = np.expand_dims(img, -1)
        label = y_test[idx]
    return img, label

def get_random_mnist_samples(n=10):
    """
    Returns n random MNIST test samples as (imgs, labels).
    Each img is (28, 28, 1), float32, [0,1].
    """
    try:
        data = np.load("datasets/mnist/test.npz")
        imgs, labels = data["images"], data["labels"]
        idxs = np.random.choice(len(imgs), n, replace=False)
        imgs = imgs[idxs].astype("float32") / 255.0
        imgs = imgs[..., np.newaxis]
        labels = labels[idxs]
    except Exception:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        idxs = np.random.choice(np.arange(x_test.shape[0]), n, replace=False)
        imgs = x_test[idxs].astype("float32") / 255.0
        imgs = np.expand_dims(imgs, -1)
        labels = y_test[idxs]
    return imgs, labels

def get_random_cifar100_sample():
    """
    Returns a random CIFAR-100 test sample as (img, label).
    Tries to use datasets/cifar100/test.npz if exists, otherwise loads from keras.
    Output img is (32, 32, 3), float32, [0,1].
    """
    try:
        data = np.load("datasets/cifar100/test.npz")
        imgs, labels = data["images"], data["labels"]
        idx = random.randint(0, len(imgs) - 1)
        img = imgs[idx].astype("float32") / 255.0
        label = labels[idx]
    except Exception:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
        idx = np.random.randint(0, x_test.shape[0])
        img = x_test[idx].astype("float32") / 255.0
        label = y_test[idx][0]
    return img, label

def get_random_imagenet_sample():
    """
    Returns a random sample ImageNet image from datasets/imagenet_samples/ (img, label=-1).
    Output img is (224, 224, 3), float32, preprocess_input() (ResNet50 style).
    """
    # Path to the folder containing ImageNet sample images
    folder_path = os.path.join("datasets", "imagenet_samples")
    # List all image files (jpg/jpeg/png)
    valid_exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    if not files:
        raise FileNotFoundError(f"No images found in {folder_path}")
    # Choose a random image
    chosen = random.choice(files)
    img_path = os.path.join(folder_path, chosen)
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array.astype("float32"))
    label = -1  # No ground-truth
    return img_array, label

def get_random_gtsrb_sample():
    """
    Returns a random GTSRB test sample as (img, label).
    Assumes datasets/gtsrb/test_images/ and test_labels.txt exist.
    Output img is (48, 48, 3), float32, [0,1].
    """
    test_dir = "datasets/gtsrb/test_images"
    label_file = "datasets/gtsrb/test_labels.txt"
    if os.path.exists(test_dir) and os.path.exists(label_file):
        with open(label_file) as f:
            lines = f.readlines()
        sample = random.choice(lines).strip().split()
        img_path, label = os.path.join(test_dir, sample[0]), int(sample[1])
        img = Image.open(img_path).resize((48, 48))
        img_array = np.array(img).astype("float32") / 255.0
        # If grayscale, convert to RGB
        if img_array.ndim == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 1:
            img_array = np.concatenate([img_array]*3, axis=-1)
        return img_array, label
    else:
        raise FileNotFoundError("GTSRB test set not found. Please prepare datasets/gtsrb/test_images and test_labels.txt")