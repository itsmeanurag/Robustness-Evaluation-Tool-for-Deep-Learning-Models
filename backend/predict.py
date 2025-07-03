import numpy as np
from PIL import Image
import random
import os
from tensorflow.keras.datasets import mnist, cifar100
from tensorflow.keras.utils import get_file

# Preprocess functions for various ImageNet models
from tensorflow.keras.applications import (
    resnet50,
    mobilenet_v2,
    inception_v3,
    efficientnet,
    densenet,
)

IMAGENET_PREPROCESS = {
    "ResNet50": resnet50.preprocess_input,
    "MobileNetV2": mobilenet_v2.preprocess_input,
    "InceptionV3": inception_v3.preprocess_input,
    "EfficientNetB0": efficientnet.preprocess_input,
    "EfficientNetB1": efficientnet.preprocess_input,
    "DenseNet121": densenet.preprocess_input,
}

def predict_image(model, img, dataset=None, top=3):
    preds = model.predict(img[np.newaxis, ...])
    if dataset is not None and dataset.lower() == "imagenet":
        decoded = decode_predictions(preds, top=top)[0]
        return decoded
    else:
        pred_class = int(np.argmax(preds))
        conf = float(np.max(preds))
        return pred_class, conf

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

def get_random_imagenet_sample(model_name="ResNet50"):
    """
    Returns a sample ImageNet image (img, label=-1).
    Downloads a sample image and preprocesses it for the given model.
    Output img is (224, 224, 3), float32, preprocess_input().
    """
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
    path = get_file("imagenet_sample.jpg", url)
    img = Image.open(path).resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # Remove alpha if present
        img_array = img_array[..., :3]
    preprocess_func = IMAGENET_PREPROCESS.get(model_name, resnet50.preprocess_input)
    img_array = preprocess_func(img_array.astype("float32"))
    label = -1  # No ground-truth
    return img_array, label

def get_random_gtsrb_sample():
    """
    Returns a random GTSRB test sample as (img, label).
    Assumes datasets/gtsrb/test_images/ and test_labels.txt exist.
    Output img is (32, 32, 3) or (32, 32, 1), float32, [0,1].
    """
    test_dir = "datasets/gtsrb/test_images"
    label_file = "datasets/gtsrb/test_labels.txt"
    if os.path.exists(test_dir) and os.path.exists(label_file):
        with open(label_file) as f:
            lines = f.readlines()
        sample = random.choice(lines).strip().split()
        img_path, label = os.path.join(test_dir, sample[0]), int(sample[1])
        img = Image.open(img_path).resize((32, 32))
        img_array = np.array(img).astype("float32") / 255.0
        if img_array.ndim == 2:
            img_array = np.expand_dims(img_array, -1)
        return img_array, label
    else:
        raise FileNotFoundError(
            "GTSRB test set not found. Please prepare datasets/gtsrb/test_images and test_labels.txt"
        )