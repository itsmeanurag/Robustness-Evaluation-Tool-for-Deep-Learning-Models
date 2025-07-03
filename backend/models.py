import tensorflow as tf

def load_mnist_model(model_path="models/mnist_model.keras"):
    return tf.keras.models.load_model(model_path)

def load_cifar100_model(model_path="models/cifar100_model.keras"):
    return tf.keras.models.load_model(model_path)

def load_gtsrb_model(model_path="models/gtsrb_model.keras"):
    return tf.keras.models.load_model(model_path)

def load_imagenet_model(model_name="ResNet50"):
    """
    Loads a pretrained ImageNet model by name.
    Supported: ResNet50, MobileNetV2, InceptionV3, EfficientNetB0, EfficientNetB1, DenseNet121
    """
    tf.keras.backend.clear_session()
    if model_name == "ResNet50":
        from tensorflow.keras.applications import ResNet50
        return ResNet50(weights="imagenet")
    elif model_name == "MobileNetV2":
        from tensorflow.keras.applications import MobileNetV2
        return MobileNetV2(weights="imagenet")
    elif model_name == "InceptionV3":
        from tensorflow.keras.applications import InceptionV3
        return InceptionV3(weights="imagenet")
    elif model_name == "EfficientNetB0":
        from tensorflow.keras.applications import EfficientNetB0
        return EfficientNetB0(weights="imagenet")
    elif model_name == "EfficientNetB1":
        from tensorflow.keras.applications import EfficientNetB1
        return EfficientNetB1(weights="imagenet")
    elif model_name == "DenseNet121":
        from tensorflow.keras.applications import DenseNet121
        return DenseNet121(weights="imagenet")
    else:
        raise ValueError(f"Unknown model name: {model_name}")