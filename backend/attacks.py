import numpy as np
import tensorflow as tf
from art.attacks.evasion import FastGradientMethod, DeepFool, SquareAttack, SimBA
from art.estimators.classification import TensorFlowV2Classifier

CLIP_VALUES_DICT = {
    "mnist": (0.0, 1.0),
    "cifar100": (0.0, 1.0),
    "gtsrb": (0.0, 1.0),
    "imagenet": (-128.0, 128.0),
}

def get_clip_values(dataset):
    return CLIP_VALUES_DICT.get(dataset.lower(), (0.0, 1.0))

def get_classifier(model, dataset, nb_classes=10, input_shape=(28, 28, 1)):
    clip_values = get_clip_values(dataset)
    return TensorFlowV2Classifier(
        model=model,
        nb_classes=nb_classes,
        input_shape=input_shape,
        loss_object=tf.keras.losses.CategoricalCrossentropy(),
        channels_first=False,
        clip_values=clip_values,
    )

def fgsm_attack(model, img, epsilon=0.1, dataset="mnist"):
    nb_classes = 10 if dataset == "mnist" else (100 if dataset == "cifar100" else 43)
    input_shape = img.shape
    art_classifier = get_classifier(model, dataset, nb_classes=nb_classes, input_shape=input_shape)
    attack = FastGradientMethod(estimator=art_classifier, eps=epsilon)  # <--- estimator
    adv_img = attack.generate(img[np.newaxis, ...])
    return adv_img[0]

def deepfool_attack(model, img, dataset="mnist", max_iter=50, epsilon=1e-6):
    nb_classes = 10 if dataset == "mnist" else (100 if dataset == "cifar100" else 43)
    input_shape = img.shape
    art_classifier = get_classifier(model, dataset, nb_classes=nb_classes, input_shape=input_shape)
    attack = DeepFool(classifier=art_classifier, max_iter=max_iter, epsilon=epsilon)  # <--- classifier
    adv_img = attack.generate(img[np.newaxis, ...])
    return adv_img[0]

def simba_attack(model, img, dataset="mnist", max_iter=200, epsilon=0.2):
    nb_classes = 10 if dataset == "mnist" else (100 if dataset == "cifar100" else 43)
    input_shape = img.shape
    art_classifier = get_classifier(model, dataset, nb_classes=nb_classes, input_shape=input_shape)
    attack = SimBA(classifier=art_classifier, max_iter=max_iter, epsilon=epsilon)  # <--- classifier
    adv_img = attack.generate(img[np.newaxis, ...])
    return adv_img[0]

def square_attack(model, img, dataset="mnist", epsilon=0.05, max_iter=1000):
    nb_classes = 10 if dataset == "mnist" else (100 if dataset == "cifar100" else 43)
    input_shape = img.shape
    art_classifier = get_classifier(model, dataset, nb_classes=nb_classes, input_shape=input_shape)
    attack = SquareAttack(estimator=art_classifier, eps=epsilon, max_iter=max_iter)  # <--- estimator
    adv_img = attack.generate(img[np.newaxis, ...])
    return adv_img[0]