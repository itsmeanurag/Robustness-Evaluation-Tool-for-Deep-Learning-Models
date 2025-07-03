import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from backend.attacks import fgsm_attack
from backend.dataset_utils import get_random_imagenet_sample

from tensorflow.keras.applications import (
    resnet50, mobilenet_v2, inception_v3, efficientnet, densenet
)

MODEL_DICT = {
    "ResNet50": resnet50,
    "MobileNetV2": mobilenet_v2,
    "InceptionV3": inception_v3,
    "EfficientNetB0": efficientnet,
    "EfficientNetB1": efficientnet,
    "DenseNet121": densenet,
}

MODEL_INPUT_SIZES = {
    "ResNet50": (224, 224),
    "MobileNetV2": (224, 224),
    "DenseNet121": (224, 224),
    "EfficientNetB0": (224, 224),
    "EfficientNetB1": (240, 240),
    "InceptionV3": (299, 299),
}

@st.cache_resource
def load_model(model_name):
    if model_name == "ResNet50":
        return resnet50.ResNet50(weights="imagenet")
    elif model_name == "MobileNetV2":
        return mobilenet_v2.MobileNetV2(weights="imagenet")
    elif model_name == "InceptionV3":
        return inception_v3.InceptionV3(weights="imagenet")
    elif model_name == "EfficientNetB0":
        return efficientnet.EfficientNetB0(weights="imagenet")
    elif model_name == "EfficientNetB1":
        return efficientnet.EfficientNetB1(weights="imagenet")
    elif model_name == "DenseNet121":
        return densenet.DenseNet121(weights="imagenet")

def preprocess_image(img_pil, model_name):
    size = MODEL_INPUT_SIZES[model_name]
    img = img_pil.convert("RGB").resize(size)
    arr = np.array(img)
    if model_name == "ResNet50":
        arr = resnet50.preprocess_input(arr.astype("float32"))
    elif model_name == "MobileNetV2":
        arr = mobilenet_v2.preprocess_input(arr.astype("float32"))
    elif model_name == "DenseNet121":
        arr = densenet.preprocess_input(arr.astype("float32"))
    elif model_name.startswith("EfficientNet"):
        arr = efficientnet.preprocess_input(arr.astype("float32"))
    elif model_name == "InceptionV3":
        arr = inception_v3.preprocess_input(arr.astype("float32"))
    return arr

def plot_robustness_curves(epsilons, avg_accs_dict, avg_conf_dict):
    # Accuracy plot
    plt.figure(figsize=(6, 4))
    color_cycle = plt.cm.tab10.colors
    handles = []
    labels = []
    for idx, (model, accs) in enumerate(avg_accs_dict.items()):
        h, = plt.plot(
            epsilons, accs, marker='o', label=model,
            color=color_cycle[idx % len(color_cycle)]
        )
        handles.append(h)
        labels.append(model)
    plt.xlabel("FGSM Epsilon")
    plt.ylabel("Average Top-1 Accuracy (%)")
    plt.ylim(-0.5, 105)
    # Set legend to upper right, smaller font, and tighter spacing
    plt.legend(handles, labels, title="Model", loc='upper right', bbox_to_anchor=(1, 1), fontsize=8, title_fontsize=9, markerscale=0.9, labelspacing=0.3)
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    # Confidence plot
    plt.figure(figsize=(6, 4))
    handles = []
    labels = []
    for idx, (model, confs) in enumerate(avg_conf_dict.items()):
        h, = plt.plot(
            epsilons, confs, marker='o', label=model,
            color=color_cycle[idx % len(color_cycle)]
        )
        handles.append(h)
        labels.append(model)
    plt.xlabel("FGSM Epsilon")
    plt.ylabel("Average Top-1 Confidence (%)")
    plt.ylim(-0.5, 105)
    plt.legend(handles, labels, title="Model", loc='upper right', bbox_to_anchor=(1, 1), fontsize=8, title_fontsize=9, markerscale=0.9, labelspacing=0.3)
    plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

def plot_robustness_curve_tab():
    st.header("🛡️ Robustness Evaluation (IMAGENET)")

    st.markdown("""
**How to use this Robustness Evaluation demo:**
1. Upload **5 to 10 images** (or click 'Random Sample' multiple times to collect).
2. Click **Run Robustness Evaluation** to analyze model robustness.
3. Two curves are shown for each model:
   - **Accuracy**: Fraction of images where the model's top-1 prediction remains unchanged after FGSM attack.
   - **Confidence**: The average of the model's maximum softmax output (confidence) on the adversarial images; if the class changes after attack, confidence is set to 0.
4. Lower accuracy/confidence at low epsilon = weaker robustness. All models are shown for comparison.
    """)

    st.subheader("1. Add Images")
    images = []
    uploaded_files = st.file_uploader(
        "Upload 5-10 images (JPG/PNG, will be resized automatically)", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True, 
        key="robustness_multi_uploader"
    )
    if uploaded_files:
        for uploaded in uploaded_files:
            img_pil = Image.open(uploaded).convert("RGB")
            images.append(img_pil)
    # "Random Sample" button: add 10 random samples to the pool
    if st.button("Add 10 Random ImageNet Samples", key="robustness_random_btn"):
        if "robustness_random_images" not in st.session_state:
            st.session_state["robustness_random_images"] = []
        for _ in range(10):
            img_arr_raw, _ = get_random_imagenet_sample()
            img_pil = Image.fromarray(np.uint8(np.clip(img_arr_raw, 0, 255))).convert("RGB")
            st.session_state["robustness_random_images"].append(img_pil)
    # Add random images from session to images list
    if "robustness_random_images" in st.session_state:
        images.extend(st.session_state["robustness_random_images"])
    # Show thumbnails
    if images:
        st.markdown("**Selected Images:**")
        st.image(images, width=120, caption=[f"Image {i+1}" for i in range(len(images))])

    num_images = len(images)
    st.info(f"Total images selected: {num_images}")

    # Epsilon values: 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.40, 0.46, 0.50, 0.55, 0.60
    epsilons = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.40, 0.46, 0.50, 0.55, 0.60]

    if num_images < 5:
        st.warning("Please add at least 5 images (upload or random) to run the evaluation.")
        return

    if st.button("Run Robustness Evaluation", key="robustness_eval_btn"):
        avg_accs_dict = {}
        avg_conf_dict = {}
        st.info("Evaluating models... This may take a few minutes depending on number of models and images.")
        for model_name in MODEL_DICT.keys():
            st.write(f"Evaluating {model_name} ...")
            model = load_model(model_name)
            accs = []
            confs = []
            for eps in epsilons:
                correct = 0
                conf_sum = 0
                for img_pil in images:
                    img_arr = preprocess_image(img_pil, model_name)
                    preds = model.predict(img_arr[np.newaxis, ...])
                    orig_top1 = np.argmax(preds)
                    adv_img = fgsm_attack(model, img_arr, epsilon=eps, dataset="imagenet")
                    preds_adv = model.predict(adv_img[np.newaxis, ...])
                    adv_top1 = np.argmax(preds_adv)
                    if adv_top1 == orig_top1:
                        correct += 1
                        adv_conf = preds_adv[0, adv_top1]
                    else:
                        adv_conf = 0.0
                    conf_sum += adv_conf
                acc = (correct / num_images) * 100.0
                avg_conf = (conf_sum / num_images) * 100.0
                accs.append(acc)
                confs.append(avg_conf)
            avg_accs_dict[model_name] = accs
            avg_conf_dict[model_name] = confs
        st.success("Evaluation complete!")
        plot_robustness_curves(epsilons, avg_accs_dict, avg_conf_dict)
        # Show table of values as well
        st.markdown("### Average Top-1 Accuracy Table")
        import pandas as pd
        df_acc = pd.DataFrame(avg_accs_dict, index=[f"{e:.3f}" for e in epsilons])
        df_acc.index.name = "Epsilon"
        st.dataframe(df_acc.round(2))
        st.markdown("### Average Confidence Table (0 if class changes)")
        df_conf = pd.DataFrame(avg_conf_dict, index=[f"{e:.3f}" for e in epsilons])
        df_conf.index.name = "Epsilon"
        st.dataframe(df_conf.round(2))