import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import (
    resnet50, mobilenet_v2, inception_v3, efficientnet, densenet
)
from backend.attacks import fgsm_attack, deepfool_attack, simba_attack, square_attack

# --- Model registry: (keras_module, input_size) ---
MODEL_DICT = {
    "ResNet50": (resnet50, (224,224)),
    "MobileNetV2": (mobilenet_v2, (224,224)),
    "InceptionV3": (inception_v3, (299,299)),
    "EfficientNetB0": (efficientnet, (224,224)),
    "EfficientNetB1": (efficientnet, (240,240)),
    "DenseNet121": (densenet, (224,224)),
}
ATTACKS = {
    "FGSM": fgsm_attack,
    "DeepFool": deepfool_attack,
    "SimBA": simba_attack,
    "Square": square_attack,
}
# Reasonable max_iters for iterative attacks
MAX_ITERS = {
    "DeepFool": 30,
    "SimBA": 1000,
    "Square": 1000,
}

def preprocess_image(img_pil, model_name):
    model_lib, input_size = MODEL_DICT[model_name]
    img = img_pil.convert("RGB").resize(input_size)
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

def l2_norm(x):
    return np.linalg.norm(x.flatten())

def mean_abs_diff(x):
    return np.mean(np.abs(x.flatten()))

def show_perturbation_image(perturb, input_size, caption, scale=10):
    # For visualization: scale up and center to [0,255]
    pert = np.clip((perturb * scale - perturb.min() * scale) + 128, 0, 255).astype(np.uint8)
    pert_img = Image.fromarray(pert).resize(input_size)
    st.image(pert_img, caption=caption, width=128)

def imagenet_compare_tab():
    st.header("🗂️ ImageNet Model Comparison: Attacks & Perturbation")

    st.markdown("""
**Upload a single image. All models will predict on it. Then, for epsilon=0.2, each model will be attacked (FGSM, DeepFool, SimBA, Square).  
For each attack, the table shows the original prediction, adversarial prediction, whether prediction changed, and the perturbation norm.  
Optionally, view the perturbation image and a summary plot of attack success rate.**
    """)

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="compare_uploader")
    if uploaded is None:
        st.info("Please upload an image.")
        return

    img_pil = Image.open(uploaded).convert("RGB")
    st.image(img_pil, caption="Uploaded image", width=224)
    epsilon = 0.2

    results = []
    perturb_images = {}

    # Gather results for all models and attacks
    for model_name in MODEL_DICT:
        model_lib, input_size = MODEL_DICT[model_name]
        model = load_model(model_name)
        arr = preprocess_image(img_pil, model_name)
        preds = model.predict(arr[np.newaxis, ...], verbose=0)
        decode_func = model_lib.decode_predictions
        orig_top = decode_func(preds, top=1)[0][0]
        orig_label = orig_top[1]
        orig_conf = orig_top[2]
        orig_class = orig_top[0]

        for attack_name, attack_fn in ATTACKS.items():
            if attack_name == "FGSM":
                adv_arr = attack_fn(model, arr, epsilon=epsilon, dataset="imagenet")
            elif attack_name == "DeepFool":
                adv_arr = attack_fn(model, arr, dataset="imagenet", max_iter=MAX_ITERS["DeepFool"], epsilon=epsilon)
            elif attack_name == "SimBA":
                adv_arr = attack_fn(model, arr, dataset="imagenet", max_iter=MAX_ITERS["SimBA"], epsilon=epsilon)
            elif attack_name == "Square":
                adv_arr = square_attack(model, arr, dataset="imagenet", epsilon=epsilon, max_iter=MAX_ITERS["Square"])
            else:
                adv_arr = arr

            adv_preds = model.predict(adv_arr[np.newaxis, ...], verbose=0)
            adv_top = decode_func(adv_preds, top=1)[0][0]
            adv_label = adv_top[1]
            adv_conf = adv_top[2]
            adv_class = adv_top[0]
            success = adv_class != orig_class

            # Perturbation norm
            perturb = adv_arr - arr
            norm = l2_norm(perturb)
            mad = mean_abs_diff(perturb)
            results.append({
                "Model": model_name,
                "Attack": attack_name,
                "Original": f"{orig_label} ({orig_conf:.2f})",
                "Adversarial": f"{adv_label} ({adv_conf:.2f})",
                "Changed": "✅" if success else "❌",
                "‖Perturbation‖₂": f"{norm:.2f}",
                "|Perturbation|̄": f"{mad:.4f}",
            })
            # Store perturbation for visualization
            perturb_images[(model_name, attack_name)] = (perturb, input_size)

    # --- Table ---
    st.markdown("### Results Table")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    # --- Perturbation Visuals ---
    st.markdown("### Visualize Perturbations")
    with st.expander("Show perturbation images for each model/attack"):
        for (model_name, attack_name), (perturb, input_size) in perturb_images.items():
            st.markdown(f"**{model_name} — {attack_name}**")
            show_perturbation_image(perturb, input_size, caption=f"Perturbation: {model_name} - {attack_name}", scale=10)

    # --- Attack Success Summary ---
    st.markdown("### Attack Success Summary")
    summary = (
        df.groupby(["Model", "Attack"])["Changed"]
        .apply(lambda ser: sum([x == "✅" for x in ser]) / len(ser))
        .unstack()
        .fillna(0)
    )
    # For this UI, for a single image, each cell is 0 or 1, but for batch, it would be a %.
    st.dataframe(summary.applymap(lambda x: f"{x*100:.0f}%" if x <= 1 else f"{x:.0f}"), use_container_width=True)

    # --- Bar chart: success per model/attack ---
    fig, ax = plt.subplots(figsize=(8,5))
    summary_plot = summary * 100
    summary_plot.plot(kind="bar", ax=ax)
    plt.ylabel("Attack Success Rate (%)")
    plt.title("Attack Success Rate per Model")
    plt.xticks(rotation=45)
    st.pyplot(fig)