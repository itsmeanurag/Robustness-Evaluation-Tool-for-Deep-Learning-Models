import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import (
    resnet50, mobilenet_v2, inception_v3, efficientnet, densenet
)
from backend.attacks import fgsm_attack, deepfool_attack, simba_attack, square_attack
from backend.dataset_utils import get_random_imagenet_sample

# Model registry and input size for each model
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

PREPROCESS_NOTES = {
    "ResNet50": "ResNet50’s preprocess_input subtracts [103.939, 116.779, 123.68] from BGR channels, which when 'inverted' for display as RGB looks okay.",
    "MobileNetV2": "MobileNetV2 scales input to [-1, 1], so displaying the preprocessed image directly will look washed out/contrasty.",
    "EfficientNetB0": "EfficientNet scales input to [-1, 1], so displaying the preprocessed image directly will look washed out/contrasty.",
    "EfficientNetB1": "EfficientNet scales input to [-1, 1], so displaying the preprocessed image directly will look washed out/contrasty.",
    "DenseNet121": "DenseNet121 uses zero-centering with mean subtraction (similar to ResNet but may differ slightly).",
    "InceptionV3": "InceptionV3 also maps input to [-1, 1], so direct display looks washed out."
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

def imagenet_tab():
    st.header("📷 IMAGENET Adversarial Attacks")

    st.markdown("""
**How to use this ImageNet model demo:**
1. Select **'Random Sample'** or **'Upload Image'**.
2. Select the model you want to use for prediction.
3. Click **Predict** to see the predictions.
4. Select an **adversarial attack** and generate an adversarial example.
5. Compare original and adversarial predictions.
    """)

    # Step 1: Image selection
    st.subheader("1. Select Image Source")
    image_source = st.selectbox(
        "Choose input mode",
        ["Random Sample", "Upload Image"],
        key="imagenet_source"
    )

    input_img_pil = None
    if image_source == "Random Sample":
        if st.button("Get Random ImageNet Sample", key="imagenet_random_btn"):
            img_arr_raw, _ = get_random_imagenet_sample()
            # Convert to PIL for display
            img_pil = Image.fromarray(np.uint8(np.clip(img_arr_raw, 0, 255))).convert("RGB")
            st.session_state["imagenet_pil"] = img_pil
    else:
        uploaded = st.file_uploader("Upload an image (will be resized automatically)", type=["png", "jpg", "jpeg"], key="imagenet_uploader")
        if uploaded is not None:
            img_pil = Image.open(uploaded).convert("RGB")
            st.session_state["imagenet_pil"] = img_pil

    input_img_pil = st.session_state.get("imagenet_pil", None)

    # Step 2: Model selection
    st.subheader("2. Select Model")
    model_name = st.selectbox(
        "Select a pretrained model",
        list(MODEL_DICT.keys()),
        index=0,
        key="imagenet_model_select"
    )
    input_size = MODEL_INPUT_SIZES[model_name]

    # Step 3: Predict button
    st.subheader("3. Predict")
    if input_img_pil is not None:
        st.image(input_img_pil.resize(input_size), width=input_size[0], caption="Selected Image")
        st.info(PREPROCESS_NOTES.get(model_name, ""))
        if st.button("Predict", key="imagenet_predict_btn"):
            # Preprocess image for model
            img_pil_resized = input_img_pil.resize(input_size)
            img_arr = preprocess_image(img_pil_resized, model_name)
            st.session_state["imagenet_img_arr"] = img_arr
            st.session_state["imagenet_model"] = model_name
            st.session_state["imagenet_predicted"] = True
    else:
        st.info("Please upload an image or select a random sample.")

    # Step 4: Show predictions if available
    img_arr = st.session_state.get("imagenet_img_arr", None)
    predicted_flag = st.session_state.get("imagenet_predicted", False)
    if img_arr is not None and predicted_flag:
        model = load_model(model_name)
        st.markdown("#### Top-3 Predictions")
        preds = model.predict(img_arr[np.newaxis, ...])
        decode_func = MODEL_DICT[model_name].decode_predictions
        decoded = decode_func(preds, top=3)[0]
        for pred_class, name, conf in decoded:
            st.markdown(f"**{name}** (`{pred_class}`): `{conf:.4f}`")
        st.markdown("---")

        # Step 5: Adversarial attack
        st.subheader("4. Generate Adversarial Example")
        attack_name = st.selectbox("Choose Attack", ["FGSM", "DeepFool", "SimBA", "Square"], key="imagenet_attack_select")
        params = {}
        if attack_name == "FGSM":
            params["epsilon"] = st.slider("Epsilon (FGSM)", 0.001, 0.5, 0.01, 0.001, key="imagenet_fgsm_eps")
        elif attack_name == "DeepFool":
            params["max_iter"] = st.slider("Max Iterations (DeepFool)", 10, 100, 30, 5, key="imagenet_deepfool_iter")
            params["epsilon"] = st.number_input("Overshoot Parameter (epsilon)", value=1e-6, format="%.1e", key="imagenet_deepfool_eps")
        elif attack_name == "SimBA":
            params["max_iter"] = st.slider("Iterations (SimBA)", 100, 2000, 500, 50, key="imagenet_simba_iter")
            params["epsilon"] = st.slider("Epsilon (SimBA)", 0.001, 0.5, 0.01, 0.001, key="imagenet_simba_eps")
        elif attack_name == "Square":
            params["epsilon"] = st.slider("Epsilon (Square Attack)", 0.001, 0.5, 0.01, 0.001, key="imagenet_square_eps")
            params["max_iter"] = st.slider("Max Iterations (Square Attack)", 100, 2000, 1000, 50, key="imagenet_square_iter")

        if st.button(f"Generate {attack_name} Adversarial Example", key="imagenet_attack_btn"):
            attack_fn = {
                "FGSM": fgsm_attack,
                "DeepFool": deepfool_attack,
                "SimBA": simba_attack,
                "Square": square_attack,
            }[attack_name]
            if attack_name == "FGSM":
                adv_img = attack_fn(model, img_arr, epsilon=params["epsilon"], dataset="imagenet")
            elif attack_name == "DeepFool":
                adv_img = attack_fn(model, img_arr, dataset="imagenet", max_iter=params["max_iter"], epsilon=params["epsilon"])
            elif attack_name == "SimBA":
                adv_img = attack_fn(model, img_arr, dataset="imagenet", max_iter=params["max_iter"], epsilon=params["epsilon"])
            elif attack_name == "Square":
                adv_img = attack_fn(model, img_arr, dataset="imagenet", epsilon=params["epsilon"], max_iter=params["max_iter"])
            else:
                adv_img = img_arr
            st.session_state["imagenet_adv_img"] = adv_img
            st.session_state["imagenet_adv_predicted"] = True

    # Step 6: Show original and adversarial side-by-side
    adv_img = st.session_state.get("imagenet_adv_img", None)
    adv_predicted_flag = st.session_state.get("imagenet_adv_predicted", False)
    if adv_img is not None and adv_predicted_flag and img_arr is not None and predicted_flag:
        st.markdown("---")
        st.subheader("5. Original vs Adversarial Predictions")
        col1, col2 = st.columns(2)
        with col1:
            st.image(input_img_pil.resize(input_size), width=input_size[0], caption="Original Image")
            model = load_model(model_name)
            preds = model.predict(img_arr[np.newaxis, ...])
            decode_func = MODEL_DICT[model_name].decode_predictions
            decoded = decode_func(preds, top=3)[0]
            st.markdown("**Original Predictions:**")
            for pred_class, name, conf in decoded:
                st.markdown(f"**{name}** (`{pred_class}`): `{conf:.4f}`")
        with col2:
            # For display, convert adv_img to [0,255] for visualization
            adv_img_disp = adv_img.copy()
            # Only for visualization! Not an exact inversion.
            adv_img_disp = np.uint8(np.clip((adv_img_disp - adv_img_disp.min()) / (adv_img_disp.max() - adv_img_disp.min()) * 255, 0, 255))
            adv_pil = Image.fromarray(adv_img_disp).resize(input_size)
            st.image(adv_pil, width=input_size[0], caption="Adversarial Image")
            preds_adv = model.predict(adv_img[np.newaxis, ...])
            decoded_adv = decode_func(preds_adv, top=3)[0]
            st.markdown("**Adversarial Predictions:**")
            for pred_class, name, conf in decoded_adv:
                st.markdown(f"**{name}** (`{pred_class}`): `{conf:.4f}`")