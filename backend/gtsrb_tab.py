import streamlit as st
import numpy as np
from PIL import Image
from backend.models import load_gtsrb_model
from backend.predict import predict_image
from backend.attacks import fgsm_attack, deepfool_attack, simba_attack, square_attack
from backend.dataset_utils import get_random_gtsrb_sample
from backend.class_names import GTSRB_CLASSES

ATTACKS = {
    "FGSM": fgsm_attack,
    "DeepFool": deepfool_attack,
    "SimBA": simba_attack,
    "Square": square_attack,
}

def preprocess_gtsrb_image(img_pil):
    """Converts PIL image to (48,48,3), float32, scaled [0,1]"""
    img = img_pil.convert("RGB").resize((48, 48))
    img_arr = np.array(img).astype("float32") / 255.0
    return img_arr

def gtsrb_tab():
    st.header("🚸 GTSRB Adversarial Attacks")
    st.markdown("""
**How to use this GTSRB model demo:**
- Select **'Random Sample'** to get a random GTSRB test image, or **'Upload Image'** to use your own image.
- Once the image is loaded, see its **prediction** and **confidence**.
- Select an **adversarial attack** from the dropdown and click to generate an adversarial example.
- Compare the original and adversarial predictions side by side.
    """)

    @st.cache_resource
    def get_model():
        return load_gtsrb_model()
    model = get_model()

    st.subheader("1. Select Image Source")
    image_source = st.selectbox(
        "Choose input mode",
        ["Random Sample", "Upload Image"],
        key="gtsrb_source"
    )

    img_arr, true_label = None, None

    if image_source == "Random Sample":
        if st.button("Get Random GTSRB Sample", key="gtsrb_random_btn"):
            img_arr, true_label = get_random_gtsrb_sample()
            st.session_state["gtsrb_img"] = img_arr
            st.session_state["gtsrb_true"] = int(true_label)
    else:
        uploaded = st.file_uploader("Upload a color image (will be resized to 48x48)", type=["png", "jpg", "jpeg"], key="gtsrb_uploader")
        if uploaded is not None:
            img_pil = Image.open(uploaded)
            img_arr = preprocess_gtsrb_image(img_pil)
            st.session_state["gtsrb_img"] = img_arr
            st.session_state["gtsrb_true"] = None

    img_arr = st.session_state.get("gtsrb_img", None)
    true_label = st.session_state.get("gtsrb_true", None)

    if img_arr is not None:
        st.subheader("2. Original Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_arr, width=180, caption="Input Image", channels="RGB")
            if true_label is not None:
                st.markdown(f"**True label:** {true_label} ({GTSRB_CLASSES[true_label]})")
        with col2:
            pred, conf = predict_image(model, img_arr, dataset="gtsrb")
            st.markdown(f"**Predicted class:** `{pred}` ({GTSRB_CLASSES[pred]})")
            st.markdown(f"**Confidence:** `{conf:.4f}`")

        st.markdown("---")

        st.subheader("3. Generate Adversarial Example")
        attack_name = st.selectbox("Choose Attack", list(ATTACKS.keys()), key="gtsrb_attack_select")
        params = {}
        if attack_name == "FGSM":
            params["epsilon"] = st.slider("Epsilon (FGSM)", 0.01, 0.5, 0.1, 0.01, key="gtsrb_fgsm_eps")
        elif attack_name == "DeepFool":
            params["max_iter"] = st.slider("Max Iterations (DeepFool)", 10, 100, 50, 5, key="gtsrb_deepfool_iter")
            params["epsilon"] = st.number_input("Overshoot Parameter (epsilon)", value=1e-6, format="%.1e", key="gtsrb_deepfool_eps")
        elif attack_name == "SimBA":
            params["max_iter"] = st.slider("Iterations (SimBA)", 100, 2000, 200, 50, key="gtsrb_simba_iter")
            params["epsilon"] = st.slider("Epsilon (SimBA)", 0.01, 0.5, 0.2, 0.01, key="gtsrb_simba_eps")
        elif attack_name == "Square":
            params["epsilon"] = st.slider("Epsilon (Square Attack)", 0.01, 0.5, 0.05, 0.01, key="gtsrb_square_eps")
            params["max_iter"] = st.slider("Max Iterations (Square Attack)", 100, 2000, 1000, 50, key="gtsrb_square_iter")

        if st.button(f"Generate {attack_name} Adversarial Example", key="gtsrb_attack_btn"):
            attack_fn = ATTACKS[attack_name]
            if attack_name == "FGSM":
                adv_img = attack_fn(model, img_arr, epsilon=params["epsilon"], dataset="gtsrb")
            elif attack_name == "DeepFool":
                adv_img = attack_fn(model, img_arr, dataset="gtsrb", max_iter=params["max_iter"], epsilon=params["epsilon"])
            elif attack_name == "SimBA":
                adv_img = attack_fn(model, img_arr, dataset="gtsrb", max_iter=params["max_iter"], epsilon=params["epsilon"])
            elif attack_name == "Square":
                adv_img = attack_fn(model, img_arr, dataset="gtsrb", epsilon=params["epsilon"], max_iter=params["max_iter"])
            else:
                adv_img = img_arr
            st.session_state["gtsrb_adv_img"] = adv_img

    adv_img = st.session_state.get("gtsrb_adv_img", None)
    if adv_img is not None:
        st.subheader("4. Adversarial Prediction (Side-by-Side)")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_arr, width=180, caption="Original Image", channels="RGB")
            orig_pred, orig_conf = predict_image(model, img_arr, dataset="gtsrb")
            st.markdown(f"**Original Predicted class:** `{orig_pred}` ({GTSRB_CLASSES[orig_pred]})")
            st.markdown(f"**Original Confidence:** `{orig_conf:.4f}`")
        with col2:
            st.image(adv_img, width=180, caption="Adversarial Image", channels="RGB")
            adv_pred, adv_conf = predict_image(model, adv_img, dataset="gtsrb")
            st.markdown(f"**Adversarial Predicted class:** `{adv_pred}` ({GTSRB_CLASSES[adv_pred]})")
            st.markdown(f"**Adversarial Confidence:** `{adv_conf:.4f}`")