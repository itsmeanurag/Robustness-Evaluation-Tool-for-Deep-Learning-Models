import streamlit as st
import numpy as np
from PIL import Image
from backend.models import load_mnist_model
from backend.predict import predict_image
from backend.attacks import fgsm_attack, deepfool_attack, simba_attack, square_attack
from backend.dataset_utils import get_random_mnist_sample
from backend.class_names import MNIST_CLASSES

ATTACKS = {
    "FGSM": fgsm_attack,
    "DeepFool": deepfool_attack,
    "SimBA": simba_attack,
    "Square": square_attack,
}

def preprocess_mnist_image(img_pil):
    """Converts PIL image to (28,28,1), float32, scaled [0,1]"""
    img = img_pil.convert("L").resize((28, 28))
    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = img_arr[..., np.newaxis]
    return img_arr

def mnist_tab():
    st.header("📝 MNIST Adversarial Attacks")
    st.markdown("""
**How to use this MNIST model demo:**
- Select **'Random Sample'** to get a random MNIST test image, or **'Upload Image'** to use your own image.
- Once the image is loaded, see its **prediction** and **confidence**.
- Select an **adversarial attack** from the dropdown and click to generate an adversarial example.
- Compare the original and adversarial predictions side by side.
    """)

    @st.cache_resource
    def get_model():
        return load_mnist_model()
    model = get_model()

    st.subheader("1. Select Image Source")
    image_source = st.selectbox(
        "Choose input mode",
        ["Random Sample", "Upload Image"],
        key="mnist_source"
    )

    img_arr, true_label = None, None

    if image_source == "Random Sample":
        if st.button("Get Random MNIST Sample", key="mnist_random_btn"):
            img_arr, true_label = get_random_mnist_sample()
            st.session_state["mnist_img"] = img_arr
            st.session_state["mnist_true"] = int(true_label)
    else:
        uploaded = st.file_uploader("Upload a grayscale image (will be resized to 28x28)", type=["png", "jpg", "jpeg"], key="mnist_uploader")
        if uploaded is not None:
            img_pil = Image.open(uploaded)
            img_arr = preprocess_mnist_image(img_pil)
            st.session_state["mnist_img"] = img_arr
            st.session_state["mnist_true"] = None

    img_arr = st.session_state.get("mnist_img", None)
    true_label = st.session_state.get("mnist_true", None)

    if img_arr is not None:
        st.subheader("2. Original Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_arr.squeeze(), width=180, caption="Input Image", channels="L")
            if true_label is not None:
                st.markdown(f"**True label:** {true_label} ({MNIST_CLASSES[true_label]})")
        with col2:
            pred, conf = predict_image(model, img_arr, dataset="mnist")
            st.markdown(f"**Predicted class:** `{pred}` ({MNIST_CLASSES[pred]})")
            st.markdown(f"**Confidence:** `{conf:.4f}`")

        st.markdown("---")

        st.subheader("3. Generate Adversarial Example")
        attack_name = st.selectbox("Choose Attack", list(ATTACKS.keys()), key="mnist_attack_select")
        params = {}
        if attack_name == "FGSM":
            params["epsilon"] = st.slider("Epsilon (FGSM)", 0.01, 0.5, 0.1, 0.01, key="mnist_fgsm_eps")
        elif attack_name == "DeepFool":
            params["max_iter"] = st.slider("Max Iterations (DeepFool)", 10, 100, 50, 5, key="mnist_deepfool_iter")
            params["epsilon"] = st.number_input("Overshoot Parameter (epsilon)", value=1e-6, format="%.1e", key="mnist_deepfool_eps")
        elif attack_name == "SimBA":
            params["max_iter"] = st.slider("Iterations (SimBA)", 100, 2000, 200, 50, key="mnist_simba_iter")
            params["epsilon"] = st.slider("Epsilon (SimBA)", 0.01, 0.5, 0.2, 0.01, key="mnist_simba_eps")
        elif attack_name == "Square":
            params["epsilon"] = st.slider("Epsilon (Square Attack)", 0.01, 0.5, 0.05, 0.01, key="mnist_square_eps")
            params["max_iter"] = st.slider("Max Iterations (Square Attack)", 100, 2000, 1000, 50, key="mnist_square_iter")

        if st.button(f"Generate {attack_name} Adversarial Example", key="mnist_attack_btn"):
            attack_fn = ATTACKS[attack_name]
            if attack_name == "FGSM":
                adv_img = attack_fn(model, img_arr, epsilon=params["epsilon"], dataset="mnist")
            elif attack_name == "DeepFool":
                adv_img = attack_fn(model, img_arr, dataset="mnist", max_iter=params["max_iter"], epsilon=params["epsilon"])
            elif attack_name == "SimBA":
                adv_img = attack_fn(model, img_arr, dataset="mnist", max_iter=params["max_iter"], epsilon=params["epsilon"])
            elif attack_name == "Square":
                adv_img = attack_fn(model, img_arr, dataset="mnist", epsilon=params["epsilon"], max_iter=params["max_iter"])
            else:
                adv_img = img_arr
            st.session_state["mnist_adv_img"] = adv_img

    adv_img = st.session_state.get("mnist_adv_img", None)
    if adv_img is not None:
        st.subheader("4. Adversarial Prediction (Side-by-Side)")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_arr.squeeze(), width=180, caption="Original Image", channels="L")
            orig_pred, orig_conf = predict_image(model, img_arr, dataset="mnist")
            st.markdown(f"**Original Predicted class:** `{orig_pred}` ({MNIST_CLASSES[orig_pred]})")
            st.markdown(f"**Original Confidence:** `{orig_conf:.4f}`")
        with col2:
            st.image(adv_img.squeeze(), width=180, caption="Adversarial Image", channels="L")
            adv_pred, adv_conf = predict_image(model, adv_img, dataset="mnist")
            st.markdown(f"**Adversarial Predicted class:** `{adv_pred}` ({MNIST_CLASSES[adv_pred]})")
            st.markdown(f"**Adversarial Confidence:** `{adv_conf:.4f}`")