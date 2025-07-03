import streamlit as st
from backend.mnist_tab import mnist_tab 
from backend.cifar100_tab import cifar100_tab
from backend.gtsrb_tab import gtsrb_tab
from backend.imagenet_tab import imagenet_tab
from backend.imagenet_robustness_tab import plot_robustness_curve_tab

st.set_page_config(page_title="Robustness Evaluation Tool for Deep Learning Models", layout="wide")

IMAGE_WIDTH = 300  # px

st.title("Robustness Evaluation Tool for Deep Learning Models")

tabs = st.tabs([
    "HOME 🏠 ",
    "ABOUT AdvML 📚 ",
    "MNIST 📝 ",
    "CIFAR100 🖼️ ",
    "GTSRB 🚸 ",
    "IMAGENET PREDICTOR 📷 ",
    "ROBUSTNESS EVALUATION TOOL (IMAGENET MODELS) 🛡️"

])

with tabs[0]:
    st.markdown("""
# 👋 Welcome!

Welcome to this interactive and educational platform designed to **explore, understand, and demonstrate the fascinating field of Adversarial Machine Learning (AML)** through a user-friendly GUI.

This tool allows users to apply and visualize the impact of various adversarial attack techniques on image classification models. These models are trained on widely-used benchmark datasets such as:

- 🖊️ **MNIST** (Handwritten Digits)
- 🌄 **CIFAR-100** (100-Class Natural Images)
- 🚦 **GTSRB** (German Traffic Sign Recognition Benchmark)
- 🖼️ **ImageNet** (Large-scale Natural Images using Pretrained Models)

Through this platform, users can:
- Gain a hands-on understanding of how small, imperceptible changes in input data can lead to significant misclassifications in deep neural networks.
- Evaluate the robustness and vulnerability of different ML models under targeted and untargeted adversarial conditions.
- Observe side-by-side comparisons of original and adversarial samples, along with the corresponding model predictions and confidence scores.

This platform offers an **intuitive and modular interface to explore the real-world implications of adversarial robustness in AI systems.**

---

## 🏛️ Academic & Institutional Context

""")


    st.markdown("""
---

## 🧠 Project Highlights

This project presents a comprehensive and hands-on exploration of Adversarial Machine Learning through an interactive, GUI-driven platform. Below are the key technical and educational accomplishments:

### ✅ Intuitive Web-Based Interface using Streamlit
- Developed a clean, user-friendly GUI using Streamlit, enabling users to experiment with adversarial attacks without writing code.
- Modular design with interactive tabs and real-time feedback for smooth user experience.

### ✅ Training and Deployment of Custom ML Models
Built and trained deep learning models on popular benchmark datasets:
- **MNIST** – Handwritten digits recognition
- **CIFAR-100** – 100-class natural image classification
- **GTSRB** – Traffic sign recognition for autonomous driving systems

### ✅ Integration of Pretrained ImageNet Models
Incorporated state-of-the-art models trained on the ImageNet dataset to analyze robustness at scale:
- 🧠 **ResNet50**
- 🔍 **InceptionV3**
- 📱 **MobileNetV2**
- ⚙️ **EfficientNet B0 & B1**
- 🧬 **DenseNet121**

These models were accessed via Torchvision or TensorFlow Hub, ensuring high accuracy baselines for adversarial evaluation.

### ✅ Implementation of Multiple Adversarial Attacks with IBM ART
Utilized the Adversarial Robustness Toolbox (ART) by IBM to generate adversarial examples.

Implemented and visualized the effects of various attack strategies:
- **FGSM (Fast Gradient Sign Method)** – one-step gradient-based
- **DeepFool** – iterative, minimal perturbation
- **SimBA** – black-box, random direction sampling
- **Square Attack** – query-efficient score-based black-box attack

### ✅ Robust Visualization and Comparative Analysis
Showcased side-by-side comparison of original vs. adversarial images, highlighting:
- Change in class label
- Drop or shift in confidence score
- Human-imperceptible perturbations leading to drastic misclassifications

### ✅ Live User-Controlled Testing and Exploration
Empowered users to:
- Select dataset and model
- Choose attack type and tweak parameters (e.g., epsilon, iterations)
- Instantly observe how models behave under attack conditions
                
### ✅ 📈 Robustness Evaluation Tool
- Upload or sample multiple images  
- Automatically test top ImageNet models with varying FGSM strengths  
- Plot and compare **Top-1 Accuracy** and **Confidence** curves  
- Gain insight into model vulnerabilities under attack  

This fosters a deep, experimental understanding of AI vulnerabilities and adversarial robustness.
    """)

with tabs[1]:
    st.header("📚 Adversarial ML Background")
    info_tab = st.selectbox(
        "Select Section",
        [
            "Pre-requisite Knowledge",
            "About Datasets",
            "ART Documentation",
            "Adversarial Attacks Explained"
        ]
    )

    if info_tab == "Pre-requisite Knowledge":
        st.header("Adversarial Machine Learning & Attack Types")
        st.markdown("""
**Adversarial Machine Learning (AdvML)** is a specialized area within machine learning that studies the vulnerabilities of machine learning models to intentionally crafted inputs called **adversarial examples**.

These adversarial examples are inputs—such as images, audio, or text—that have been subtly and strategically manipulated to cause a machine learning model to make mistakes, often while appearing unchanged to human observers.

**Why does this happen?**

Machine learning models, especially deep neural networks, learn complex patterns from data. However, they can be sensitive to small, imperceptible changes in their input. Attackers exploit this by introducing slight perturbations that can lead the model to misclassify the input or make incorrect predictions.

**Importance of Adversarial ML:**
- Adversarial examples reveal fundamental weaknesses in AI systems.
- They highlight the need for robust and secure ML models, especially in safety-critical applications.
- Understanding adversarial ML helps researchers build defenses and improve model reliability.
        """)

        st.image("https://miro.medium.com/v2/resize:fit:1400/1*g3EnId7Urp0TtaIsyw3NTw.png", caption="How adversarial examples can mislead ML models", use_container_width= 500)

        st.markdown("""
---
### Real-life Impact Example: Self-Driving Cars

Consider an autonomous vehicle using deep learning to recognize traffic signs.

Attackers can place tiny stickers or modify a stop sign in ways that are invisible to human drivers but fool the car's AI into misreading the sign as a speed limit or yield sign. This could cause the vehicle to ignore stop signs, resulting in dangerous or even life-threatening situations.

Such adversarial attacks pose significant risks in real-world deployments of AI, emphasizing the urgent need for robust adversarial defenses.

**In summary:**

Adversarial ML is not just about breaking models, but about understanding their limitations so we can build safer, more trustworthy AI systems.
        """)

        st.image("https://miro.medium.com/v2/resize:fit:1400/1*T3KBNWJ7ldL_Lrp0_OSV0w.png", caption="Fooling the Eye of AI: The School Bus That Became an Ostrich", use_container_width= 500)
        st.markdown("""
A neural network correctly identifies a clean image of a school bus with 94% confidence.
By adding a tiny, carefully crafted noise (adversarial perturbation) to the image, it still looks like a bus to humans.

However, the altered image fools the model into predicting "ostrich" with 97% confidence.
This small change shifts the input just enough to cross the model’s decision boundary.

Adversarial attacks expose a critical weakness in AI models — they can be easily tricked by nearly invisible changes.
It highlights the need for stronger, more robust AI defenses.
        """)

        st.image("https://imerit.net/wp-content/uploads/2022/11/Feature__Four-Defenses-Against-Adversarial-Attacks.jpg", caption="Defenses against adversarial attacks", use_container_width= 500)

        st.header("Types of Adversarial Attacks")

        st.markdown("""
Adversarial attacks are generally categorized based on the attacker's knowledge of the model:

---
**1. White-box Attacks**

In white-box attacks, the attacker has complete access to the model, including its architecture, parameters, gradients, and sometimes even the training data. This allows them to craft highly effective adversarial examples by directly computing how to change input data to maximize the model's prediction error.

**Key Points:**
- Full transparency into the model
- Attackers can use gradient information to design precise perturbations
- Examples: FGSM (Fast Gradient Sign Method), PGD (Projected Gradient Descent), Carlini & Wagner (C&W) attack

**Use Case:**
- Research settings, model debugging, and robust defense evaluations

White-box attacks represent a "worst-case scenario" for model robustness and are often used to benchmark the strength of defenses.

---
**2. Black-box Attacks**

In black-box attacks, the attacker has no internal knowledge of the model. They can only observe input-output behavior—i.e., they can query the model and see its predictions, but do not know its parameters or architecture.

**Key Points:**
- No access to gradients or model internals
- Attackers often create adversarial examples on a surrogate model and transfer them to the target (transferability)
- Can use techniques like query-based optimization or evolutionary algorithms

**Use Case:**
- Real-world scenarios where models are deployed as APIs or cloud services

Black-box attacks are more reflective of practical threats, as most production ML systems do not expose their internals.

---
**Both types of attacks are critical for evaluating the security of AI systems.**

Defending against white-box attacks usually offers stronger guarantees, while black-box attacks simulate more realistic threat models.
        """)

    elif info_tab == "About Datasets":
        st.header("About Datasets Used")
        st.markdown("This project demonstrates adversarial attacks on several widely used datasets:")

        # MNIST
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            st.subheader("MNIST")
            st.markdown("""
- Contains 70,000 handwritten digit images (0-9)
- Each image is 28x28 pixels, grayscale
- Simple and clean dataset, ideal for quick model prototyping
- Used extensively for introductory image classification and adversarial example research
- Commonly serves as a "hello world" for deep learning and adversarial ML
            """)
        with col2:
            st.markdown(
                f"""<div style="display:flex; justify-content:center;">
                <img src="https://media.geeksforgeeks.org/wp-content/uploads/20240430155943/download-(39).png"
                alt="MNIST Sample Images" width="{IMAGE_WIDTH}">
                <div style="text-align:center; font-size: 0.95em; color: #666;">MNIST Sample Images</div>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # CIFAR-100
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            st.subheader("CIFAR-100")
            st.markdown("""
- 60,000 color images (32x32 pixels)
- 100 fine-grained classes, each with 600 images
- Diverse categories: animals, objects, vehicles, etc.
- More challenging than MNIST due to color and variability
- Widely used to benchmark adversarial robustness in small-scale vision tasks
            """)
        with col2:
            st.markdown(
                f"""<div style="display:flex; justify-content:center;">
                <img src="https://production-media.paperswithcode.com/datasets/CIFAR-100-0000000433-b71f61c0_hPEzMRg.jpg"
                alt="CIFAR-100 Sample Images" width="{IMAGE_WIDTH}">
                <div style="text-align:center; font-size: 0.95em; color: #666;">CIFAR-100 Sample Images</div>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ImageNet
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            st.subheader("ImageNet")
            st.markdown("""
- Large-scale dataset with over 1 million high-resolution images
- 1,000 object classes (e.g., animals, vehicles, household items)
- Gold standard for deep learning benchmarking
- Used for training and evaluating state-of-the-art neural networks
- Major focus for adversarial attacks and defenses due to its complexity and real-world relevance
            """)
        with col2:
            st.markdown(
                f"""<div style="display:flex; justify-content:center;">
                <img src="https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/datasets/MINC-2500.png"
                alt="ImageNet Sample Images" width="{IMAGE_WIDTH}">
                <div style="text-align:center; font-size: 0.95em; color: #666;">ImageNet Sample Images</div>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # GTSRB
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            st.subheader("GTSRB (German Traffic Sign Recognition Benchmark)")
            st.markdown("""
- Contains over 50,000 images of German road signs(48x48 pixels)
- 43 different classes (e.g., stop, speed limit, warning signs)
- Images have variable sizes and lighting conditions, simulating real-world challenges
- Critical for research in autonomous driving and adversarial robustness in safety-critical systems
- Demonstrates how minor alterations can affect traffic sign recognition
            """)
        with col2:
            st.markdown(
                f"""<div style="display:flex; justify-content:center;">
                <img src="https://www.researchgate.net/publication/359144832/figure/fig3/AS:1137083763103744@1649326398565/Samples-of-different-categories-in-GTSRB-image-dataset-50.png"
                alt="GTSRB Sample Images" width="{IMAGE_WIDTH}">
                <div style="text-align:center; font-size: 0.95em; color: #666;">GTSRB Sample Images</div>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("""
These datasets allow us to evaluate how adversarial attacks impact different model and data types, from simple digits to complex, real-world scenarios.
        """)

    elif info_tab == "ART Documentation":
        st.header("Adversarial Robustness Toolbox (ART) Documentation")
        st.markdown("""
The **Adversarial Robustness Toolbox (ART)** is an open-source library by IBM for adversarial ML research and defense.

**Key Features:**
- Numerous attack implementations (white-box and black-box)
- Evaluation and defense tools for ML models
- Supports TensorFlow, PyTorch, scikit-learn, Keras, and more

**Example Attacks in ART:**
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- Carlini & Wagner (C&W)
- DeepFool
- Boundary Attack

**Documentation:**  
- [ART GitHub Repository](https://github.com/Trusted-AI/adversarial-robustness-toolbox)  
- [Official Docs](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/)

**Example Usage:**
```python
from art.attacks.evasion import FastGradientMethod
attack = FastGradientMethod(estimator=model, eps=0.2)
x_adv = attack.generate(x=x_test)
   """)
        
    elif info_tab == "Adversarial Attacks Explained":
        st.header("Adversarial Attacks: Simple Explanations")

        st.subheader("FGSM (Fast Gradient Sign Method)")
        st.markdown("""
- **FGSM** is a quick way to trick a neural network. It makes tiny changes to an image so the model makes a wrong prediction.
- It looks at which direction to change each pixel to make the model more confused, and then tweaks all pixels just a little.
- **Think of it as:** Nudging the image just enough to fool the model, but so small a human barely notices.
- **In practice:** You pick how strong the nudge is (epsilon). Bigger nudges = more likely to fool the model, but also more noticeable.
    """)

        st.subheader("DeepFool")
        st.markdown("""
- **DeepFool** is a smarter, step-by-step attack. It keeps making tiny changes to the image until the model finally changes its mind.
- It tries to find the smallest possible tweak so the image still looks the same to us, but the model is tricked.
- **Think of it as:** Gently and carefully pushing the image over the edge of the model's decision boundary.
    """)

        st.subheader("SimBA (Simple Black-box Attack)")
        st.markdown("""
- **SimBA** is a "black-box" attack, meaning it doesn't need to know how the model works inside.
- It randomly changes one part of the image at a time and checks if the model’s confidence drops.
- It keeps only the changes that make the model less sure or wrong.
- **Think of it as:** Randomly poking the image and keeping the pokes that confuse the model.
    """)

        st.subheader("Square Attack")
        st.markdown("""
- **Square Attack** is another black-box attack. Instead of changing one pixel, it changes random square patches in the image.
- It keeps doing this, changing different squares, until the model is fooled.
- **Think of it as:** Putting tiny stickers (squares) on the image in random spots to mess with the model’s prediction.
    """)

        st.markdown("""
---
**Summary Table**

| Attack   | Needs Model Details? | How It Changes Image | Goal                  |
|----------|---------------------|---------------------|-----------------------|
| FGSM     | Yes (white-box)     | Tiny change all over| Quick, simple fooling |
| DeepFool | Yes (white-box)     | Smallest possible   | Minimal change to fool|
| SimBA    | No (black-box)      | Random pixels       | Fool by poking        |
| Square   | No (black-box)      | Random squares      | Fool by patching      |
    """)
with tabs[2]:
    mnist_tab()
with tabs[3]:
    cifar100_tab()
with tabs[4]:
    gtsrb_tab()
with tabs[5]:
    imagenet_tab()
with tabs[6]:
    plot_robustness_curve_tab()
