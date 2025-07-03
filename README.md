
# 🔐 Robustness Evaluation Tool for Deep Learning Models

This project was developed during a Summer Internship. It implements a GUI-based tool that demonstrates, evaluates, and benchmarks the adversarial robustness of image classification models using white-box and black-box attacks.

---

## 🚀 Project Overview

Adversarial examples are specially crafted inputs that deceive machine learning models with minimal pixel changes. This tool allows users to:

- Upload or randomly sample images
- Apply adversarial attacks (FGSM, DeepFool, SimBA, Square)
- Visualize the prediction flip or confidence drop
- Evaluate pretrained models (e.g., ResNet50, EfficientNet) under adversarial stress

---

## 🎯 Key Features

- Interactive **Streamlit GUI**
- Support for **white-box and black-box attacks**
- Dataset Tabs for **MNIST, CIFAR-100, GTSRB, and ImageNet**
- Batch-based robustness evaluation using **Top-1 Accuracy** and **Confidence graphs**
- Real-time image upload, attack application, and model prediction

---

## 🛠️ Tech Stack

| Tool/Library        | Purpose                                                |
|---------------------|--------------------------------------------------------|
| Streamlit           | GUI framework                                          |
| TensorFlow / Keras  | Model inference & loading                              |
| IBM ART             | Adversarial attack implementations                     |
| Matplotlib          | Robustness graph plotting                              |
| TensorFlow Hub      | Pretrained ImageNet models                             |
| NumPy, Pandas       | Data handling and analysis                             |
| PIL (Pillow)        | Image transformation & preprocessing                   |

---

## 🧪 Supported Attacks

- **FGSM** (Fast Gradient Sign Method)
- **DeepFool**
- **SimBA** (Simple Black-box Attack)
- **Square Attack**

---

## 🖼️ Demonstration

**Example 1: ImageNet Misclassification**

> A clean image of a **school bus** was classified with **94% confidence**, but after FGSM attack, it was misclassified as an **ostrich** with **97% confidence**.

![School Bus to Ostrich](https://miro.medium.com/v2/resize:fit:1400/1*g3EnId7Urp0TtaIsyw3NTw.png)

**Example 2: Traffic Sign Confidence Drop**

> A GTSRB **stop sign** image was initially predicted with **95% confidence**. Post attack, confidence dropped to **10%**, even though it visually looked the same.

![Stop Sign Attack](https://miro.medium.com/v2/resize:fit:1400/1*T3KBNWJ7ldL_Lrp0_OSV0w.png)

---

## 📊 Robustness Evaluation Graphs

The ImageNet tab benchmarks model robustness across FGSM epsilon values using:

- **Top-1 Accuracy vs Epsilon**
- **Average Confidence vs Epsilon**

---

## 📚 References

### 📘 Academic References

1. Yu, N., et al. (2020). *AI-powered GUI attack and its defensive methods*. ACM Southeast Conference.  
2. Lin, J., et al. (2021). *ML attack models: Adversarial and data poisoning attacks*. arXiv:2112.02797  
3. Daanouni, O., et al. (2022). *NSL-MHA-CNN: A novel CNN architecture for robust diabetic retinopathy prediction*. IEEE Access  
4. Alotaibi, A., & Rassam, M. A. (2023). *Enhancing sustainability of deep-learning intrusion detection against adversarial attacks*. Sustainability  
5. IBM ART GitHub: [https://github.com/Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

### 🖼️ Image Credits

- Medium: [School bus to ostrich](https://miro.medium.com/v2/resize:fit:1400/1*g3EnId7Urp0TtaIsyw3NTw.png)  
- Medium: [Stop sign attack](https://miro.medium.com/v2/resize:fit:1400/1*T3KBNWJ7ldL_Lrp0_OSV0w.png)  
- iMerit: [Four defenses](https://imerit.net/wp-content/uploads/2022/11/Feature__Four-Defenses-Against-Adversarial-Attacks.jpg)  
- GeeksforGeeks: [Adversarial image pipeline](https://media.geeksforgeeks.org/wp-content/uploads/20240430155943/download-(39).png)  
- PapersWithCode: [CIFAR-100 samples](https://production-media.paperswithcode.com/datasets/CIFAR-100-0000000433-b71f61c0_hPEzMRg.jpg)  
- GluonCV: [MINC-2500](https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/datasets/MINC-2500.png)  
- ResearchGate: [GTSRB categories](https://www.researchgate.net/publication/359144832/figure/fig3/...png)

---
