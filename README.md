# 🌿 Plant Seedling Classification using CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Project Overview
This project applies **Deep Learning** and **Computer Vision** to the field of **Precision Agriculture**. The goal was to build a Convolutional Neural Network (CNN) capable of distinguishing between 12 different plant species (including crops and weeds) at the seedling stage.

Automating this process allows farmers to identify weeds early and apply herbicides only where necessary, reducing environmental impact and cost.

## 🎯 Objectives
1.  **Build a CNN from scratch** to classify images into 12 categories.
2.  **Establish a Baseline** performance using standard preprocessing.
3.  **Conduct a Comparative Analysis** by introducing Data Augmentation to test model robustness vs. memorization.

---

## 🛠️ Tech Stack
*   **Core:** Python
*   **Deep Learning:** TensorFlow, Keras
*   **Data Processing:** NumPy, Pandas
*   **Visualization:** Matplotlib
*   **Environment:** Jupyter Notebook / VS Code

---

## 📊 Dataset & Preprocessing
*   **Source:** Plant Seedlings Classification Dataset
*   **Training Data:** 4,750 images
*   **Test Data:** 794 images
*   **Classes:** 12 (e.g., Maize, Wheat, Sugar beet, Black-grass, etc.)
*   **Preprocessing Pipeline:**
    *   **Resizing:** All images standardized to `128x128` pixels.
    *   **Normalization:** Pixel values scaled to `0-1` range.
    *   **Encoding:** Labels converted to categorical vectors (One-Hot Encoding).

---

## 🧠 Model Architecture
The solution uses a **Sequential CNN** designed to capture texture and shape features of leaves:

1.  **Conv2D Layers (3 blocks):** 32, 64, and 128 filters with ReLU activation.
2.  **MaxPooling2D:** To downsample and reduce computational cost.
3.  **Flatten:** Converts 2D feature maps to 1D vectors.
4.  **Dense Layers:** 128 neurons (ReLU) followed by a **Dropout (0.5)** layer to reduce overfitting.
5.  **Output Layer:** 12 neurons (Softmax) for final classification.

---

## 🔬 Results & Experimentation

I conducted a two-part experiment to evaluate model stability.

### 1. Baseline vs. Augmented Results

| Metric | Baseline Model | Augmented Model (Bonus Task) |
| :--- | :--- | :--- |
| **Strategy** | Rescaling only | Rotation, Shifts, Flips |
| **Epochs** | 15 | 15 |
| **Accuracy** | **~69.70%** | **~52.70%** |
| **Behavior** | Fast convergence | Slow, steady learning |

### 2. Analysis: Why did Accuracy Drop?
A key finding in this project was the **Regularization Effect** observed during the Bonus Task.
*   **The Baseline Model (70%)** achieved higher accuracy quickly, likely because it began "memorizing" the static orientation of the training images (Overfitting).
*   **The Augmented Model (51%)** faced a constantly changing dataset (images rotating and moving). This prevented memorization and forced the model to learn actual leaf patterns.
*   **Conclusion:** While the augmented score is lower within the limited 15-epoch run, the model is theoretically more robust for real-world scenarios where plants appear at different angles.

---

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone git@github.com:KshitijMittal/Plant-Seedling-Classification.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Notebook:**
    Open `Plant_Seedling_Classification.ipynb` in Jupyter or VS Code and run all cells.
4.  **Generate Predictions:**
    The final cell generates a `submission.csv` file containing predictions for the test set.

---

## 👨‍💻 Author
**Kshitij**
*Computer Science Student & AI Enthusiast*
Focusing on Machine Learning, Open Source, and solving real-world problems through code.

---
*Note: This project was developed as part of the NextGenX AI Club recruitment task.*
