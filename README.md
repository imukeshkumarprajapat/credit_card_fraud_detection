# 💳 Credit Card Fraud Detection using ANN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project is a Deep Learning-based solution designed to detect fraudulent credit card transactions. Utilizing **Artificial Neural Networks (ANN)**, the model analyzes transaction patterns to distinguish between legitimate and fraudulent activities.

The system addresses the challenge of **imbalanced data** using techniques like **SMOTE** and provides a user-friendly web interface via **Streamlit** for real-time predictions.

## 🚀 Features
* **Deep Learning Model:** Built using TensorFlow/Keras with a Sequential ANN architecture.
* **Data Preprocessing:** Implemented **StandardScaler** for feature normalization.
* **Imbalance Handling:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.
* **Interactive UI:** Deployed on **Streamlit** for easy user interaction.
* **Real-time Prediction:** Classifies transactions as "Normal" or "Fraud" instantly.

## 🛠️ Technologies Used
* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Deployment:** Streamlit
* **Tools:** Jupyter Notebook / VS Code

## 🧠 Model Architecture
The Artificial Neural Network (ANN) consists of:
1.  **Input Layer:** Matches the number of features in the dataset.
2.  **Hidden Layers:** Multiple Dense layers with **ReLU** activation function to capture complex non-linear patterns.
3.  **Dropout Layers:** Added to prevent overfitting.
4.  **Output Layer:** Single neuron with **Sigmoid** activation function (binary classification: 0 or 1).
5.  **Compilation:**
    * **Optimizer:** Adam
    * **Loss Function:** Binary Crossentropy

## 📂 Dataset
The dataset used typically contains transactions made by credit cards in September 2013 by European cardholders.
* **Note:** Due to confidentiality issues, the original features are transformed using PCA (V1, V2, ... V28). Only 'Time' and 'Amount' are original features.

## ⚙️ Installation & Usage

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/credit-card-fraud-detection.git](https://github.com/imukeshkumarprajapat/credit-card-fraud-detection.git)
    cd credit-card-fraud-detection
    ```

2.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    ```bash
    app.py
    ```

## 📸 
<img width="1912" height="1046" alt="Screenshot 2025-12-26 092540" src="https://github.com/user-attachments/assets/79b7f618-9666-4460-afdb-6e769abdefae" />


*(You can add screenshots of your Streamlit app interface here)*
1. **Home Page**
2. **Prediction Result (Fraud/Normal)**

## 🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request.

## 📧 Contact
* **Author:** [mukesh kumar prajapat]
* **LinkedIn:** [https://www.linkedin.com/in/mukesh-kumar-prajapat-a51485383/]
* **Email:** [12iamdevil@gmail.com]
