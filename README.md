# ANN-Classification-churn 



This project builds an Artificial Neural Network (ANN) to predict customer churn using the "Churn_Modelling.csv" dataset. It includes complete preprocessing, model training, and a simple web app interface to make real-time predictions.

---

## 📌 Project Overview

Customer churn prediction is vital for improving customer retention. Using a multi-layer ANN, this project learns patterns from customer data (like credit score, geography, age, balance, etc.) to classify whether a customer is likely to churn or stay.

---

## 📂 Project Structure

```text
.
├── Churn_Modelling.csv           # Dataset
├── app.py                        # Web app for real-time predictions
├── experiments.ipynb             # Data exploration, preprocessing, model training
├── prediction.ipynb              # Manual prediction walkthrough using saved artifacts
├── model.h5                      # Trained ANN model
├── scaler.pkl                    # StandardScaler for numeric features
├── label_encoder_gender.pkl      # LabelEncoder for 'Gender'
├── onehot_encoder_geo.pkl        # OneHotEncoder for 'Geography'
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
````

---

 ## 🔍 Dataset Description

* Source: `[Churn_Modelling.csv](https://www.kaggle.com/datasets/shubhendra7/customer-churn-prediction)`
* Size: 10,000 rows × 14 columns
* Target variable: `Exited` (1 → churned, 0 → retained)

Key features used:

* `CreditScore`
* `Geography` (France, Spain, Germany)
* `Gender`
* `Age`
* `Tenure`
* `Balance`
* `NumOfProducts`
* `HasCrCard`
* `IsActiveMember`
* `EstimatedSalary`

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:**

  * `TensorFlow / Keras` – Model development
  * `pandas, numpy` – Data processing
  * `scikit-learn` – Preprocessing, encoders
  * `joblib` – Model and encoder saving
  * `Streamlit` – Web interface

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Github-Shashwat/ANN-Classification-churn.git
cd ANN-Classification-churn
```

### 2. Install Dependencies

It’s recommended to use a virtual environment (e.g., `venv` or `conda`).

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

If you want to retrain the model:

* Open `experiments.ipynb`
* Follow the steps: EDA → preprocessing → training
* The trained model and encoders will be saved as `.h5` and `.pkl` files

### 4. Run the Web App

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to access the interface.

---

## 📈 Model Architecture

* Input layer: 11 features
* Hidden layers:

  * Dense(6) with ReLU
  * Dense(6) with ReLU
* Output layer:

  * Dense(1) with Sigmoid

Loss: `binary_crossentropy`
Optimizer: `adam`
Metric: `accuracy`

---

## 🎯 Prediction Pipeline

1. User enters inputs via the web form.
2. Categorical features (`Gender`, `Geography`) are encoded using saved `.pkl` files.
3. Numerical features are scaled using the stored `StandardScaler`.
4. The ANN model makes a prediction: churn probability and class label (0 or 1).

---

## 🔍 Example Output

```text
Input:
- Geography: France
- Gender: Female
- Age: 40
- Balance: 50000
...

Output:
- Churn Probability: 0.76
- Prediction: High Risk of Churn (1)
```

---

## 📊 Evaluation Metrics

Check `experiments.ipynb` for:

* Accuracy
* Confusion Matrix
* Precision, Recall, F1-score
* Training/Validation Loss curves

---

## ✅ Future Improvements

* Add SHAP/Explainable AI visualizations
* Deploy app using Docker or Cloud (AWS/GCP)
* Enable CSV upload for batch prediction
* Add cross-validation and hyperparameter tuning


