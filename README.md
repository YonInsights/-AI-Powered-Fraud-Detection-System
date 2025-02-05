# 🛡️ AI-Powered Fraud Detection System

## 📌 Project Overview
Fraudulent activities in **e-commerce and banking transactions** cause massive financial losses worldwide. This project aims to build an **AI-driven fraud detection system** capable of identifying suspicious transactions using **machine learning models**. The system is designed to detect fraudulent patterns in **bank transactions** and **e-commerce purchases** using transaction data, geolocation tracking, and behavioral analysis.

✅ **Key Objectives:**
- Accurately **detect fraudulent transactions**.
- Utilize **machine learning algorithms** to improve fraud detection.
- Deploy an **API using Flask** for real-time fraud classification.
- Build a **dashboard using Dash** to visualize fraud trends.

## 📊 Dataset Information
This project uses **three datasets**:

### **1️⃣ Fraud_Data.csv** (E-commerce Transactions)
- **user_id**: Unique identifier for the user.
- **signup_time**: User account registration time.
- **purchase_time**: Timestamp when the purchase was made.
- **purchase_value**: Value of the purchase in USD.
- **device_id**: Device used to make the purchase.
- **source**: Referral source (SEO, Ads, etc.).
- **browser**: Browser used for the transaction.
- **sex**: Gender of the user.
- **age**: Age of the user.
- **ip_address**: IP address of the transaction.
- **class**: 1 → Fraudulent, 0 → Legitimate.

### **2️⃣ creditcard.csv** (Bank Transactions)
- **Time**: Seconds elapsed since first transaction.
- **V1 - V28**: PCA-transformed features.
- **Amount**: Transaction amount in USD.
- **Class**: 1 → Fraudulent, 0 → Legitimate.

### **3️⃣ IpAddress_to_Country.csv** (IP to Geolocation Mapping)
- **lower_bound_ip_address**: Start of IP range.
- **upper_bound_ip_address**: End of IP range.
- **country**: Associated country.

## 🚀 Key Features
✅ **End-to-End Machine Learning Pipeline**
✅ **Real-time Fraud Detection with Flask API**
✅ **Explainability using SHAP & LIME**
✅ **Dockerized Deployment**
✅ **Interactive Dashboard for Insights**

## 🛠️ Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- **Machine Learning Models** (Logistic Regression, Decision Trees, Random Forest, LSTM, CNN, RNN)
- **Flask** (API Development)
- **Dash** (Dashboard Visualization)
- **Docker** (Containerization)
- **MLflow** (Model Tracking & Experimentation)
- **PostgreSQL** (Data Storage)

## ⚙️ Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### **2️⃣ Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the API**
```bash
python src/api.py
```

### **5️⃣ Run the Dashboard**
```bash
python src/dashboard.py
```

## 📂 Project Structure
```
|-- project_root/
|   |-- data/
|   |   |-- raw/                  # Original datasets
|   |   |-- processed/             # Cleaned datasets
|   |-- notebooks/
|   |   |-- eda.ipynb              # Exploratory Data Analysis
|   |   |-- feature_engineering.ipynb  # Feature extraction
|   |   |-- model_training.ipynb   # Model training
|   |   |-- model_explainability.ipynb  # SHAP & LIME analysis
|   |   |-- deployment.ipynb       # API & model deployment
|   |-- src/
|   |   |-- data_processing.py     # Data cleaning
|   |   |-- feature_engineering.py # Feature extraction
|   |   |-- model.py               # Model training
|   |   |-- api.py                 # Flask API
|   |   |-- dashboard.py           # Dash visualization
|   |   |-- logging_utils.py       # Logging
|   |-- tests/
|   |   |-- test_api.py            # API testing
|   |   |-- test_model.py          # Model testing
|   |-- models/                    # Trained models
|   |-- logs/                      # Debug logs
|   |-- requirements.txt           # Dependencies
|   |-- README.md                  # Project documentation
|   |-- .gitignore                 # Ignore unnecessary files
```

## 📌 Milestones & Implementation Steps

### **🔹 Milestone 1: Data Preprocessing & Cleaning**
- Handle **missing values, duplicates, and data types**.
- Merge datasets and create new **features**.

### **🔹 Milestone 2: Exploratory Data Analysis (EDA)**
- Univariate & bivariate analysis.
- Identify fraud trends over **time, devices, locations**.

### **🔹 Milestone 3: Feature Engineering**
- Create **time-based** and **geolocation** features.
- Normalize and encode categorical variables.

### **🔹 Milestone 4: Model Training & Evaluation**
- Train models (**Logistic Regression, Random Forest, LSTM**).
- Evaluate using **Precision, Recall, F1-score, ROC curve**.

### **🔹 Milestone 5: Model Explainability**
- Use **SHAP & LIME** to understand model decisions.

### **🔹 Milestone 6: API Deployment (Flask)**
- Create a **REST API** for real-time fraud detection.

### **🔹 Milestone 7: Dockerization**
- Build and run **Docker containers** for deployment.

### **🔹 Milestone 8: Fraud Detection Dashboard**
- Use **Dash** to visualize fraud patterns dynamically.

### **🔹 Milestone 9: Final Documentation & Portfolio Submission**
- Write **detailed reports & blog posts**.
- Upload final code to **GitHub**.
## 📊 Results & Visualizations
🔹 **Fraud Trends by Time & Location**
🔹 **Feature Importance Plots (SHAP & LIME)**
🔹 **Confusion Matrix & Model Metrics**
## 🔮 Future Enhancements
🚀 Improve fraud detection with **Graph Neural Networks (GNNs)**.
🚀 Integrate **real-time fraud alerts via SMS or email**.
🚀 Expand system to detect **identity fraud & money laundering**.

## 📬 Contact Information
📌 **Yonatan Abrham**
📌 Email: [email2yonatan@gmail.com](mailto:email2yonatan@gmail.com)
📌 LinkedIn: [Yonatan Abrham](https://www.linkedin.com/in/yonatan-abrham1/)
📌 GitHub: [YonInsights](https://github.com/YonInsights)
