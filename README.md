# IT Salary Prediction System: End-to-End Machine Learning Solution

## 📋 Project Overview
**IT Salary Prediction System** is a complete *End-to-End Machine Learning* solution designed to automate the process of salary estimation in the technology sector.

This project is not just a standalone analysis notebook—it is a fully functional **Microservice** that:
1. Processes raw historical data (ETL).
2. Trains and evaluates multiple classification algorithms.
3. Deploys the best-performing model via a **REST API**.
4. Monitors predictions by logging query history into a relational database.

### 🎯 Business Value
This tool supports HR departments and Technical Recruiters in rapid job offer benchmarking by classifying positions into specific salary segments:
* **High Salary** (>140k USD / year)
* **Standard Salary** (<=140k USD / year)

---

## 🛠️ Tech Stack
The project demonstrates practical proficiency with the modern Python data stack:

| Area | Technologies | Application in Project |
| :--- | :--- | :--- |
| **Backend / API** | **Flask**, JSON | Serving the model as a web service (REST API). |
| **Machine Learning** | **Scikit-learn**, Joblib | Training pipelines, Cross-validation (F1-Score, ROC AUC), Model serialization. |
| **Data Engineering** | **Pandas**, NumPy | Data cleaning, Feature Engineering, Categorical grouping. |
| **Database** | **SQLite**, SQL | Data persistence, Audit logging (history & predictions). |
| **Architecture** | **MVC Pattern** | Separation of concerns: Training logic (`train_model.py`) vs. Application logic (`app.py`). |

---

## ⚙️ Solution Architecture

The project is modularized to ensure code readability and maintainability.

### 1. Training Module (`train_model.py`)
Handles the complete Model Lifecycle:
* **Data Ingestion:** Loads data from `Zarobki_IT.csv`.
* **Preprocessing:** Handles missing values and duplicates automatically.
* **Feature Engineering:** Implements intelligent mapping of job titles (e.g., *'Principal Data Scientist'* → *'Data Scientist'*) to reduce dimensionality.
* **Model Selection:** Automatically benchmarks three algorithms:
    * *Logistic Regression*
    * *Decision Tree*
    * *Random Forest*
* **Pipeline:** Utilizes `sklearn.pipeline.Pipeline` and `ColumnTransformer` (OneHotEncoding) to prevent **data leakage** and streamline production deployment.

### 2. Application & API Module (`app.py`)
A web interface for serving real-time predictions.
* **Endpoint `/predict` (POST):** Accepts JSON input, processes it through the saved pipeline, and returns the classification with a probability score.
* **Endpoint `/history` (GET):** Allows auditing by retrieving the query history stored in SQLite.

### 3. Database (`predictions.db`)
A lightweight database storing logs: `timestamp`, `input features`, `prediction result`, and `confidence score`.

---

## 🚀 How to Run

### Prerequisites
* Python 3.8+
* Install dependencies (flask
pandas
numpy
scikit-learn
joblib):

```bash
pip install pandas numpy scikit-learn flask```

### Step 1: Train the Model

Run the ETL and training pipeline. The script automatically selects the best model based on the F1-Score and saves it as best_model_pipeline.pkl.

```bash
python train_model.py```

### Step 2: Start the API Server

Launch the Flask application. The predictions.db database will be initialized automatically upon the first run.

```bash
python app.py```

## API Documentation
### 1. Predict Salary

    **URL:** /predict
    **Method:** POST
    **Format:** JSON

*Sample Request:*
```json{
    "work_year": 2024,
    "experience_level": "SE",
    "employment_type": "FT",
    "job_title": "Machine Learning Engineer",
    "salary_currency": "USD",
    "employee_residence": "US",
    "remote_ratio": 0,
    "company_location": "US",
    "company_size": "M"
}```

*Sample Response:*

```json{
    "label": "High Salary (>140k USD)",
    "prediction": 1,
    "probability": 0.85
}```
