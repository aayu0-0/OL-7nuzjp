# 🧠 Mental Wellness Analysis and Support Strategy

## 📌 Project Summary

In today’s fast-paced tech industry, mental health challenges are widespread yet often overlooked. Many employees hesitate to seek help due to stigma, lack of resources, or poor workplace support systems. This project tackles that gap by analyzing real-world mental health survey data to **identify patterns, predict risks, and suggest data-driven solutions** for better workplace mental wellness.

**Key aims:**
- Understand which workplace and personal factors influence whether someone seeks mental health treatment.
- Predict age based on mental health and workplace attributes to uncover hidden age-based patterns or biases.
- Segment employees into distinct **mental health personas** to help HR teams design more personalized support strategies.

---

## ❓ The Problem

Mental health in the workplace is **complex** — influenced by workplace culture, openness about mental illness, support availability, and personal background. Despite growing awareness, companies often lack **actionable insights** to tailor interventions.

Without data, policies stay generic. This project provides:
- **Predictive insights** to identify employees who might need support but aren’t seeking it.
- **Demographic insights** to uncover whether certain age groups are more vulnerable.
- **Clustering** to group employees with similar patterns and needs, enabling **targeted interventions**.

---

## 💡 The Solution

This project uses **machine learning** and an interactive **Streamlit app** to:
1. **Classify** whether an employee is likely to seek treatment.
2. **Predict** an employee’s age using workplace factors.
3. **Cluster** employees into mental health personas for smarter HR planning.

---

## 🤖 Models Used

### 🎯 Classification — *Support Vector Machine (SVM)*
Predicts if an employee is likely to seek mental health treatment (Yes/No).  
*SVM handles complex, non-linear data with good generalization.*

### 📏 Regression — *Random Forest Regressor*
Predicts an employee’s age from workplace and mental health features.  
*Random Forests handle non-linear, noisy patterns robustly.*

### 🧩 Clustering — *KMeans + PCA*
Groups employees into mental health personas.  
*PCA reduces dimensionality for clearer clusters. KMeans is efficient and interpretable.*

---

## 🔍 How It Works

✔️ **Preprocessing:**  
- Numeric features scaled with `StandardScaler`.  
- Categorical features encoded with `OneHotEncoder`.  
- Cleaned pipelines ensure consistent results.

✔️ **Model Training:**  
- Hyperparameters tuned manually.
- Pipelines trained and saved with `joblib`.

✔️ **Evaluation:**  
- **Classification:** Confusion matrix, F1, ROC-AUC.  
- **Regression:** RMSE, MAE, R², scatter plots.  
- **Clustering:** Silhouette score, cluster sizes, PCA scatter plots.

✔️ **Deployment:**  
- Fully interactive **multi-page Streamlit app**.
- Users can upload or input data and get predictions.
- Clusters are explained with clear persona profiles.

---

## ⚙️ Project Structure

```plaintext
OL-7nuzjp/
 ├─ streamlit/
 │   └─ app.py          # Main Streamlit app
 ├─ Models/             # Model training scripts
 ├─ Jupyter_notebook/   # Exploratory notebooks
 ├─ EDA and Visual/     # EDA scripts
 ├─ pkl_files/          # Trained model files
 ├─ cleaned_survey.csv  # Cleaned dataset
 ├─ requirements.txt    # Dependencies
 ├─ README.md           # Project overview
 └─ survey.csv          # Raw survey data


---

```## Project link

https://ol-7nuzjp-a2fxocjobq6jbzshsdutht.streamlit.app/

---
