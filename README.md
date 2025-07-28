# 🧠 Mental Wellness Analysis and Support Strategy

## 📌 Project Summary

In today’s fast-paced tech industry, mental health challenges are widespread yet often overlooked. Many employees hesitate to seek help due to stigma, lack of resources, or poor workplace support systems. This project addresses that gap by analyzing real-world mental health survey data to **identify patterns, predict risks, and suggest data-driven solutions** for better workplace mental wellness.

**Key aims:**
- Understand which workplace and personal factors influence whether someone seeks mental health treatment.
- Predict age based on mental health and workplace attributes to uncover hidden age-based patterns or biases.
- Segment employees into distinct **mental health personas** to enable HR teams to design more personalized support strategies.

---

## ❓ The Problem

Mental health in the workplace is **complex** and influenced by multiple factors — workplace culture, openness about mental illness, availability of support, and personal background. Despite growing awareness, companies often lack **actionable insights** to tailor interventions.

Without data, policies remain generic. This project bridges that gap by providing:
- **Predictive insights** to identify employees who might need support but are not seeking it.
- **Demographic insights** to uncover whether certain age groups are more vulnerable.
- **Clustering** to group employees with similar patterns and needs, enabling **targeted interventions**.

---

## 💡 The Solution

This solution uses **machine learning** and an interactive **Streamlit app** to:
1. **Classify** whether an employee is likely to seek treatment.
2. **Predict** an employee’s age using workplace features.
3. **Cluster** employees into mental health personas for better HR planning.

---

## 🤖 Models & Why I Chose Them

### 🎯 1️⃣ Classification — *Support Vector Machine (SVM)*

- **Goal:** Predict if an employee is likely to seek mental health treatment (Yes/No).
- **Why SVM?**  
  SVM is robust for binary classification, especially when the data isn’t linearly separable. Its kernel trick handles complex relationships in survey data and helps maintain good generalization.

---

### 📏 2️⃣ Regression — *Random Forest Regressor*

- **Goal:** Predict an employee’s age based on workplace and mental health factors.
- **Why Random Forest?**  
  Mental health data can have **nonlinear relationships** and **noisy patterns**. Random Forest handles these by combining multiple decision trees, reducing overfitting and improving accuracy on unseen data.

---

### 🧩 3️⃣ Clustering — *KMeans + PCA*

- **Goal:** Group employees into mental health personas based on shared patterns.
- **Why KMeans + PCA?**  
  Clustering large, high-dimensional survey data can be messy. PCA reduces the dimensionality, making clusters clearer and more stable. KMeans is efficient for finding well-separated groups, which helps HR teams interpret employee segments.

---

## 🔍 How It Works

✔️ **Data Preprocessing:**  
- Numeric features scaled with `StandardScaler`.
- Categorical features encoded with `OneHotEncoder`.
- Pipelines ensure consistent preprocessing for training and predictions.

✔️ **Model Training:**  
- Hyperparameters tuned manually.
- Pipelines trained and saved using `joblib`.

✔️ **Evaluation:**  
- **Classification:** Confusion matrix, F1 score, ROC-AUC, ROC curve.
- **Regression:** RMSE, MAE, R², scatter plots.
- **Clustering:** Silhouette score, cluster counts, t-SNE & PCA plots.

✔️ **Deployment:**  
- All results visualized in a user-friendly **multi-page Streamlit app**.
- Users can enter new data and see live predictions.
- Cluster predictions are explained with clear persona descriptions.

---

## ⚙️ Project Structure

