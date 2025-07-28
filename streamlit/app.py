import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay,
    mean_squared_error, mean_absolute_error, r2_score, silhouette_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, '..', 'pkl_files')

rf_pipeline = joblib.load(os.path.join(PKL_DIR, 'final_rf_pipeline.pkl'))
svm_pipeline = joblib.load(os.path.join(PKL_DIR, 'final_svm_pipeline.pkl'))
persona_kmeans = joblib.load(os.path.join(PKL_DIR, 'persona_kmeans.pkl'))
persona_pca = joblib.load(os.path.join(PKL_DIR, 'persona_pca.pkl'))
persona_preprocessor = joblib.load(os.path.join(PKL_DIR, 'persona_preprocessor.pkl'))

df = pd.read_csv(os.path.join(BASE_DIR, '..', 'cleaned_survey.csv'))
persona_profiles = pd.read_csv(os.path.join(BASE_DIR, '..', 'personas_cluster_profiles.csv'))

st.set_page_config(page_title="Mental Health App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data", "📈 Visuals", "🎯 Regression", "🔑 Classification", "🧩 Persona Clustering"])

def user_inputs():
    return {
        'Country': st.selectbox("Country", df['Country'].unique()),
        'Gender': st.selectbox("Gender", df['Gender'].unique()),
        'self_employed': st.selectbox("Self Employed", df['self_employed'].unique()),
        'family_history': st.selectbox("Family History", df['family_history'].unique()),
        'treatment': st.selectbox("Treatment", df['treatment'].unique()),
        'remote_work': st.selectbox("Remote Work", df['remote_work'].unique()),
        'tech_company': st.selectbox("Tech Company", df['tech_company'].unique()),
        'benefits': st.selectbox("Benefits", df['benefits'].unique()),
        'care_options': st.selectbox("Care Options", df['care_options'].unique()),
        'wellness_program': st.selectbox("Wellness Program", df['wellness_program'].unique()),
        'seek_help': st.selectbox("Seek Help", df['seek_help'].unique()),
        'anonymity': st.selectbox("Anonymity", df['anonymity'].unique()),
        'mental_health_consequence': st.selectbox("Mental Health Consequence", df['mental_health_consequence'].unique()),
        'phys_health_consequence': st.selectbox("Phys Health Consequence", df['phys_health_consequence'].unique()),
        'coworkers': st.selectbox("Coworkers", df['coworkers'].unique()),
        'supervisor': st.selectbox("Supervisor", df['supervisor'].unique()),
        'mental_health_interview': st.selectbox("Mental Health Interview", df['mental_health_interview'].unique()),
        'phys_health_interview': st.selectbox("Phys Health Interview", df['phys_health_interview'].unique()),
        'mental_vs_physical': st.selectbox("Mental vs Physical", df['mental_vs_physical'].unique()),
        'obs_consequence': st.selectbox("Obs Consequence", df['obs_consequence'].unique()),
        'no_employees': st.selectbox("No Employees", df['no_employees'].unique()),
        'leave': st.selectbox("Leave", df['leave'].unique()),
        'work_interfere': st.selectbox("Work Interfere", df['work_interfere'].unique()),
        'Age': st.number_input("Age", min_value=18, max_value=100, value=30, step=1),
    }

if page == "🏠 Home":
    st.title("🏠 Welcome to the Mental Health App")
    st.write("This app predicts mental health personas, age regression, and treatment classification using ML models.")
    st.info("Use the sidebar to switch between pages.")

elif page == "📊 Data":
    st.title("📊 Data")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
    st.write(df.describe())

elif page == "📈 Visuals":
    st.title("📈 Model Testing & Visualizations")

    X_reg = df.drop(columns=['Age'])
    y_reg = df['Age']
    y_pred_reg = rf_pipeline.predict(X_reg)
    rmse = np.sqrt(mean_squared_error(y_reg, y_pred_reg))
    mae = mean_absolute_error(y_reg, y_pred_reg)
    r2 = r2_score(y_reg, y_pred_reg)

    st.subheader("🎯 Random Forest Regression")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**R²:** {r2:.4f}")

    fig1, ax1 = plt.subplots()
    ax1.scatter(y_reg, y_pred_reg, alpha=0.5)
    ax1.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'r--')
    ax1.set_xlabel("Actual Age")
    ax1.set_ylabel("Predicted Age")
    ax1.set_title("Random Forest Regression: Actual vs Predicted")
    st.pyplot(fig1)

    X_clf = df.drop(columns=['treatment'])
    y_clf = df['treatment'].map({'Yes': 1, 'No': 0})
    y_pred_clf = svm_pipeline.predict(X_clf)
    y_prob_clf = svm_pipeline.predict_proba(X_clf)[:, 1]

    acc = accuracy_score(y_clf, y_pred_clf)
    f1 = f1_score(y_clf, y_pred_clf)
    roc = roc_auc_score(y_clf, y_prob_clf)

    st.subheader("🔑 SVM Classification")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**ROC-AUC:** {roc:.4f}")

    fig2, ax2 = plt.subplots()
    RocCurveDisplay.from_predictions(y_clf, y_prob_clf, ax=ax2)
    ax2.set_title("SVM Classifier ROC Curve")
    st.pyplot(fig2)

    cm = confusion_matrix(y_clf, y_pred_clf)
    st.write("**Confusion Matrix:**")
    st.write(cm)
    st.text(classification_report(y_clf, y_pred_clf))

    st.subheader("🧩 KMeans Persona Clustering")
    persona_input = df.drop(columns=[col for col in ['Cluster', 'Persona'] if col in df.columns])
    X_proc = persona_preprocessor.transform(persona_input)
    X_pca = persona_pca.transform(X_proc)
    labels = persona_kmeans.predict(X_pca)
    silhouette = silhouette_score(X_pca, labels)

    st.write(f"**Silhouette Score:** {silhouette:.3f}")
    st.write(f"**Cluster Sizes:** {np.bincount(labels)}")

    fig3, ax3 = plt.subplots()
    ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
    ax3.set_xlabel("PCA 1")
    ax3.set_ylabel("PCA 2")
    ax3.set_title("KMeans Clusters (First 2 PCA Components)")
    st.pyplot(fig3)

elif page == "🎯 Regression":
    st.title("🎯 Predict Age")
    st.write("**Fill all features below:**")
    features = user_inputs()
    if st.button("Predict Age"):
        X = pd.DataFrame([features])
        pred_age = rf_pipeline.predict(X)[0]
        st.success(f"🎯 Predicted Age: {pred_age:.1f} years")

elif page == "🔑 Classification":
    st.title("🔑 Predict Treatment")
    st.write("**Fill all features below:**")
    features = user_inputs()
    features.pop('treatment')
    X = pd.DataFrame([features])
    if st.button("Predict Treatment"):
        pred = svm_pipeline.predict(X)[0]
        label = "Yes" if pred == 1 else "No"
        st.success(f"🔑 Predicted Treatment: {label}")

elif page == "🧩 Persona Clustering":
    st.title("🧩 Persona Clustering")
    st.write("**Fill all features below:**")
    features = user_inputs()
    X = pd.DataFrame([features])
    if st.button("Predict Persona Cluster"):
        X_proc = persona_preprocessor.transform(X)
        X_pca = persona_pca.transform(X_proc)
        cluster = persona_kmeans.predict(X_pca)[0]
        st.success(f"🧩 Predicted Persona Cluster: {cluster}")
        profile_row = persona_profiles.iloc[cluster]
        description = profile_row['Description']
        st.info(f"**Cluster Description:** {description}")
