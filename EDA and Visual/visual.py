import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay,
    mean_squared_error, mean_absolute_error, r2_score, silhouette_score
)
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, '..', 'pkl_files')

def load_models():
    rf_pipeline = joblib.load(os.path.join(PKL_DIR, 'final_rf_pipeline.pkl'))
    svm_pipeline = joblib.load(os.path.join(PKL_DIR, 'final_svm_pipeline.pkl'))
    persona_kmeans = joblib.load(os.path.join(PKL_DIR, 'persona_kmeans.pkl'))
    persona_pca = joblib.load(os.path.join(PKL_DIR, 'persona_pca.pkl'))
    persona_preprocessor = joblib.load(os.path.join(PKL_DIR, 'persona_preprocessor.pkl'))
    print("All models loaded.")
    return rf_pipeline, svm_pipeline, persona_kmeans, persona_pca, persona_preprocessor

def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, '..', 'cleaned_survey.csv'))
    print(f"Data loaded. Shape: {df.shape}")
    return df


def test_regression(df, rf_pipeline):
    X = df.drop(columns=['Age'])
    y = df['Age']
    y_pred = rf_pipeline.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print("\n🎯 Random Forest Regression Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    plt.scatter(y, y_pred, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Random Forest Regression: Actual vs Predicted")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.show()

def test_classification(df, svm_pipeline):
    X = df.drop(columns=['treatment'])
    y_true = df['treatment'].map({'Yes': 1, 'No': 0})
    y_pred = svm_pipeline.predict(X)
    y_prob = svm_pipeline.predict_proba(X)[:, 1]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    print("\n🔑 SVM Classification Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("SVM Classifier ROC Curve")
    plt.show()

def test_persona_clustering(df, persona_preprocessor, persona_pca, persona_kmeans):
    persona_input = df.drop(columns=[col for col in ['Cluster', 'Persona'] if col in df.columns])
    X_proc = persona_preprocessor.transform(persona_input)
    X_pca = persona_pca.transform(X_proc)
    labels = persona_kmeans.predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"\n🧩 KMeans Clustering Results:")
    print(f"Silhouette Score: {score:.3f}")
    print("Cluster sizes:", np.bincount(labels))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title("KMeans Clusters (First 2 PCA Components)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def main():
    rf_pipeline, svm_pipeline, persona_kmeans, persona_pca, persona_preprocessor = load_models()
    df = load_data()
    test_regression(df, rf_pipeline)
    test_classification(df, svm_pipeline)
    test_persona_clustering(df, persona_preprocessor, persona_pca, persona_kmeans)
    print("\nAll tests & visualizations done.")

main()
