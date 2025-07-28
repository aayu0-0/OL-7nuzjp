import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
import matplotlib.pyplot as plt
import joblib


def read_data():
    """ Read the data """
    return pd.read_csv(r"../OL-7NUZJP/cleaned_survey.csv")


def build_pipe(num_cols, cat_cols):
    """ 
    Make the pipeline here as seperate one for all 2st then combine them all
    """

    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    best_svm = SVC(
        C=1,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )

    main_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_svm)
    ])

    return main_pipe


def train_and_evaluate_svm(df):
    """
    Finally train the model and save it after testing and results
    """

    X = df.drop(columns=['treatment'])
    y = df['treatment'].map({'Yes': 1, 'No': 0})

    num_cols = ['Age']
    cat_cols = [col for col in X.columns if col not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    main_pipe = build_pipe(num_cols, cat_cols)
    main_pipe.fit(X_train, y_train)

    y_pred = main_pipe.predict(X_test)
    y_prob = main_pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", cr)

    RocCurveDisplay.from_estimator(main_pipe, X_test, y_test)
    plt.title("ROC Curve - Final Tuned SVM")
    plt.show()

    joblib.dump(main_pipe, 'Test/final_svm_model.pkl')
    print(" Final SVM model saved as 'final_svm_model.pkl'")

    joblib.dump(main_pipe, 'Test/final_svm_pipeline.pkl')
    print("Final SVM pipeline saved as 'final_svm_pipeline.pkl'")





df = read_data()


train_and_evaluate_svm(df)
