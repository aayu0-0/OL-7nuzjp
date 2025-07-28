import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(path):
    """
    Load cleaned survey CSV
    """
    return pd.read_csv(path)


def build_rf_pipeline(num_cols, cat_cols, params):
    """
    Build Random Forest pipeline with preprocessing
    """
    num_pipe = StandardScaler()
    cat_pipe = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    rf = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=42
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', rf)
    ])

    return pipeline



def train_evaluate_rf(df, params):
    """
    Train RF on Age, evaluate and save model
    """
    X = df.drop(columns=['Age'])
    y = df['Age']

    num_cols = []  
    cat_cols = [col for col in X.columns if col not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_rf_pipeline(num_cols, cat_cols, params)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nFinal Random Forest Regression")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")

    joblib.dump(pipe, 'Test/final_rf_regressor.pkl')
    print("\n Final RF Regressor saved as 'final_rf_regressor.pkl'")

    joblib.dump(pipe, 'Test/final_rf_pipeline.pkl')
    print("\n Final RF pipeline saved as 'final_rf_pipeline.pkl'")

    return pipe



df = load_data(r"../OL-7NUZJP/cleaned_survey.csv")


best_rf_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2
    }


final_model = train_evaluate_rf(df, best_rf_params)

    

    
