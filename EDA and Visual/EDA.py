import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "survey.csv")

print("Using dataset at:", DATA_PATH)


def load_data():
    """
    Load the cleaned_survey.csv from OL-7NUZJP directory
    """
    df = pd.read_csv(DATA_PATH)
    return df


def drop_columns(df):
    """
    Drop unnecessary columns
    """
    return df.drop(columns=["Timestamp", "comments", "state"])


def fill_self_employed(df, column='self_employed'):
    """
    Fill NaN values in self_employed  with Yes or No
    """
    nan_indices = df[df[column].isna()].index

    if len(nan_indices) >= 2:
        df.loc[nan_indices[:2], column] = "Yes"
        df.loc[nan_indices[2:], column] = "No"
    
    return df


def clean_gender(gender):
    """
    cleaning the  gender and categorizing them in just 3 categories. Male Female Others
    """
    gender = str(gender).strip().lower()

    male_terms = {'male', 'm', 'man', 'make', 'cis male', 'male (cis)', 'mal', 'maile', 'msle', 'malr', 'mail', 'cis man', 'guy (-ish) ^_^', 'trans man'}
    
    female_terms = {'female', 'f', 'woman', 'cis female', 'cis-female/femme', 'trans woman', 'trans-female', 'female (trans)', 'female (cis)', 'femake', 'femail'}
    
    if gender in male_terms:
        return 'Male'
    elif gender in female_terms:
        return 'Female'
    else:
        return 'Others'


def apply_clean_gender(df, column='Gender'):
    """
    Apply clean_gender fundtion
    """

    df[column] = df[column].apply(clean_gender)
    return df


def fix_work_interfere(df):
    """
    Fix NaNs in work_interfere
    """

    print("Before:")
    print(df["work_interfere"].value_counts(), "\nMissing:", df['work_interfere'].isna().sum())
    
    target = 'work_interfere'
    num_cols = ['Age']
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' and col != target]
    
    train_df = df[df[target].notna()].copy()
    test_df = df[df[target].isna()].copy()
    
    y = train_df[target]
    
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    scaler = StandardScaler()
    X_num = scaler.fit_transform(train_df[num_cols])
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = ohe.fit_transform(train_df[cat_cols])
    
    X = np.concatenate([X_num, X_cat], axis=1)
    X_test_num = scaler.transform(test_df[num_cols])
    X_test_cat = ohe.transform(test_df[cat_cols])
    X_test = np.concatenate([X_test_num, X_test_cat], axis=1)
    
    X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    clf = ExtraTreesClassifier(n_estimators=200, max_depth=10, max_features='sqrt', min_samples_split=5, min_samples_leaf=2, random_state=42)
    
    clf.fit(X_train, y_train)
    
    preds_encoded = clf.predict(X_test)
    preds = le_y.inverse_transform(preds_encoded)
    
    df.loc[test_df.index, target] = preds
    
    print("\nAfter:")
    print(f"Filled {len(test_df)} missing '{target}' values.")
    print(df["work_interfere"].value_counts(), "\nMissing:", df['work_interfere'].isna().sum())
    return df


def fix_age(df):
    """
    Fix the Age column
    """
    print("Before:")
    print(df['Age'].value_counts(), "\nMissing:", df['Age'].isna().sum())

    df['Age'] = abs(df['Age'])
    df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]

    print("\nAfter:")
    print(df['Age'].value_counts(), "\nMissing:", df['Age'].isna().sum())

    return df

def plot_summary(df):
    """
    Visualize the data
    """
    pair_df = df.copy()

    pair_df['self_employed'] = pair_df['self_employed'].astype(str)

    sns.pairplot(pair_df, vars=['Age'], hue='Gender', palette='Set2', diag_kind='kde', markers=['o', 's', 'D'])
    
    plt.suptitle('Pairplot: Age vs. Gender', y=1.02)
    plt.show()

    plt.figure(figsize=(10, 6))
    for interfere in df['work_interfere'].dropna().unique():
        subset = df[df['work_interfere'] == interfere]
        sns.kdeplot(subset['Age'], label=f'{interfere}')
    plt.title('Age Distribution by Work Interfere')
    plt.xlabel('Age')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='self_employed', hue='Gender', palette='pastel')
    plt.title('Self-employed Status by Gender')
    plt.show()

def all_functions():
    """ 
    Apply all the functions 
    """
    df = load_data()
    df = drop_columns(df)
    df = fill_self_employed(df)
    df = apply_clean_gender(df)
    df = fix_work_interfere(df)
    df = fix_age(df)
    plot_summary(df)
    return df

df = all_functions()
print(df.value_counts())

output_folder = os.path.join(BASE_DIR, "test")
output_path = os.path.join(output_folder, "cleaned_survey.csv")
df.to_csv(output_path, index=False)