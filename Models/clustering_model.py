import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib  


def load_persona_data(path):
    """
    Load cleaned survey data
    """
    return pd.read_csv(path)


def clean_columns(df):
    """
    Clean columns again to be sure results are solid
    """
    df['Country'] = df['Country'].apply(lambda x: x if x in ['United States', 'Canada', 'United Kingdom'] else 'Other')

    for col in ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity']:
        df[col] = df[col].replace(["Don't know", "Not sure"], 'No')

    leave_order = ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"]
    df['leave'] = df['leave'].apply(lambda x: x if x in leave_order else "Don't know")

    work_order = ['Never', 'Rarely', 'Sometimes', 'Often']
    df['work_interfere'] = df['work_interfere'].apply(lambda x: x if x in work_order else 'Never')

    no_emp_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
    df['no_employees'] = df['no_employees'].apply(lambda x: x if x in no_emp_order else '1-5')

    return df, leave_order, work_order, no_emp_order


def build_pipeline(leave_order, work_order, no_emp_order):
    """
    Preprocessing pipeline
    """
    num_features = ['Age']
    ord_features = ['work_interfere', 'no_employees', 'leave', 'coworkers', 'supervisor']
    nom_features = [
        'Gender', 'Country', 'self_employed', 'family_history', 'treatment',
        'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program',
        'seek_help', 'anonymity', 'mental_health_consequence', 'phys_health_consequence',
        'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
    ]

    num_pipeline = Pipeline([('scaler', StandardScaler())])

    ord_pipeline = Pipeline([
        ('ord', OrdinalEncoder(categories=[
            work_order, no_emp_order, leave_order,
            ['No', 'Some of them', 'Yes'],
            ['No', 'Some of them', 'Yes']
        ]))
    ])

    nom_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('ord', ord_pipeline, ord_features),
        ('nom', nom_pipeline, nom_features)
    ])

    return preprocessor


def apply_pca(df, preprocessor):
    """
    Fit & apply PCA for dimensionality reduction
    """
    X_processed = preprocessor.fit_transform(df)
    pca = PCA(n_components=0.3)
    X_reduced = pca.fit_transform(X_processed)
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.2f}")

    # Save preprocessor and PCA for reuse
    joblib.dump(preprocessor, 'Test/persona_preprocessor.pkl')
    joblib.dump(pca, 'Test/persona_pca.pkl')

    return X_reduced


def run_clustering(X_reduced, df):
    """
    Run KMeans
    """
    clusterer = KMeans(n_clusters=3, n_init=20, max_iter=200, random_state=42)
    cluster_labels = clusterer.fit_predict(X_reduced)
    df['Cluster'] = cluster_labels

    
    joblib.dump(clusterer, 'Test/persona_kmeans.pkl')

    return df


def map_persona_labels(df):
    """
    Attach persona labels to clusters
    """
    labels = {}

    for cluster_id in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster_id]
        avg_age = cluster_df['Age'].mean()
        treatment = cluster_df['treatment'].mode().values[0]
        benefits = cluster_df['benefits'].mode().values[0]
        interfere = cluster_df['work_interfere'].mode().values[0]
        country = cluster_df['Country'].mode().values[0]
        company_size = cluster_df['no_employees'].mode().values[0]

        if benefits == 'No' and treatment == 'Yes' and interfere in ['Often', 'Sometimes']:
            persona = "Young Silent Sufferers" if avg_age < 30 else "Mature Silent Sufferers"
        elif benefits == 'Yes' and treatment == 'Yes' and interfere in ['Sometimes', 'Often']:
            persona = "Open Advocates"
        elif benefits == 'No' and treatment == 'No' and country == 'United States':
            persona = "Small Firm Hidden Strugglers" if company_size in ['1-5', '6-25'] else "Corporate Hidden Strugglers"
        elif benefits == 'No' and treatment == 'No' and country != 'United States':
            persona = "Hidden Global Strugglers"
        elif benefits == 'Yes' and treatment == 'No':
            persona = "Well-Supported Skeptics"
        elif benefits == 'Yes' and interfere == 'Never':
            persona = "Protected Professionals"
        else:
            persona = f"Unclear Persona {cluster_id}"

        labels[cluster_id] = persona

    df['Persona'] = df['Cluster'].map(labels)

    print("\nCluster Personas:")
    for k, v in labels.items():
        print(f"Cluster {k}: {v}")

    return df


def profile_personas(df):
    """ Build final profiles """
    profile = df.groupby('Persona').agg({
        'Age': ['mean', 'std'],
        'Country': lambda x: x.value_counts().index[0],
        'Gender': lambda x: x.value_counts().index[0],
        'benefits': lambda x: x.value_counts().index[0],
        'treatment': lambda x: x.value_counts().index[0],
        'work_interfere': lambda x: x.value_counts().index[0],
        'no_employees': lambda x: x.value_counts().index[0]
    }).reset_index()

    descriptions = {
        "Mature Silent Sufferers": "Older pros, quiet, stressed, low benefit use.",
        "Young Silent Sufferers": "Younger, silent stress bearers.",
        "Well-Supported Skeptics": "Has benefits, rarely uses them.",
        "Open Advocates": "Openly talk about mental health, use benefits.",
        "Small Firm Hidden Strugglers": "Small orgs, hidden stress.",
        "Corporate Hidden Strugglers": "Big firms, low trust in support.",
        "Hidden Global Strugglers": "Non-US, minimal support.",
        "Protected Professionals": "Good benefits, low stress right now."
    }

    profile['Description'] = profile['Persona'].map(descriptions)

    print("\nPersona Profiles:")
    print(profile)

    return profile


def evaluate_clusters(X_reduced, cluster_labels):
    """
    Evaluate silhouette
    """
    score = silhouette_score(X_reduced, cluster_labels)
    print(f"\n Silhouette Score: {score:.3f}")
    return score


def run_persona_pipeline():
    """
    all functions called
    """
    df = load_persona_data(r"../OL-7NUZJP/cleaned_survey.csv")
    df, leave_order, work_order, no_emp_order = clean_columns(df)
    preprocessor = build_pipeline(leave_order, work_order, no_emp_order)
    X_reduced = apply_pca(df, preprocessor)
    df = run_clustering(X_reduced, df)
    df = map_persona_labels(df)
    profile = profile_personas(df)
    evaluate_clusters(X_reduced, df['Cluster'])

    df.to_csv("personas_clustered_output.csv", index=False)
    profile.to_csv("personas_cluster_profiles.csv", index=False)

    print("\nAll done! Model & personas exported.")



run_persona_pipeline()
