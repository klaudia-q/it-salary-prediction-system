import pandas as pd
import numpy as np
import joblib  # Do zapisywania modelu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

FILE_PATH = 'Zarobki_IT.csv'
MODEL_PATH = 'best_model_pipeline.pkl'

def load_and_preprocess(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

    # Feature Engineering 
    df.dropna(subset=['salary_in_usd'], inplace=True)
    df = df.drop_duplicates()
    
    title = df['job_title'].str.lower()
    conditions = [
        title.str.contains('manager|director|head|lead'),
        title.str.contains('machine learning|ml|ai|nlp'),
        title.str.contains('scientist|research'),
        title.str.contains('analyst|business intelligence|bi'),
        title.str.contains('engineer|architect|etl')
    ]
    choices = ['Management', 'Machine Learning', 'Data Scientist', 'Data Analyst', 'Data Engineering']
    df['job_category'] = np.select(conditions, choices, default='Other')
    
    # Target
    threshold = 140000
    df['label'] = np.where(df['salary_in_usd'] > threshold, 1, 0)
    
    return df

def get_pipeline(classifier):
    """
    Tworzy Pipeline, który automatycznie przetwarza dane tekstowe (OneHot) i numeryczne.
    Dzięki temu aplikacja nie musi martwić się o ręczne kodowanie zmiennych.
    """
    cat_cols = ["job_category", "experience_level", "employment_type", "company_size", "remote_ratio"]
    
    # Preprocessor: Automatyczne One-Hot Encoding dla zmiennych kategorycznych
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'  # Zmienne numeryczne (work_year) przechodzą bez zmian
    )
    
    # Łączymy preprocessor z modelem w jeden obiekt
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline

def main():
    df = load_and_preprocess(FILE_PATH)
    if df.empty:
        print("Brak danych.")
        return

    # Wybór cech (X) i celu (y)
    features = ["job_category", "experience_level", "employment_type", "company_size", "remote_ratio", "work_year"]
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Definicja modeli wewnątrz Pipeline
    models = {
        "Regresja Logistyczna": get_pipeline(LogisticRegression(solver='liblinear', random_state=42)),
        "Drzewo Decyzyjne": get_pipeline(DecisionTreeClassifier(max_depth=5, random_state=42)),
        "Las Losowy": get_pipeline(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    }

    best_score = 0
    best_model = None
    best_name = ""

    print("--- Rozpoczynam trenowanie i ocenę modeli ---")
    
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        # Obsługa predict_proba dla ROC AUC
        if hasattr(pipeline['classifier'], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
        else:
            roc = 0
            
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nModel: {name}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc:.4f}")
        
        # Wybór najlepszego modelu na podstawie F1
        if f1 > best_score:
            best_score = f1
            best_model = pipeline
            best_name = name

    print(f"\nNajlepszy model: {best_name} (F1: {best_score:.4f})")
    
    # Zapisanie modelu do pliku .pkl
    joblib.dump(best_model, MODEL_PATH)
    print(f"Zapisano najlepszy model do pliku: {MODEL_PATH}")

if __name__ == "__main__":

    main()
