from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
MODEL_PATH = 'best_model_pipeline.pkl'
DB_PATH = 'predictions.db'

# 1. Ładowanie modelu przy starcie aplikacji
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model załadowany pomyślnie.")
else:
    print("BŁĄD: Nie znaleziono pliku modelu. Uruchom najpierw train_model.py")
    model = None

# 2. Inicjalizacja bazy danych
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            job_category TEXT,
            experience_level TEXT,
            prediction INTEGER,
            probability REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Pobranie danych JSON od użytkownika
        data = request.get_json()
        
        # Konwersja do DataFrame (format oczekiwany przez Pipeline)
        input_df = pd.DataFrame([data])
        
        # Predykcja
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] # Prawdopodobieństwo klasy 1
        
        result_label = "Wysokie zarobki (>140k USD)" if prediction == 1 else "Standardowe zarobki"

        # Zapis do bazy danych
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history (timestamp, job_category, experience_level, prediction, probability)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), data.get('job_category'), data.get('experience_level'), int(prediction), float(probability)))
        conn.commit()
        conn.close()

        return jsonify({
            'prediction': int(prediction),
            'label': result_label,
            'probability': float(probability)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint do podglądu historii z bazy
@app.route('/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC LIMIT 10", conn)
    conn.close()
    return df.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True, port=5000)