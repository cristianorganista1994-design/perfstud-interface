from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

# Componente oficial y moderno de Keras 3
import keras

app = Flask(__name__)
CORS(app)  # Habilita la comunicación cruzada con la interfaz en Vercel

# Variables globales para contener los modelos en memoria
scaler = None
modelo_rf = None
modelo_lr = None
modelo_nn = None

def cargar_modelos_sistema():
    global scaler, modelo_rf, modelo_lr, modelo_nn
    try:
        # 1. Cargar el escalador y los modelos tradicionales (.pkl)
        scaler = joblib.load('scaler.pkl')
        modelo_rf = joblib.load('modelo_rf.pkl')
        modelo_lr = joblib.load('modelo_lr.pkl')
        print("✅ Escalador, Random Forest y Regresión Logística cargados (.pkl)")

        # 2. Reconstruir la Red Neuronal de Keras 3 usando el string plano del JSON
        if os.path.exists('config.json') and os.path.exists('model.weights.h5'):
            with open('config.json', 'r') as json_file:
                # Leemos el archivo estrictamente como un String de texto plano
                configuracion_string = json_file.read()
            
            # Reconstrucción nativa pasándole el string de texto
            modelo_nn = keras.models.model_from_json(configuracion_string)
            
            # Acoplar los pesos exactos optimizados por el Algoritmo Genético
            modelo_nn.load_weights('model.weights.h5')
            print("🧠 ¡Red Neuronal + Algoritmo Genético reconstruida y cargada con éxito en Keras 3!")
        else:
            print("❌ Error crítico: Faltan archivos de la red neuronal (config.json o model.weights.h5)")
            
    except Exception as e:
        print(f"❌ Error crítico en la inicialización de los modelos: {str(e)}")

# Invocar la carga de los modelos al iniciar el script
cargar_modelos_sistema()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'No se proporcionaron características en el cuerpo del mensaje'}), 400
        
        feat_dict = data['features']
        
        # 3. Preprocesamiento: Escalado de las 3 variables continuas reales con MinMaxScaler
        continuas = np.array([[
            float(feat_dict['Age']),
            float(feat_dict['StudyTimeWeekly']),
            float(feat_dict['Absences'])
        ]])
        continuas_escaladas = scaler.transform(continuas)[0]
        
        # 4. Preprocesamiento: Expansión One-Hot Encoding manual para Ethnicity (0 a 3)
        etnia_seleccionada = int(feat_dict['Ethnicity'])
        eth_0 = 1.0 if etnia_seleccionada == 0 else 0.0
        eth_1 = 1.0 if etnia_seleccionada == 1 else 0.0
        eth_2 = 1.0 if etnia_seleccionada == 2 else 0.0
        eth_3 = 1.0 if etnia_semibold == 3 else 0.0 # Control de asignación binaria
        
        # 5. Configuración estructurada del vector de entrada final (15 dimensiones exactas)
        vector_final = np.array([[
            continuas_escaladas[0],         # Age (Escalado)
            float(feat_dict['Gender']),      # Gender
            float(feat_dict['ParentalEducation']),
            continuas_escaladas[1],         # StudyTimeWeekly (Escalado)
            continuas_escaladas[2],         # Absences (Escalado)
            float(feat_dict['Tutoring']),
            float(feat_dict['ParentalSupport']),
            float(feat_dict['Extracurricular']),
            float(feat_dict['Sports']),
            float(feat_dict['Music']),
            float(feat_dict['Volunteering']),
            eth_0, eth_1, eth_2, eth_3      # Columnas dummies binarias expandidas
        ]], dtype=np.float32)

        # 6. Inferencia - Modelo 1: Random Forest
        clase_rf = int(modelo_rf.predict(vector_final)[0])
        conf_rf = float(np.max(modelo_rf.predict_proba(vector_final)))

        # 7. Inferencia - Modelo 2: Regresión Logística
        clase_lr = int(modelo_lr.predict(vector_final)[0])
        conf_lr = float(np.max(modelo_lr.predict_proba(vector_final)))

        # 8. Inferencia - Modelo 3: Red Neuronal Secuencial (Keras 3)
        predicciones_softmax = modelo_nn.predict(vector_final, verbose=0)
        clase_nn = int(np.argmax(predicciones_softmax[0]))  # Índice (0-4) con la probabilidad más alta
        conf_nn = float(np.max(predicciones_softmax[0]))    # Confianza matemática de esa clase

        # 9. Metamodelo: Fusión por Ensamble de Promedio Ordinal Híbrido (Hard/Soft Voting)
        clase_ensamble = int(round((clase_rf + clase_lr + clase_nn) / 3.0))
        conf_ensamble = float((conf_rf + conf_lr + conf_nn) / 3.0)

        # 10. Retornar la respuesta JSON limpia esperada por los componentes de React
        return jsonify({
            'status': 'success',
            'models_outputs': [
                {
                    'name': 'Red Neuronal + Algoritmo Genético',
                    'claseId': clase_nn,
                    'confidence': conf_nn,
                    'isEnsemble': False
                },
                {
                    'name': 'Árboles de Decisión',
                    'claseId': clase_rf,
                    'confidence': conf_rf,
                    'isEnsemble': False
                },
                {
                    'name': 'Regresión Logística',
                    'claseId': clase_lr,
                    'confidence': conf_lr,
                    'isEnsemble': False
                },
                {
                    'name': 'Ensamble Predictivo (Promedio de Modelos)',
                    'claseId': clase_ensamble,
                    'confidence': conf_ensamble,
                    'isEnsemble': True
                }
            ]
        })

    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

if __name__ == '__main__':
    # Lanzamos el backend en el puerto 5000 de forma síncrona para Keras
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
