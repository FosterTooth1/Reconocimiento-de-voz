import numpy as np
from hmmlearn import hmm
import librosa
import os

# Función para obtener la ruta del archivo
def get_file_path(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Ruta del script
    file_path = os.path.join(current_dir, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo {filename} no encontrado en: {current_dir}")
    
    return file_path

# Generamos datos sintéticos de audio y características MFCC
def generar_datos_sinteticos():
    # Configuración de parámetros
    # Número de muestras por palabra y palabras a simular
    n_muestras = 300  # 10 muestras por palabra
    palabras = ["hola", "adios"]
    mfccs_por_palabra = {}
    
    # Para cada palabra, generar MFCCs sintéticos con patrones diferenciables
    for palabra in palabras:
        mfccs = []
        for _ in range(n_muestras):
            # Simulamos MFCCs con media diferente para cada palabra
            if palabra == "hola":
                base = np.linspace(0, 2, 13) + np.random.normal(0, 0.2, 13)
            else:
                base = np.linspace(2, 0, 13) + np.random.normal(0, 0.2, 13)
            
            # Creamos una secuencia temporal (10 frames)
            secuencia = np.array([base + i*0.1 for i in range(10)])
            mfccs.append(secuencia)
        
        mfccs_por_palabra[palabra] = mfccs
    
    return mfccs_por_palabra

# Entrenamos los modelos HMM para cada palabra
def entrenar_modelos(datos_entrenamiento):
    modelos = {}
    
    for palabra, secuencias in datos_entrenamiento.items():
        # Configuración modelo HMM
        modelo = hmm.GaussianHMM(
            n_components=3,  # 3 estados ocultos
            covariance_type="diag",  # Matriz de covarianza diagonal
            n_iter=1000  # Iteraciones para entrenamiento
        )
        
        # Preparamos los datos: concatenar secuencias y obtener longitudes
        X = np.concatenate(secuencias)
        lengths = [len(s) for s in secuencias]
        
        # Entrenamos el modelo con el algoritmo Baum-Welch
        modelo.fit(X, lengths=lengths)
        modelos[palabra] = modelo
    
    return modelos

# Clasificamos una nueva muestra
def clasificar_audio(audio_filename, modelos):
    try:
        # Extraemos MFCCs del audio (usando librosa)
        audio_path = get_file_path(audio_filename)
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Calculamos la probabilidad logarítmica para cada modelo
        X = mfcc.T  
        scores = {palabra: modelo.score(mfcc) for palabra, modelo in modelos.items()}
        
        # Seleccionamos la palabra con mayor probabilidad
        return max(scores, key=scores.get)
    except Exception as e:
        print(f"Error procesando {audio_filename}: {str(e)}")
        return None


# Ejecución de la simulación
if __name__ == "__main__":
    # Paso 1: Generamos los datos de entrenamiento
    datos_entrenamiento = generar_datos_sinteticos()
    
    # Paso 2: Entrenamos los modelos HMM
    modelos = entrenar_modelos(datos_entrenamiento)
    
    # Paso 3: Clasificamos la nueva muestra
    # Archivo con la palabra "hola"
    test_file = "p_46133354_698.wav"
    try:
        resultado = clasificar_audio(test_file, modelos)
        if resultado:
            print(f"Palabra reconocida: {resultado}")
        else:
            print("Clasificación fallida")
    except Exception as e:
        print(f"Error general: {str(e)}")