import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Procesamos los archivos WAV de una carpeta y recolectamos datos
def procesar_carpeta(carpeta_entrada, carpeta_salida="resultados"):
    Path(carpeta_salida).mkdir(exist_ok=True)
    archivos = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith('.wav')]
    
    dataset = []
    for archivo in archivos:
        ruta_completa = os.path.join(carpeta_entrada, archivo)
        print(f"\nProcesando: {archivo}")
        datos_audio = procesar_audio(ruta_completa, carpeta_salida)
        dataset.extend(datos_audio)
    
    return dataset

# Carga el audio y devuelve la señal y la frecuencia de muestreo
def cargar_audio(ruta_archivo):
    señal, fs = librosa.load(ruta_archivo, sr=None)
    return señal, fs

# Segmenta el audio en fragmentos de 100ms
def segmentar_audio(señal, fs, ventana_ms=100):
    muestras_por_ventana = int(fs * ventana_ms / 1000)
    return [señal[i:i + muestras_por_ventana] for i in range(0, len(señal), muestras_por_ventana)]

# Clasificación del segmento y extracción de características
def clasificar_segmento(segmento, fs, umbral_silencio=0.02):
    if len(segmento) == 0:
        return 'S', 0.0, 0.0
    
    energia = np.mean(np.abs(segmento))
    zcr = librosa.feature.zero_crossing_rate(segmento, frame_length=2048, hop_length=512)[0]
    zcr_mean = np.mean(zcr)
    
    if energia < umbral_silencio:
        etiqueta = 'S'
    else:
        etiqueta = 'V' if zcr_mean < 0.1 else 'U'
    
    return etiqueta, energia, zcr_mean

# Procesa un archivo y genera visualizaciones
def procesar_audio(ruta_archivo, carpeta_salida):
    señal, fs = cargar_audio(ruta_archivo)
    nombre_base = os.path.splitext(os.path.basename(ruta_archivo))[0]
    segmentos = segmentar_audio(señal, fs)
    
    datos_segmentos = []
    for i, seg in enumerate(segmentos):
        etq, energia, zcr = clasificar_segmento(seg, fs)
        inicio = i * 0.1
        datos_segmentos.append({
            'archivo': nombre_base,
            'segmento_idx': i,
            'inicio': inicio,
            'energia': energia,
            'zcr': zcr,
            'etiqueta': etq
        })
    
    # Generar visualizaciones
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(señal, sr=fs)
    plt.title(f'Forma de onda: {nombre_base}')
    
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(señal, n_fft=256, hop_length=64)), ref=np.max)
    librosa.display.specshow(D, sr=fs, hop_length=64, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma banda ancha (15ms)')
    
    plt.subplot(3, 1, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(señal, n_fft=2048, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, sr=fs, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma banda estrecha (50ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, f"{nombre_base}_analisis.png"))
    plt.close()
    
    return datos_segmentos

if __name__ == "__main__":
    carpeta_audios = "C:\\Users\\albsa\\Desktop\\Reconocimiento de voz\\Practica_2\\Audios_P1"
    carpeta_resultados = "C:\\Users\\albsa\\Desktop\\Reconocimiento de voz\\Practica_2\\Resultados"
    
    # Procesar audios y recolectar datos
    dataset = procesar_carpeta(carpeta_audios, carpeta_resultados)
    
    # Preparar datos para modelos
    X = np.array([[d['energia'], d['zcr']] for d in dataset])
    y = np.array([d['etiqueta'] for d in dataset])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelos
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    nn_model.fit(X_train, y_train)
    
    # Evaluar modelos
    nb_acc = nb_model.score(X_val, y_val)
    dt_acc = dt_model.score(X_val, y_val)
    nn_acc = nn_model.score(X_val, y_val)
    
    print(f"\nPrecisión Naive Bayes: {nb_acc:.2f}")
    print(f"Precisión Árbol de Decisión: {dt_acc:.2f}")
    print(f"Precisión Red Neuronal: {nn_acc:.2f}")
    
    # Predecir etiquetas para todos los segmentos
    X_all = np.array([[d['energia'], d['zcr']] for d in dataset])
    nb_preds = nb_model.predict(X_all)
    dt_preds = dt_model.predict(X_all)
    nn_preds = nn_model.predict(X_all)
    
    for i, d in enumerate(dataset):
        d['nb_pred'] = nb_preds[i]
        d['dt_pred'] = dt_preds[i]
        d['nn_pred'] = nn_preds[i]
    
    # Escribir archivos de predicción
    audio_files = defaultdict(list)
    for d in dataset:
        audio_files[d['archivo']].append(d)
    
    for archivo, segmentos in audio_files.items():
        segmentos_sorted = sorted(segmentos, key=lambda x: x['segmento_idx'])
        for model, suffix in zip(['nb_pred', 'dt_pred', 'nn_pred'], ['Bayes', 'Arbol', 'RedNeuronal']):
            with open(os.path.join(carpeta_resultados, f"{archivo}_clasificacion_{suffix}.txt"), 'w') as f:
                for seg in segmentos_sorted:
                    inicio = seg['inicio']
                    etiqueta = seg[model]
                    f.write(f"[{inicio:.1f}-{inicio + 0.1:.1f}s]: {etiqueta}\n")
    
    # Generar reporte de precisión
    with open(os.path.join(carpeta_resultados, "reporte_modelos.txt"), 'w') as f:
        f.write(f"Naive Bayes: {nb_acc:.4f}\n")
        f.write(f"Árbol de Decisión: {dt_acc:.4f}\n")
        f.write(f"Red Neuronal: {nn_acc:.4f}\n")
    
    print("\nProcesamiento completado. Resultados guardados en:", carpeta_resultados)