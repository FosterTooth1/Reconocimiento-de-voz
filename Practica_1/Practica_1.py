import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import soundfile as sf
from pathlib import Path

# Procesamos los archivos WAV de una carpeta
def procesar_carpeta(carpeta_entrada, carpeta_salida="resultados"):
    # Creamos la carpeta de resultados
    Path(carpeta_salida).mkdir(exist_ok=True)
    
    # Listamos los archivos WAV
    archivos = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith('.wav')]
    
    for archivo in archivos:
        ruta_completa = os.path.join(carpeta_entrada, archivo)
        print(f"\nProcesando: {archivo}")
        procesar_audio(ruta_completa, carpeta_salida)

# Carga el audio y devuelve la señal y la frecuencia de muestreo
def cargar_audio(ruta_archivo):
    señal, fs = librosa.load(ruta_archivo, sr=None)
    return señal, fs
# Segmentamos el audio en fragmentos de 100ms
def segmentar_audio(señal, fs, ventana_ms=100):
    muestras_por_ventana = int(fs * ventana_ms / 1000)
    return [señal[i:i + muestras_por_ventana] for i in range(0, len(señal), muestras_por_ventana)]

# Clasificamos el segmento en S, U o V según energía y tasa de cruces por cero
def clasificar_segmento(segmento, fs, umbral_silencio=0.02):
    if len(segmento) == 0:
        return 'S'
    
    energia = np.mean(np.abs(segmento))
    
    # Detectamos silencio (S)
    if energia < umbral_silencio:
        return 'S'
    
    # Detectamos sonoridad (V vs U)
    zcr = librosa.feature.zero_crossing_rate(segmento, frame_length=2048, hop_length=512)[0]
    return 'V' if np.mean(zcr) < 0.1 else 'U'

# Procesamos un archivo y guardamos los resultados
def procesar_audio(ruta_archivo, carpeta_salida):
    señal, fs = cargar_audio(ruta_archivo)
    nombre_base = os.path.splitext(os.path.basename(ruta_archivo))[0]
    segmentos = segmentar_audio(señal, fs)
    
    # Clasificamos cada segmento
    etiquetas = [clasificar_segmento(seg, fs) for seg in segmentos]
    
    # Guardamos la clasificación en un archivo de texto
    with open(os.path.join(carpeta_salida, f"{nombre_base}_clasificacion.txt"), 'w') as f:
        for i, etq in enumerate(etiquetas):
            inicio = i * 0.1
            f.write(f"[{inicio:.1f}-{inicio + 0.1:.1f}s]: {etq}\n")
    
    # Generamos visualizaciones del audio
    plt.figure(figsize=(15, 15))
    
    # Mostramos la forma de onda
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(señal, sr=fs)
    plt.title(f'Forma de onda: {nombre_base}')
    
    # Mostramos el espectrograma de banda ancha (15ms)
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(señal, n_fft=256, hop_length=64)), ref=np.max)
    librosa.display.specshow(D, sr=fs, hop_length=64, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma banda ancha (15ms)')
    
    # Mostramos el espectrograma de banda estrecha (50ms)
    plt.subplot(3, 1, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(señal, n_fft=2048, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, sr=fs, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma banda estrecha (50ms)')
    
    # Guardamos y cerramos la figura
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, f"{nombre_base}_analisis.png"))
    plt.close()

if __name__ == "__main__":
    # Definimos las rutas de entrada y salida
    carpeta_audios = "C:\\Users\\albsa\\Desktop\\Reconocimiento de voz\\Audios_P1"  # Ruta con los archivos WAV
    carpeta_resultados = "C:\\Users\\albsa\\Desktop\\Reconocimiento de voz\\Resultados"  # Ruta para guardar los resultados
    
    # Procesamos todos los audios
    procesar_carpeta(carpeta_audios, carpeta_resultados)
    print("\nProcesamiento completo! Resultados guardados en:", carpeta_resultados)
