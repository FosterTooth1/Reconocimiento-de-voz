import os
import numpy as np
from hmmlearn import hmm
import librosa
from sklearn.preprocessing import StandardScaler

def get_path_relative_to_script(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, relative_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No existe: {full_path}")
    return full_path

def preprocesar_audio(y, sr, 
                      trim_top_db=20, 
                      preemph_coef=0.97,
                      norm_amplitude=True):
    """
    - Trim de silencios inicial/final (umbral en dB).
    - Normalización de amplitud al rango [-1,1].
    - Preénfasis: y[n] = y[n] - coef * y[n-1].
    """
    # 1) Recorte de silencios al inicio/fin
    try:
        y_trim, _ = librosa.effects.trim(y, top_db=trim_top_db)
    except Exception:
        y_trim = y  # Si falla, usar original
    y = y_trim

    # 2) Normalización de amplitud
    if norm_amplitude and np.any(y):
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

    # 3) Preénfasis
    if preemph_coef is not None and preemph_coef > 0:
        # y[n] = y[n] - α * y[n-1]
        y = np.append(y[0], y[1:] - preemph_coef * y[:-1])

    return y

def cargar_mfcc_desde_carpetas(carpeta_hola_rel, carpeta_adios_rel, sr=16000, n_mfcc=13):
    datos = {"hola": [], "adios": []}
    for palabra, carpeta_rel in [("hola", carpeta_hola_rel), ("adios", carpeta_adios_rel)]:
        carpeta = get_path_relative_to_script(carpeta_rel)
        if not os.path.isdir(carpeta):
            raise NotADirectoryError(f"No es carpeta: {carpeta}")
        for fname in os.listdir(carpeta):
            if not fname.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                continue
            ruta = os.path.join(carpeta, fname)
            try:
                y, _ = librosa.load(ruta, sr=sr)
                # Preprocesamiento
                y_proc = preprocesar_audio(y, sr,
                                           trim_top_db=20,
                                           preemph_coef=0.97,
                                           norm_amplitude=True)
                # Extraer MFCC: (n_mfcc, n_frames)
                mfcc = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=n_mfcc)
                X = mfcc.T  # (n_frames, n_mfcc)
                if X.shape[0] < 3:
                    print(f"Audio demasiado corto o pocos frames tras trim: {ruta}")
                    continue
                datos[palabra].append(X)
            except Exception as e:
                print(f"Advertencia: no se pudo procesar {ruta}: {e}")
    # Verificar que haya datos
    for palabra, listas in datos.items():
        if len(listas) == 0:
            raise ValueError(f"No se cargó ningún MFCC para '{palabra}' desde la carpeta {locals()['carpeta_rel']}")
    return datos

def normalizar_datos(datos_dict):
    all_feats = np.vstack([X for listas in datos_dict.values() for X in listas])
    scaler = StandardScaler().fit(all_feats)
    datos_norm = {}
    for palabra, listas in datos_dict.items():
        datos_norm[palabra] = [scaler.transform(X) for X in listas]
    return datos_norm, scaler

def entrenar_modelos(datos_entrenamiento, n_components=3, covariance_type="diag", n_iter=1000):
    modelos = {}
    for palabra, secuencias in datos_entrenamiento.items():
        modelo = hmm.GaussianHMM(n_components=n_components,
                                 covariance_type=covariance_type,
                                 n_iter=n_iter)
        X = np.vstack(secuencias)
        lengths = [len(s) for s in secuencias]
        modelo.fit(X, lengths=lengths)
        modelos[palabra] = modelo
    return modelos

def clasificar_audio(audio_filename, modelos, scaler=None, sr=16000, n_mfcc=13):
    try:
        audio_path = get_path_relative_to_script(audio_filename)
        y, _ = librosa.load(audio_path, sr=sr)
        # Preprocesar igual que en entrenamiento
        y_proc = preprocesar_audio(y, sr,
                                   trim_top_db=20,
                                   preemph_coef=0.97,
                                   norm_amplitude=True)
        mfcc = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=n_mfcc)
        X = mfcc.T
        if X.shape[0] < 3:
            print(f"Audio demasiado corto o pocos frames tras trim: {audio_filename}")
            return None
        if scaler is not None:
            X = scaler.transform(X)
        scores = {}
        for palabra, modelo in modelos.items():
            try:
                scores[palabra] = modelo.score(X)
            except Exception as e:
                scores[palabra] = -np.inf
                print(f"Error al puntuar con modelo '{palabra}': {e}")
        palabra_pred = max(scores, key=scores.get)
        return palabra_pred
    except Exception as e:
        print(f"Error procesando {audio_filename}: {e}")
        return None

if __name__ == "__main__":
    carpeta_hola = "Hola_mp3"     # carpeta con 100 audios "Hola"
    carpeta_adios = "Adios_mp3"   # carpeta con 100 audios "Adios"

    # 1) Cargar MFCCs con preprocesamiento
    datos = cargar_mfcc_desde_carpetas(carpeta_hola, carpeta_adios)

    # 2) Normalizar
    datos_norm, scaler = normalizar_datos(datos)

    # 3) Entrenar modelos HMM
    modelos = entrenar_modelos(datos_norm, n_components=3, covariance_type="diag", n_iter=1000)

    # 4) Clasificar un ejemplo de prueba
    test_file = "Test.mp3" 
    resultado = clasificar_audio(test_file, modelos, scaler=scaler)
    if resultado:
        print(f"Palabra reconocida: {resultado}")
    else:
        print("Clasificación fallida")