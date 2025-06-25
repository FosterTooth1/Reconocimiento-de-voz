import os
import numpy as np
from hmmlearn import hmm
import librosa
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def get_path_relative_to_script(relative_path):
    """
    Obtiene la ruta absoluta de un archivo o carpeta, a partir de una ruta relativa
    """
    # Directorio donde está el script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Unir ruta actual con la ruta relativa 
    full_path = os.path.join(current_dir, relative_path)
    # Verificar que existe
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No existe: {full_path}")
    return full_path


def preprocesar_audio(y, sr, trim_top_db=20, preemph_coef=0.97, norm_amplitude=True):
    """
    Aplica los siguientes pasos de preprocesamiento a una señal de audio:
    1. Eliminación de silencio al inicio y fin (trim).
    2. Normalización de amplitud al pico máximo.
    3. Preénfasis para realzar altas frecuencias (comunmente la voz).
    """
    try:
        # Recorta silencios según un umbral en dB
        y_trim, _ = librosa.effects.trim(y, top_db=trim_top_db)
    except Exception:
        # Si falla el trim, se usa el audio original
        y_trim = y
    y = y_trim

    if norm_amplitude and np.any(y):
        # Normalizamos la ampllitud para que el valor máximo absoluto sea 1
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

    if preemph_coef is not None and preemph_coef > 0:
        # Aplicar filtro de preénfasis: y[n] = x[n] - coef * x[n-1]
        y = np.append(y[0], y[1:] - preemph_coef * y[:-1])

    return y


def cargar_mfcc_desde_carpetas(carpeta_hola_rel, carpeta_adios_rel, sr=16000, n_mfcc=13):
    """
    Carga archivos de audio de dos carpetas ("hola" y "adios"),
    los preprocesa y extrae características MFCC.
    Devuelve un dict con listas de secuencias MFCC para cada palabra.
    """
    datos = {"hola": [], "adios": []}

    # Recorre ambas etiquetas y sus carpetas
    for palabra, carpeta_rel in [("hola", carpeta_hola_rel), ("adios", carpeta_adios_rel)]:
        carpeta = get_path_relative_to_script(carpeta_rel)
        if not os.path.isdir(carpeta):
            raise NotADirectoryError(f"No es carpeta: {carpeta}")

        # Procesa cada archivo de audio
        for fname in os.listdir(carpeta):
            if not fname.lower().endswith(('mp3')):
                continue
            ruta = os.path.join(carpeta, fname)
            try:
                # Carga el audio con tasa sr fija
                y, _ = librosa.load(ruta, sr=sr)
                # Preprocesa el audio (trim + norm + preénfasis)
                y_proc = preprocesar_audio(y, sr)
                # Extrae MFCC y transpone: frames x coeficientes
                mfcc = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=n_mfcc)
                X = mfcc.T
                # Ignora audios demasiado cortos (<3 frames)
                if X.shape[0] < 3:
                    print(f"Audio demasiado corto: {ruta}")
                    continue
                datos[palabra].append(X)
            except Exception as e:
                print(f"Advertencia al procesar {ruta}: {e}")

    # Verifica que se haya cargado al menos un archivo por etiqueta
    for palabra, listas in datos.items():
        if len(listas) == 0:
            raise ValueError(f"No se cargó ningún MFCC para '{palabra}'")

    return datos


def normalizar_datos(datos_dict):
    """
    Ajusta un StandardScaler global a todas las secuencias de entrenamiento
    y transforma cada secuencia con la media y desviación estándar calculadas.
    Devuelve el dict normalizado y el scaler entrenado.
    Se utiliza para que los valores mas grandes no predominen el aprendizaje
    """
    # Apilar todas las características para aprendizaje de escala
    all_feats = np.vstack([X for listas in datos_dict.values() for X in listas])
    scaler = StandardScaler().fit(all_feats)
    datos_norm = {}

    # Transformar cada secuencia de cada etiqueta
    for palabra, listas in datos_dict.items():
        datos_norm[palabra] = [scaler.transform(X) for X in listas]

    return datos_norm, scaler


def entrenar_modelos(datos_entrenamiento, n_components=3, covariance_type="diag", n_iter=1000):
    """
    Entrena un modelo HMM Gaussiano para cada etiqueta (palabra)
    sobre las secuencias de MFCC proporcionadas.
    Devuelve un dict de modelos entrenados.
    """
    modelos = {}
    for palabra, secuencias in datos_entrenamiento.items():
        # Concatenar secuencias y definir longitudes para HMM
        X = np.vstack(secuencias)
        lengths = [len(s) for s in secuencias]
        modelo = hmm.GaussianHMM(n_components=n_components,
                                 covariance_type=covariance_type,
                                 n_iter=n_iter)
        # Ajustar modelo
        modelo.fit(X, lengths=lengths)
        modelos[palabra] = modelo
    return modelos


def evaluar_individuo_hmm_test_unico(individuo, datos_norm, X_test,
                                      true_label, n_repeats=10):
    """
    Evalúa un individuo en el espacio de parámetros HMM [n_components, n_iter]
    usando una única muestra de prueba X_test. Se repite entrenamiento y prueba
    n_repeats veces para calcular accuracy promedio.

    Devuelve 1 - accuracy para usar como función de fitness (a minimizar).
    
    """
    # Se redondean los valores de los genotipos como enteros válidos
    n_components = max(1, int(round(individuo[0])))
    n_iter = max(1, int(round(individuo[1])))
    clases = list(datos_norm.keys())
    aciertos = 0

    for _ in range(n_repeats):
        modelos = {}
        fallo = False
        # Entrenamos un HMM por clase
        for c in clases:
            seqs = datos_norm[c]
            X_all = np.vstack(seqs)
            lengths = [len(s) for s in seqs]
            try:
                modelo = hmm.GaussianHMM(n_components=n_components,
                                         covariance_type="diag",
                                         n_iter=n_iter)
                modelo.fit(X_all, lengths=lengths)
                modelos[c] = modelo
            except Exception:
                fallo = True
                break
        if fallo:
            # Si falla alguna etapa, omitir repetición
            continue

        # Calcular score de similitud para cada modelo
        scores = {}
        for c, mod in modelos.items():
            try:
                scores[c] = mod.score(X_test)
            except Exception:
                scores[c] = -np.inf

        # Predecir etiqueta con mayor score
        pred = max(scores, key=scores.get)
        if pred == true_label:
            aciertos += 1

    acc_prom = aciertos / n_repeats
    return 1.0 - acc_prom


def ga_optimize_hmm_un_audio(datos_norm, X_test, true_label,
                             rango_n_components, rango_n_iter,
                             num_generaciones, num_pob, Pc, Pm,
                             n_repeats_eval=10,
                             random_seed=None, verbose=True):
    """
    Aplica un Algoritmo Genético (GA) para optimizar los parámetros [n_components, n_iter]
    de un HMM, evaluando en una muestra de prueba.

    Parámetros:
      - rango_n_components: tupla (min, max) para n_components
      - rango_n_iter: tupla (min, max) para n_iter
      - num_generaciones: número de generaciones del GA
      - num_pob: tamaño de población
      - Pc: probabilidad de cruce
      - Pm: probabilidad de mutación

    Devuelve diccionario con mejor solución, fitness y evolución histórica.
    """
    # Configurar generador aleatorio reproducible
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    li = np.array([rango_n_components[0], rango_n_iter[0]], dtype=float)
    ls = np.array([rango_n_components[1], rango_n_iter[1]], dtype=float)
    num_var = 2

    # Inicializar población con valores uniformes en los rangos
    poblacion = rng.uniform(low=li, high=ls, size=(num_pob, num_var))
    fitness = np.zeros(num_pob)

    # Evaluamos la población inicial
    for i in range(num_pob):
        fitness[i] = evaluar_individuo_hmm_test_unico(
            poblacion[i], datos_norm, X_test, true_label,
            n_repeats=n_repeats_eval
        )

    history = []
    idx_best = np.argmin(fitness)
    best_ind = poblacion[idx_best].copy()
    best_fitness = fitness[idx_best]
    best_acc = 1.0 - best_fitness
    history.append(best_acc)

    if verbose:
        bc0, bc1 = map(lambda x: max(1, int(round(x))), best_ind)
        print(f"Generación 0: mejor accuracy ≈ {best_acc:.4f} con (n_components={bc0}, n_iter={bc1})")

    # Parámetros para operadores SBX y mutación polinómica
    Nc = 20  # distribución de cruce
    Nm = 100  # distribución de mutación

    # Bucle de generaciones
    for gen in range(1, num_generaciones + 1):
        if verbose:
            print(f"Generación {gen}:")
            for i in range(num_pob):
                nc, ni = map(lambda x: max(1, int(round(x))), poblacion[i])
                print(f"  Individuo {i}: (n_components={nc}, n_iter={ni}), fitness ≈ {fitness[i]:.4f}")

        # Selección por torneo binario
        padres = np.zeros_like(poblacion)
        for i in range(num_pob):
            a, b = rng.choice(num_pob, size=2, replace=False)
            padres[i] = poblacion[a] if fitness[a] < fitness[b] else poblacion[b]

        # Cruce SBX
        hijos = np.zeros_like(padres)
        for i in range(0, num_pob, 2):
            if i + 1 >= num_pob:
                hijos[i] = padres[i]
                break
            if rng.rand() <= Pc:
                p1, p2 = padres[i], padres[i + 1]
                c1, c2 = p1.copy(), p2.copy()
                for j in range(num_var):
                    if abs(p2[j] - p1[j]) < 1e-14:
                        continue
                    low, high = (p1[j], p2[j]) if p1[j] < p2[j] else (p2[j], p1[j])
                    delta = high - low
                    beta = 1 + (2.0 / delta) * min(low - li[j], ls[j] - high)
                    alpha = 2.0 - beta ** (-(Nc + 1))
                    U = rng.rand()
                    beta_c = (U * alpha) ** (1.0/(Nc + 1)) if U <= 1.0/alpha else (1.0/(2.0 - U * alpha)) ** (1.0/(Nc + 1))
                    # Generar dos nuevos valores
                    val1 = 0.5 * (p1[j] + p2[j] - beta_c * delta)
                    val2 = 0.5 * (p1[j] + p2[j] + beta_c * delta)
                    c1[j] = np.clip(val1, li[j], ls[j])
                    c2[j] = np.clip(val2, li[j], ls[j])
                hijos[i], hijos[i + 1] = c1, c2
            else:
                hijos[i], hijos[i + 1] = padres[i], padres[i + 1]

        # Mutación polinómica
        for i in range(num_pob):
            for j in range(num_var):
                if rng.rand() <= Pm:
                    delta = min(hijos[i, j] - li[j], ls[j] - hijos[i, j]) / (ls[j] - li[j])
                    r = rng.rand()
                    if r <= 0.5:
                        deltaq = (2 * r + (1 - 2 * r) * (1 - delta) ** (Nm + 1)) ** (1.0/(Nm + 1)) - 1
                    else:
                        deltaq = 1 - (2 * (1 - r) + 2 * (r - 0.5) * (1 - delta) ** (Nm + 1)) ** (1.0/(Nm + 1))
                    hijos[i, j] += deltaq * (ls[j] - li[j])
                    hijos[i, j] = np.clip(hijos[i, j], li[j], ls[j])

        # Evaluar nuevos individuos
        fitness_hijos = np.zeros(num_pob)
        for i in range(num_pob):
            fitness_hijos[i] = evaluar_individuo_hmm_test_unico(
                hijos[i], datos_norm, X_test, true_label,
                n_repeats=n_repeats_eval
            )

        # Reemplazo elitista: conservar el mejor padre y a todos los demas hijos excepto el peor
        idx_mejor_padre = np.argmin(fitness)
        mejor_padre = poblacion[idx_mejor_padre].copy()
        idx_hijos_orden = np.argsort(fitness_hijos)
        n = num_pob - 1
        mejores_hijos = hijos[idx_hijos_orden][:n]
        fitness_mejores_hijos = fitness_hijos[idx_hijos_orden][:n]

        poblacion = np.vstack((mejor_padre[np.newaxis, :], mejores_hijos))
        fitness = np.hstack((fitness[idx_mejor_padre], fitness_mejores_hijos))

        # Actualizar mejor global
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_ind = poblacion[0].copy()
            best_acc = 1.0 - best_fitness
        history.append(best_acc)

        if verbose:
            bc0, bc1 = map(lambda x: max(1, int(round(x))), best_ind)
            print(f"  → Mejor hasta ahora accuracy ≈ {best_acc:.4f} con (n_components={bc0}, n_iter={bc1})")

    # Devolver parámetros óptimos y trayectoria de accuracy
    best_n_components, best_n_iter = map(lambda x: max(1, int(round(x))), best_ind)
    return {
        "best_params": (best_n_components, best_n_iter),
        "best_fitness": best_fitness,
        "best_accuracy": 1.0 - best_fitness,
        "history": history
    }


if __name__ == "__main__":
    # Definimos las carpetas de entrenamiento y etiquetas de prueba
    carpeta_hola = "Hola_mp3"
    carpeta_adios = "Adios_mp3"
    datos = cargar_mfcc_desde_carpetas(carpeta_hola, carpeta_adios)
    datos_norm, scaler = normalizar_datos(datos)

    test_file = "Test.mp3"
    true_label = "hola"
    
    # Cargar y preprocesar audio de prueba
    audio_path = get_path_relative_to_script(test_file)
    y, _ = librosa.load(audio_path, sr=16000)
    y_proc = preprocesar_audio(y, 16000)
    mfcc = librosa.feature.mfcc(y=y_proc, sr=16000, n_mfcc=13)
    X_test = mfcc.T
    if X_test.shape[0] >= 3:
        # Normalizar secuencia de prueba
        X_test = scaler.transform(X_test)
    else:
        raise ValueError("Audio de prueba demasiado corto")

    # Parámetros GA
    rango_n_components = (1, 10)
    rango_n_iter = (50, 1000)
    num_generaciones = 10
    num_pob = 10
    Pc = 0.8  # probabilidad de cruce
    Pm = 0.1  # probabilidad de mutación
    random_seed = None
    n_repeats_eval = 10

    # Ejecutar GA
    resultado_ga = ga_optimize_hmm_un_audio(
        datos_norm, X_test, true_label,
        rango_n_components, rango_n_iter,
        num_generaciones, num_pob, Pc, Pm,
        n_repeats_eval=n_repeats_eval,
        random_seed=random_seed,
        verbose=True
    )

    # Mostrar resultados finales
    print("\n=== Resultado GA ===")
    print(f"Mejor n_components: {resultado_ga['best_params'][0]}")
    print(f"Mejor n_iter: {resultado_ga['best_params'][1]}")
    print(f"Mejor accuracy promedio en {n_repeats_eval} repeticiones: {resultado_ga['best_accuracy']:.4f}")

    # Graficar evolución de la mejor accuracy
    plt.figure()
    plt.plot(resultado_ga["history"], marker='o')
    plt.xlabel("Generación")
    plt.ylabel("Mejor accuracy promedio")
    plt.title("Evolución GA")
    plt.grid(True)
    plt.show()
