
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import Tuple, Dict


def distancia_euclidiana(punto1, punto2):
  """
  Calcula la distancia euclidiana entre dos puntos utilizando NumPy.

  Args:
    punto1: Un array de NumPy que representa el primer punto.
    punto2: Un array de NumPy que representa el segundo punto.

  Returns:
    La distancia euclidiana entre los dos puntos.
  """
  return np.linalg.norm(np.array(punto1) - np.array(punto2))




def kmeans_clustering_sklearn(n_clusters, clientes):
    '''
    Implementación de K-means usando scikit-learn, adaptada para trabajar
    con diccionarios de coordenadas de clientes.
    
    Args:
        n_clusters (int): Número de clusters a formar
        clientes (dict): Diccionario con formato {id_cliente: {'x': coord_x, 'y': coord_y}}
    
    Returns:
        dict: Diccionario con formato {id_cluster: [lista_de_ids_de_clientes]}
    '''
    # Convertir el diccionario de clientes a una matriz numpy
    cliente_ids = list(clientes.keys())
    X = np.array([[clientes[id_cliente]['x'], clientes[id_cliente]['y']]  for id_cliente in cliente_ids])
    
    # Prevenir error ValueError si hay más clusters que muestras (vehículos > clientes)
    n_clusters_seguro = min(n_clusters, len(cliente_ids))
    # Crear y ajustar el modelo K-means
    kmeans = KMeans(
        n_clusters=n_clusters_seguro,
        init='k-means++',  # Usa k-means++ para mejor inicialización
        n_init=10,         # Número de veces que se ejecutará con diferentes semillas
        random_state=42    # Para reproducibilidad
    )
    
    # Ajustar el modelo y obtener las etiquetas
    etiquetas = kmeans.fit_predict(X)
    
    # Convertir las etiquetas al formato deseado
    clusters = {i+1: [] for i in range(n_clusters)}
    for idx, etiqueta in enumerate(etiquetas):
        clusters[etiqueta + 1].append(cliente_ids[idx])
    
    return clusters


# No se usa
def kmeans_clustering(n_clusters, clientes):
    '''
    Descripción:
    Esta función implementa el algoritmo K-means para agrupar clientes en clusters.
    El algoritmo funciona en 4 pasos principales:
    1. Inicialización de centroides aleatorios
    2. Asignación de puntos al centroide más cercano
    3. Actualización de centroides
    4. Repetición hasta convergencia

    Args:
        n_clusters (int): El número de clusters a formar
        clientes (dict): Diccionario con formato {id_cliente: {'x': coord_x, 'y': coord_y}}
    
    Returns:
        dict: Diccionario con formato {id_cluster: [lista_de_ids_de_clientes]}
    '''

    random.seed(1)  # Semilla para reproducibilidad
    # PASO 1: INICIALIZACIÓN
    # Crear diccionario para almacenar los centroides
    centroides = {}
    # Extraer todas las coordenadas para encontrar los límites del espacio
    coordenadas = [(cliente['x'], cliente['y']) for cliente in clientes.values()]
    # Encontrar los límites mínimos y máximos para x e y
    x_min, x_max = min(x for x, _ in coordenadas), max(x for x, _ in coordenadas)
    y_min, y_max = min(y for _, y in coordenadas), max(y for _, y in coordenadas)
    
    # Inicializar centroides de manera aleatoria dentro de los límites del espacio
    for i in range(n_clusters):
        centroides[i] = {
            'x': random.uniform(x_min, x_max),  # Valor aleatorio entre x_min y x_max
            'y': random.uniform(y_min, y_max)   # Valor aleatorio entre y_min y y_max
        }
    
    # Variables de control para el algoritmo
    max_iteraciones = 100  # Límite máximo de iteraciones para evitar bucles infinitos
    convergencia = False   # Bandera que indica si el algoritmo ha convergido
    iteracion = 0         # Contador de iteraciones
    
    # Inicializar diccionario para almacenar la asignación de clientes a clusters
    clusters = {i: [] for i in range(n_clusters)}
    
    # PASO 2: ITERACIÓN PRINCIPAL
    while not convergencia and iteracion < max_iteraciones:
        # Reiniciar clusters en cada iteración
        clusters = {i: [] for i in range(n_clusters)}
        
        # PASO 2.1: ASIGNACIÓN DE PUNTOS
        # Recorrer cada cliente para asignarlo al centroide más cercano
        for id_cliente, coords in clientes.items():
            distancias = []
            # Calcular la distancia euclidiana a cada centroide
            for id_centroide, centroide in centroides.items():
                # Fórmula de distancia euclidiana: sqrt((x2-x1)^2 + (y2-y1)^2)
                dist = math.sqrt(
                    (coords['x'] - centroide['x'])**2 + 
                    (coords['y'] - centroide['y'])**2
                )
                distancias.append((dist, id_centroide))
            
            # Asignar el cliente al cluster con la menor distancia
            cluster_asignado = min(distancias, key=lambda x: x[0])[1]
            clusters[cluster_asignado].append(id_cliente)
        
        # PASO 2.2: ACTUALIZACIÓN DE CENTROIDES
        nuevos_centroides = {}
        cambio = False  # Bandera para detectar si hubo cambios significativos
        
        # Recalcular la posición de cada centroide
        for id_cluster, clientes_cluster in clusters.items():
            # Si el cluster está vacío, mantener el centroide actual
            if not clientes_cluster:
                nuevos_centroides[id_cluster] = centroides[id_cluster]
                continue
            
            # Calcular la media de las coordenadas x e y de todos los puntos en el cluster
            sum_x = sum(clientes[cliente_id]['x'] for cliente_id in clientes_cluster)
            sum_y = sum(clientes[cliente_id]['y'] for cliente_id in clientes_cluster)
            n_clientes = len(clientes_cluster)
            
            # Calcular las nuevas coordenadas del centroide
            nuevo_x = sum_x / n_clientes
            nuevo_y = sum_y / n_clientes
            
            nuevos_centroides[id_cluster] = {'x': nuevo_x, 'y': nuevo_y}
            
            # PASO 2.3: VERIFICACIÓN DE CONVERGENCIA
            # Comprobar si el centroide se movió más que el umbral de tolerancia (0.0001)
            if (abs(nuevo_x - centroides[id_cluster]['x']) > 0.0001 or 
                abs(nuevo_y - centroides[id_cluster]['y']) > 0.0001):
                cambio = True
        
        # Actualizar los centroides para la siguiente iteración
        centroides = nuevos_centroides
        
        # Si no hubo cambios significativos, el algoritmo ha convergido
        if not cambio:
            convergencia = True
        
        iteracion += 1
    
    # PASO 3: AJUSTE FINAL
    # Ajustar los índices de los clusters para que empiecen desde 1 en lugar de 0
    clusters_ajustados = {i+1: clientes_cluster for i, clientes_cluster in clusters.items()}
    
    return clusters_ajustados



def visualizar_clusters(clientes, clusters, depot):
    '''
    Descripción:
    Esta función visualiza los clusters generados por el algoritmo de agrupamiento K-Means.
    Cada cluster se representa con un color diferente y los puntos se muestran como círculos.
    Args:
        clientes (dict): Diccionario con formato {id_cliente: {'x': coordenada_x, 'y': coordenada_y}}
        clusters (dict): Diccionario con formato {id_cluster: [lista_de_ids_de_clientes]}
        depot (dict): Diccionario con formato {'x': coordenada_x, 'y': coordenada_y}}
        rns:
        None
    '''
    # Colores para cada cluster
    colores = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # Crear figura
    plt.figure(figsize=(10, 6))
    
    # Graficar puntos de cada cluster
    for id_cluster, clientes_cluster in clusters.items():
        # Obtener coordenadas x, y de los clientes en este cluster
        x = [clientes[cliente_id]['x'] for cliente_id in clientes_cluster]
        y = [clientes[cliente_id]['y'] for cliente_id in clientes_cluster]
        
        # Graficar los puntos del cluster
        plt.scatter(x, y, c=colores[(id_cluster-1) % len(colores)], label=f'Vehículo {id_cluster}')

        plt.scatter(depot['x'], depot['y'], c='black', marker='o')

        # Añadir etiquetas a los puntos
        for i, cliente_id in enumerate(clientes_cluster):
            plt.annotate(f'Cliente {cliente_id}', (x[i], y[i]), 
                        xytext=(5, 5), textcoords='offset points')
    
    plt.title('Visualización de Clusters K-means')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.legend()
    plt.grid(True)
    plt.show()



def entregar_planificacion_segun_demanda(planificacion, capacidad, demandas, productos):
    # Calcular la demanda total para los productos en la planificación
    total_demanda = sum(demandas[id] for id in planificacion.keys())
    
    # Crear una lista de productos con sus pesos de demanda negativos para orden descendente
    productos_ordenados = []
    for id_producto in planificacion:
        if id_producto not in demandas or id_producto not in productos:
            continue  # Ignorar productos sin demanda o sin información de volumen
        peso_demanda = demandas[id_producto] / total_demanda
        productos_ordenados.append((-peso_demanda, id_producto))
    
    # Ordenar los productos por prioridad (mayor peso primero)
    productos_ordenados.sort()
    
    remaining = capacidad
    allocated = {id: 0 for id in planificacion}
    
    for peso_neg, id_producto in productos_ordenados:
        producto = productos[id_producto]
        vol = producto.peso
        max_unidades = min(planificacion[id_producto], remaining // vol)
        allocated[id_producto] = max_unidades
        remaining -= max_unidades * vol
        if remaining <= 0:
            break  # No hay más capacidad disponible
    
    return allocated


def fourier_transform_features_vectorized(features, max_i=5):
    """
    CORREGIDA: Expansión de Fourier 1D (Base de Cosenos y Senos).
    Aplica la fórmula: cos(pi * i * x) y sin(pi * i * x).
    
    Args:
        features: np.array, shape (n_features,)
        max_i: int, orden de la aproximación (n)
        
    Return:
        np.array, shape (n_features * 2 * max_i,)
    """
    # 1. Crear array de índices i = [1, 2, ..., max_i]
    i_vals = np.arange(1, max_i + 1).reshape(-1, 1)  # shape (max_i, 1)

    # 2. Expandir cada feature: shape (max_i, n_features)
    # CORRECCIÓN MATEMÁTICA: Se agrega np.pi al argumento
    # Argumento = pi * i * x
    x_expanded = np.pi * i_vals * features.reshape(1, -1)

    # 3. Calcular coseno y seno en forma vectorizada
    cos_vals = np.cos(x_expanded) 
    sin_vals = np.sin(x_expanded)

    # 4. Intercalamos cos y sin para cada i, por cada feature
    # 'F' (Fortran order) asegura que queden agrupados por feature: 
    # [cos(pi*x1), sin(pi*x1), cos(2pi*x1)..., cos(pi*x2)...]
    combined = np.vstack([cos_vals, sin_vals]).reshape(-1, order='F')

    return combined

def fourier_transform_interactions_vectorized(feature_1, feature_2, max_i1=5, max_i2=5):
    """
    NUEVA: Expansión de Fourier 2D para Interacciones (Efectos Cruzados).
    Implementa la Ec. 7.6 de tu tesis: cos(pi*(i*x + j*y)) y sin...
    
    Úsala para pares de variables acopladas (ej: Inventario y Entrega).
    
    Args:
        feature_1: float o np.array, primera variable (x) normalizada
        feature_2: float o np.array, segunda variable (y) normalizada
        max_i1: int, orden para la variable 1 (n1)
        max_i2: int, orden para la variable 2 (n2)
    """
    features_list = []
    
    # Generamos la grilla de combinaciones i, j
    # Empezamos desde 1 para evitar duplicar el bias o términos 1D puros si ya los incluiste aparte
    # Según Lagos (2025), la interacción es sumatoria doble completa.
    
    for i in range(1, max_i1 + 1):
        for j in range(1, max_i2 + 1):
            # Argumento acoplado: pi * (i*x + j*y)
            arg = np.pi * (i * feature_1 + j * feature_2)
            
            features_list.append(np.cos(arg))
            features_list.append(np.sin(arg))
            
    return np.array(features_list, dtype=np.float32)

def identificar_extremos_impacto(instancia) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Identifica los pares (Cliente, Producto) con el mayor y menor impacto esperado.
    El impacto se calcula como: Costo Penalización * Demanda Media.
    
    Esta métrica ayuda a identificar qué clientes/productos son críticos (alto costo y alta demanda)
    y cuáles son marginales.

    Args:
        instancia: Objeto de la clase Instancia que contiene clientes y productos.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: 
            - Primer elemento: (id_cliente, id_producto) del MAYOR impacto.
            - Segundo elemento: (id_cliente, id_producto) del MENOR impacto.
    """
    # Inicializamos valores extremos para búsqueda
    max_impacto = -1.0
    min_impacto = float('inf')
    
    # Índices resultantes: (id_cliente, id_producto)
    indices_max: Tuple[int, int] = (-1, -1)
    indices_min: Tuple[int, int] = (-1, -1)

    # Iteramos sobre todos los clientes
    for id_cliente, cliente in instancia.clientes.items():
        
        # Iteramos sobre todos los productos que maneja este cliente
        # Asumimos que las llaves de demanda_media y costos_penalizacion coinciden con los productos
        for id_producto in instancia.productos.keys():
            
            # Validación de seguridad: asegurarse que el producto existe para el cliente
            if id_producto not in cliente.costos_penalizacion or id_producto not in cliente.demanda_media:
                continue

            # Extracción de valores
            float_costo_penalizacion = cliente.costos_penalizacion[id_producto]
            float_demanda_media = cliente.demanda_media[id_producto]

            # Cálculo de la métrica solicitada (Multiplicación)
            impacto_calculado = float_costo_penalizacion * float_demanda_media

            # Actualización de Máximo
            if impacto_calculado > max_impacto:
                max_impacto = impacto_calculado
                indices_max = (id_cliente, id_producto)

            # Actualización de Mínimo
            if impacto_calculado < min_impacto:
                min_impacto = impacto_calculado
                indices_min = (id_cliente, id_producto)

    # Reporte en consola para verificación inmediata
    print(f"--- Análisis de Impacto (Costo * Demanda) ---")
    print(f"Mayor Impacto ({max_impacto:.2f}): Cliente {indices_max[0]}, Producto {indices_max[1]}")
    print(f"Menor Impacto ({min_impacto:.2f}): Cliente {indices_min[0]}, Producto {indices_min[1]}")

    return indices_max, indices_min
