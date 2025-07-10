
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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
    # Crear y ajustar el modelo K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
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




# optimizado
def fourier_transform_features_vectorized( features, max_i=5):
    """
        Recibe un array de features (sin intercepto) y devuelve un array
    con las transformaciones de Fourier vectorizadas: 
    cos(i * x) y sin(i * x) para i = 1..max_i.
        
    Args:
        features: np.array, shape (n_features,)
        max_i: int, máximo valor de i en la transformación
        
    Return:
        np.array, shape (n_features * 2 * max_i,)
    """
    # Crear array de índices i = [1, 2, ..., max_i]
    i_vals = np.arange(1, max_i + 1).reshape(-1, 1)  # shape (max_i, 1)

    # Expandir cada feature: shape (max_i, n_features)
    # Cada columna corresponde a i * feature_j
    x_expanded = i_vals * features.reshape(1, -1)

    # Calcular coseno y seno en forma vectorizada
    cos_vals = np.cos(x_expanded)  # shape (max_i, n_features)
    sin_vals = np.sin(x_expanded)  # shape (max_i, n_features)

    # Intercalamos cos y sin para cada i, por cada feature
    # Primero aplanamos en columna
    combined = np.vstack([cos_vals, sin_vals]).reshape(-1, order='F')

    return combined
    
