from Estado import Estado
import numpy as np
from FuncionesAuxiliares import distancia_euclidiana, kmeans_clustering_sklearn, fourier_transform_features_vectorized
from abc import ABC, abstractmethod
import copy
#import tensorflow as tf
from collections import deque
import random



class Politica:
    '''
    Clase padre que implementa la politica de decisiones y ejecuta la simulación del Proceso
    '''
    def __init__(self, instancia, proceso):
        self.instancia = instancia
        self.proceso = proceso

    @abstractmethod
    def tomar_accion(self, estado: Estado):
        '''
        Método que toma una acción en base al estado actual del problema
        '''
        pass

    def run(self):
        '''
        Método que ejecuta la política
        '''
        estado = self.proceso.determinar_estado_inicial() # Se recupera el estado inicial de la instancia
        trayectoria = [] # Lista que almacena los estado-accion-recompensas tomadas a lo largo de la simulación, serán guardados como diccionarios 
        costo_traslado = 0
        costos_de_insatisfecha = 0
        costo_total = 0
        # Se ejecuta hasta que se encuentre un estado terminal
        while True:
            accion = self.tomar_accion(estado)
            estado_nuevo, recompensa, costos_de_traslado, costos_de_demanda_insatisfecha = self.proceso.transicion(estado, accion)
            trayectoria.append({'estado': estado, 'accion': accion, 'recompensa': recompensa})
            costo_traslado += costos_de_traslado
            costos_de_insatisfecha += costos_de_demanda_insatisfecha
            costo_total += recompensa
            if self.es_terminal(estado):
                break
            else:
                estado = estado_nuevo
        return trayectoria , costo_traslado/(costo_total+1), costos_de_insatisfecha/(costo_total+1)

    def es_terminal(self, estado: Estado):
        '''
        Método que determina si el estado es terminal
        Parámetros: 
        - estado: Estado actual del problema
        Retorna:
        - True: Si el estado es terminal
        - False: Si el estado no es terminal
        '''
        if estado.tiempo >= self.instancia.horizonte_tiempo:
            return True
        else:
            return False

    def planificar_entrega(self, inventario, demandas):
        """
        Calcula un plan de entrega de productos basado en la disponibilidad del inventario y la demanda relativa de cada producto.
        
        La función distribuye el inventario total de manera proporcional a la demanda esperada de cada producto.
        
        Parámetros:
        - inventario (dict): Diccionario donde las claves son los IDs de los productos y los valores la cantidad disponible en el vehículo.
        - demandas (dict): Diccionario donde las claves son los IDs de los productos y los valores representan la demanda promedio esperada.
        
        Retorna:
        - dict: Un diccionario con la planificación de entrega de cada producto basada en la disponibilidad y la proporción de demanda.
        """
        total_demanda = sum(demandas.values())
        total_inventario = sum(inventario.values())
        
        if not total_demanda or not total_inventario:
            return dict.fromkeys(demandas, 0)
        
        ratio = total_inventario / total_demanda
        return {
            producto: min(round(demanda * ratio), inventario.get(producto, 0))
            for producto, demanda in demandas.items()
        }

class PoliticasSimples(Politica):
    '''
    Clase Padre que implementa las politicas de decisiones y ejecuta la simulación utilizando las politica simple
    '''
    def __init__(self, instancia, proceso):
        self.instancia = instancia
        self.proceso = proceso


    def determinar_vehiculos_con_quiebre(self, estado: Estado) -> list:
        '''
        Método que determina los vehiculos que tienen un porcentaje de inventario menor a un umbral del 20% optimizado
        Parámetros:
            - estado: Estado actual del problema
        Retorna:
            Lista con los id de vehiculos con quiebre (list)
        '''
        # Precomputar datos necesarios fuera del bucle (optimización clave)
        productos = self.instancia.productos  # Diccionario de productos
        umbral = self.instancia.umbral_inventario_vehiculos  # Valor numérico (ej: 0.2)
        vehiculos_data = {
            idv: {
                'capacidad': v.capacidad,
                'inventario': estado.inventarios_vehiculos[idv]
            } for idv, v in self.instancia.vehiculos.items()
        }

        # List comprehension vectorizada + operaciones nativas
        return [
            idv for idv, data in vehiculos_data.items()
            if sum(
                cantidad * productos[idp].peso 
                for idp, cantidad in data['inventario'].items()
            ) < (umbral * data['capacidad'])
        ]

    def determinar_cliente_critico(self, estado: Estado, clientes_disponibles: list) -> int:
        '''
        Determina el cliente crítico usando vectorización con NumPy para mejorar rendimiento.
        
        Args:
            - estado: Estado actual del problema.
            - clientes_disponibles: Lista de clientes a evaluar.
            
        Return:
            ID del cliente con el menor ratio inventario/demanda entre aquellos con al menos un producto no negativo.
        '''
        # Precomputar datos clave como arrays NumPy
        productos = list(self.instancia.productos.keys())
        demandas = np.array([self.instancia.demandas_medias[idp] for idp in productos], dtype=np.float32)
        
        min_cliente = None
        min_ratio = np.inf
        
        for idc in clientes_disponibles:
            # Obtener inventario del cliente como array NumPy
            inventario = np.array([estado.inventarios_clientes[idc][idp] for idp in productos], dtype=np.float32)
            
            # Verificar si hay al menos un producto con inventario >= 0
            if not np.any(inventario >= 0):
                continue
            
            # Calcular ratios y obtener el mínimo
            ratios = np.divide(inventario, demandas, where=demandas != 0)  # Evita división por cero
            ratio_min = np.nanmin(ratios)  # Ignora NaNs (productos con demanda 0)
            
            if ratio_min < min_ratio:
                min_ratio = ratio_min
                min_cliente = idc
        
        return min_cliente

    def determinar_vehiculos_disponibles(self, estado: Estado):
        '''
        Método que devuelve los vehiculos que no tienen planificación
        
        Parámetros:
            - estado: Estado actual del problema

        Return: lista con ids de vehiculos disponibles
        '''
        # Determinamos los vehiculos que no tienen nada asignado
        planificacion = estado.planificacion  # Guardar referencia
        return [idv for idv in self.instancia.id_vehiculos if not planificacion[idv]]

    def determinar_clientes_disponibles(self, estado: Estado):
        '''
        Método que devuelve los clientes que no están asignados a ningún vehículo OPTIMIZADA

        Args:
            * estado: Estado actual del problema

        Return: lista con ids de clientes disponibles
        '''
        # Obtener todos los clientes asignados en la planificación
        clientes_asignados = set()
        for planificacion in estado.planificacion.values():
            clientes_asignados.update(planificacion.keys())

        # Clientes disponibles son los que no están asignados
        todos_los_clientes = set(self.instancia.clientes.keys())
        clientes_disponibles = list(todos_los_clientes - clientes_asignados)

        return clientes_disponibles
    
    def obtener_4_clientes_criticos(self, estado, clientes_disponibles):
        '''
        Descripción:
            Método que determina los 4 clientes con stock crítico considerando el peso de los productos OPTIMIZADA.
        Args:
            estado: Estado actual (Estado)
            clientes_disponibles (List): lista de ids de clientes disponibles
        Return:
            Lista con los ids de los 4 clientes más críticos
        '''
        
        # Lista para almacenar clientes con su criticidad
        clientes_criticidad = []

        for idc, inventario in estado.inventarios_clientes.items():
            if idc in clientes_disponibles:
                inventario_total_peso = sum(cantidad * self.instancia.productos[idp].peso 
                                            for idp, cantidad in inventario.items())
                clientes_criticidad.append((idc, inventario_total_peso/self.instancia.clientes[idc].capacidad_almacenamiento))

        # Ordenar por inventario más bajo (mayor criticidad)
        clientes_criticidad.sort(key=lambda x: x[1])

        # Seleccionar los 4 clientes más críticos
        return [idc for idc, _ in clientes_criticidad[:4]]

    def obtener_acciones(self, estado: Estado):
        '''
        Descripción:
        Método que devuelve una lista con las acciones posibles para un estado
        Genera todas las acciones posibles para el estado actual, optimizando:
        1. Reducción de llamadas redundantes a métodos.
        2. Uso de estructuras de datos eficientes (sets, generadores).
        3. Minimización de operaciones costosas.
        
        Args: 
            * estado: Objeto Estado
        Return:
            * acciones_posibles: Lista con las acciones factibles

        '''
        acciones_posibles = [{}]  # Acción nula

        # --- Sección 1: Acción de redirigir vehículos con quiebre de stock al depot ---
        vehiculos_disponibles = self.determinar_vehiculos_disponibles(estado)
        vehiculos_con_quiebre = set(self.determinar_vehiculos_con_quiebre(estado))  # Convertir a set para O(1) lookups
        
        # Encontrar el primer vehículo disponible con quiebre usando next() + generador (más eficiente que loop)
        vehiculo_quiebre = next((idv for idv in vehiculos_disponibles if idv in vehiculos_con_quiebre), None)
        
        if vehiculo_quiebre is not None:
            acciones_posibles.append({vehiculo_quiebre: {0: {}}})  # Acción de redirección

        # --- Sección 2: Acciones para clientes críticos ---
        clientes_disponibles = self.determinar_clientes_disponibles(estado)
        criticos = self.obtener_4_clientes_criticos(estado, clientes_disponibles)
        
        # Precomputar demandas de clientes para evitar accesos múltiples
        demandas_clientes = {idc: self.instancia.clientes[idc].demanda_media for idc in criticos}
        
        # Generar todas las acciones en una comprensión de lista (evita loops anidados)
        nuevas_acciones = [
            {idv: {id_cliente: self.planificar_entrega(
                estado.inventarios_vehiculos[idv], 
                demandas_clientes[id_cliente]
            )}}
            for id_cliente in criticos
            for idv in self.determinar_2_vehiculos_mas_cercanos_disponibles(
                estado, id_cliente, vehiculos_disponibles
            )
        ]
        
        acciones_posibles.extend(nuevas_acciones)
        return acciones_posibles

    def determinar_2_vehiculos_mas_cercanos_disponibles(self, estado, id_cliente, vehiculos_disponibles):
        '''
        Descripción:
            Método que determina los 2 vehículos más cercanos a un cliente.

        Args:
            * estado: Objeto estado
            * id_cliente: int
            * vehiculos_disponibles:  List

        Return:
            * vehiculos_mas_cercanos: Lista con los ids de los 2 vehículos más cercanos
        '''

        # Se obtiene la posición del cliente
        posicion_cliente = np.array([
            self.instancia.clientes[id_cliente].posicion_x,
            self.instancia.clientes[id_cliente].posicion_y
        ])

        # Lista para almacenar (id_vehiculo, distancia)
        distancias_vehiculos = []

        # Calcular la distancia de cada vehículo al cliente
        for idv in vehiculos_disponibles:
            coordenada_vehiculo = np.array([
                estado.posiciones_vehiculos[idv]['x'],
                estado.posiciones_vehiculos[idv]['y']
            ])

            # Calcular distancia euclidiana
            distancia = np.linalg.norm(posicion_cliente - coordenada_vehiculo)

            # Guardar la distancia junto con el ID del vehículo
            distancias_vehiculos.append((idv, distancia))

        # Ordenar vehículos por distancia (de menor a mayor)
        distancias_vehiculos.sort(key=lambda x: x[1])

        # Devolver los 2 vehículos más cercanos
        return [idv for idv, _ in distancias_vehiculos[:2]]




class PoliticaSimpleClusterisada(PoliticasSimples):
    '''
    Clase que implementa la politica de decisiones y ejecuta la simulación utilizando la politica simple clusterizada
    '''
    def __init__(self, instancia, proceso):
        self.instancia = instancia
        self.proceso = proceso
        self.asignaciones_k_means = self.aplicar_k_means()
    
    def aplicar_k_means(self):
        '''
        Método que aplica el algoritmo de k-means a los clientes
        '''
        
        # Inicializamos el númmero de clusters
        M = len(self.instancia.vehiculos.keys())
        #Obtenemos las posiciones de los clientes
        posiciones_clientes = {idc: {'x':cliente.posicion_x, 'y': cliente.posicion_y} for idc, cliente in self.instancia.clientes.items()}
        # devuelve un diccionario con la asignación de los clientes a los vehiculos. Ejemplo {1:[1,2,3], 2:[4,5,6], 3:[7,8,9]}
        asignacion = kmeans_clustering_sklearn(n_clusters=M, clientes=posiciones_clientes)
        return asignacion

    def tomar_accion(self, estado):
        '''
        Descripción: Método que toma una acción en base al estado actual del problema siguiendo la politica simple clusterisada 
        Parámetros:
        estado: Estado
        return: Accion
        '''
        # determinamos la asignación de los clientes a los vehiculos según k-means
        #asignaciones_k_means = self.aplicar_k_means() # dict que tiene llave el id_vehiculo y valor una lista de ids de clientes asignados
        accion = {} # diccionario que no tiene nada asignado de momento

        # Determinamos los vehiculos que no tienen nada asignado
        vehiculos_disponibles = self.determinar_vehiculos_disponibles(estado) # lista de ids de vehiculos sin planificación
        #Si no hay vehículos disponibles entonces no se toma ninguna acción
        if vehiculos_disponibles ==  []:
            return accion
        # Determinamos los vehiculos con quiebres
        vehiculos_con_quiebre = self.determinar_vehiculos_con_quiebre(estado) # lista de ids de vehiculos con inventario bajo
        # asignamos una planificación a los vehiculos sin planificación
        # vemos los que tienen quiebre
        for idv in vehiculos_disponibles:
            # Si tiene quiebre, se asigna su ruta al depot
            if idv in vehiculos_con_quiebre:
                accion[idv] = {0 :{}} # su nueva ruta será a depot con planificación vacía
                return accion
        # Determinamos los clientes que no están asignados a ningún vehiculo
        clientes_disponibles = self.determinar_clientes_disponibles(estado) # lista de ids de clientes sin planificación

        for idv in vehiculos_disponibles:
                # Determinamos los clientes que se encuentran en el cluster del vehiculo
                clientes_disponibles_k_means = (elemento for elemento in clientes_disponibles if elemento in self.asignaciones_k_means[idv])
                if clientes_disponibles_k_means == (): # Si no quedan clientes, no se modifica la ruta y se queda 'quieto'
                    break
                # Si no tiene quiebre, se asigna su ruta al cliente crítico
                id_cliente_con_inventario_bajo = self.determinar_cliente_critico(estado, clientes_disponibles_k_means)
                planificacion_del_vehiculo = self.planificar_entrega(inventario = estado.inventarios_vehiculos[idv], demandas= self.instancia.clientes[id_cliente_con_inventario_bajo].demanda_media)
                # Una vez que se asignó la planificación, se agrega al diccionario acciones
                accion[idv] = {id_cliente_con_inventario_bajo : planificacion_del_vehiculo}
                return accion
        return accion

class PoliticaSimple(PoliticasSimples):
    '''
    Clase que implementa la politica de decisiones y ejecuta la simulación del Proceso sin clusters
    '''
    def __init__(self, instancia, proceso):
        super().__init__(instancia, proceso)


    def tomar_accion(self, estado: Estado):
        '''
        Método que toma una acción en base al estado actual del problema

        Retorna: 
        '''
        accion = {} # dict donde se guardarán las acciones que se tomarán
        # Determinamos los vehiculos que no tienen nada asignado
        vehiculos_disponibles = self.determinar_vehiculos_disponibles(estado) # lista de ids de vehiculos sin planificación
        #Si no hay vehículos disponibles entonces no se toma ninguna acción
        if vehiculos_disponibles ==  []:
            return accion
        # Determinamos los clientes que no están asignados
        clientes_disponibles = self.determinar_clientes_disponibles(estado) # lista de ids de clientes sin planificación
        # Determinamos los vehiculos que tengan un porcentaje de inventario menor a un umbral del 20%
        vehiculos_con_quiebre = self.determinar_vehiculos_con_quiebre(estado) # lista de ids de vehiculos con inventario bajo el umbral
        # asignamos una planificación a los vehiculos sin planificación
        for idv in vehiculos_disponibles:
            # Si tiene quiebre, se asigna su ruta al depot
            if idv in vehiculos_con_quiebre:
                accion[idv] = {0 :{}} # su nueva ruta será a depot con planificación vacía
                return accion
            
        for idv in vehiculos_disponibles:
                # Si no tiene quiebre, se asigna su ruta al cliente crítico
                id_cliente_con_inventario_bajo = self.determinar_cliente_critico(estado, clientes_disponibles)
                #if id_cliente_con_inventario_bajo == None: # Si no quedan clientes, no se modifica la ruta y se queda 'quieto'
                #    break
                planificacion_del_vehiculo = self.planificar_entrega(inventario = estado.inventarios_vehiculos[idv], demandas= self.instancia.clientes[id_cliente_con_inventario_bajo].demanda_media)
                # Se determina su planificación a ese cliente como entregar todo lo que tiene (PREGUNTAR COMO HACER ESTO)
                # Una vez que se asignó la planificación, se agrega al diccionario acciones
                accion[idv] = {id_cliente_con_inventario_bajo : planificacion_del_vehiculo}
                return accion 

class RollOutSimple(PoliticasSimples):
    '''
    Descripción: Clase que determina la política RollOut y implementa las acciones factibles para cada estado
    
    Args:
        * instancia: Objeto Instancia
        * proceso: Objeto Proceso
    '''

    def __init__(self, instancia, proceso):
        super().__init__(instancia, proceso)

    def tomar_accion(self, estado: Estado):
        '''
        Método que determina la mejor acción según la política RollOut.
        
        Args: 
            * estado: Objeto Estado

        Return:
            * Accion (Dict): La acción con el menor costo esperado.
        '''

        # Obtenemos una lista con las acciones posibles que se pueden tomar
        acciones_factibles = self.obtener_acciones(estado) 

        # Lista para almacenar (acción, costo promedio)
        valores_objetivo = []

        # Simular cada acción
        for accion in acciones_factibles:
            # Realizar 5 simulaciones y calcular el costo promedio
            recompensas_totales = [self.simular_episodio_rollout(accion, estado) for _ in range(2)] # 6 antes
            costo_promedio = np.mean(recompensas_totales)

            # Guardar la acción junto con su costo esperado
            valores_objetivo.append((accion, costo_promedio))

        # Seleccionar la acción con el menor costo promedio
        accion_seleccionada = min(valores_objetivo, key=lambda x: x[1])[0]

        return accion_seleccionada

    def tomar_accion_politica_simple(self, estado: Estado):
        '''
        Método que toma una acción en base al estado actual del problema

        Retorna: 
        '''
        accion = {} # dict donde se guardarán las acciones que se tomarán
        # Determinamos los vehiculos que no tienen nada asignado
        vehiculos_disponibles = self.determinar_vehiculos_disponibles(estado) # lista de ids de vehiculos sin planificación
        #Si no hay vehículos disponibles entonces no se toma ninguna acción
        if vehiculos_disponibles ==  []:
            return accion
        # Determinamos los vehiculos que tengan un porcentaje de inventario menor a un umbral del 20%
        vehiculos_con_quiebre = self.determinar_vehiculos_con_quiebre(estado) # lista de ids de vehiculos con inventario bajo el umbral
        # asignamos una planificación a los vehiculos sin planificación
        for idv in vehiculos_disponibles:
            # Si tiene quiebre, se asigna su ruta al depot
            if idv in vehiculos_con_quiebre:
                accion[idv] = {0 :{}} # su nueva ruta será a depot con planificación vacía
                return accion

        # Determinamos los clientes que no están asignados
        clientes_disponibles = self.determinar_clientes_disponibles(estado) # lista de ids de clientes sin planificación

        for idv in vehiculos_disponibles:
                # Si no tiene quiebre, se asigna su ruta al cliente crítico
                id_cliente_con_inventario_bajo = self.determinar_cliente_critico(estado, clientes_disponibles)
                #if id_cliente_con_inventario_bajo == None: # Si no quedan clientes, no se modifica la ruta y se queda 'quieto'
                 #   break
                planificacion_del_vehiculo = self.planificar_entrega(inventario = estado.inventarios_vehiculos[idv], demandas= self.instancia.clientes[id_cliente_con_inventario_bajo].demanda_media)
                # Se determina su planificación a ese cliente como entregar todo lo que tiene (PREGUNTAR COMO HACER ESTO)
                # Una vez que se asignó la planificación, se agrega al diccionario acciones
                accion[idv] = {id_cliente_con_inventario_bajo : planificacion_del_vehiculo}
                return accion 

    def simular_episodio_rollout(self, accion, estado = Estado):
        '''
        Método que ejecuta la política

        Args:
        *   accion: Dict
        *   estado: Objeto Estado
        Return:
        *   recompensa_total: float
        '''
        estado, recompensa, _ , _ = self.proceso.transicion(estado, accion) # Hacemos la transición inicial

        trayectoria = [] # Lista que almacena los estado-accion-recompensas tomadas a lo largo de la simulación, serán guardados como diccionarios 
        costo_traslado = 0
        costos_de_insatisfecha = 0
        costo_total = recompensa
        # Se ejecuta hasta que se encuentre un estado terminal
        while True:
            accion = self.tomar_accion_politica_simple(estado)
            estado_nuevo, recompensa, costos_de_traslado, costos_de_demanda_insatisfecha = self.proceso.transicion(estado, accion)
            trayectoria.append({'estado': estado, 'accion': accion, 'recompensa': recompensa})
            costo_traslado += costos_de_traslado
            costos_de_insatisfecha += costos_de_demanda_insatisfecha
            costo_total += recompensa
            if self.es_terminal(estado):
                break
            else:
                estado = estado_nuevo
        return costo_total

class RollOutCluster(PoliticasSimples):
    '''
    Descripción: Clase que determina la política RollOut y implementa las acciones factibles para cada estado
    
    Args:
        * instancia: Objeto Instancia
        * proceso: Objeto Proceso
    '''

    def __init__(self, instancia, proceso):
        super().__init__(instancia, proceso)
        self.asignaciones_k_means = self.aplicar_k_means()

    def tomar_accion(self, estado: Estado):
        '''
        Método que determina la mejor acción según la política RollOut.
        
        Args: 
            * estado: Objeto Estado

        Return:
            * Accion (Dict): La acción con el menor costo esperado.
        '''

        # Obtenemos una lista con las acciones posibles que se pueden tomar
        acciones_factibles = self.obtener_acciones(estado) 

        # Lista para almacenar (acción, costo promedio)
        valores_objetivo = []

        # Simular cada acción
        for accion in acciones_factibles:
            # Realizar 5 simulaciones y calcular el costo promedio
            recompensas_totales = np.array([self.simular_episodio_rollout(accion, estado) for _ in range(2)])
            costo_promedio = np.mean(recompensas_totales)

            # Guardar la acción junto con su costo esperado
            valores_objetivo.append((accion, costo_promedio))

        # Seleccionar la acción con el menor costo promedio
        accion_seleccionada = min(valores_objetivo, key=lambda x: x[1])[0]

        return accion_seleccionada

    def simular_episodio_rollout(self, accion, estado = Estado):
        '''
        Método que ejecuta la política

        Args:
        *   accion: Dict
        *   estado: Objeto Estado
        Return:
        *   recompensa_total: float
        '''
        estado, recompensa, _ , _ = self.proceso.transicion(estado, accion) # Hacemos la transición inicial

        trayectoria = [] # Lista que almacena los estado-accion-recompensas tomadas a lo largo de la simulación, serán guardados como diccionarios 
        costo_traslado = 0
        costos_de_insatisfecha = 0
        costo_total = recompensa
        # Se ejecuta hasta que se encuentre un estado terminal
        while True:
            accion = self.tomar_accion_politica_cluster(estado)
            estado_nuevo, recompensa, costos_de_traslado, costos_de_demanda_insatisfecha = self.proceso.transicion(estado, accion)
            trayectoria.append({'estado': estado, 'accion': accion, 'recompensa': recompensa})
            costo_traslado += costos_de_traslado
            costos_de_insatisfecha += costos_de_demanda_insatisfecha
            costo_total += recompensa
            if self.es_terminal(estado):
                break
            else:
                estado = estado_nuevo
        return costo_total

    def tomar_accion_politica_cluster(self, estado):
        '''
        Descripción: Método que toma una acción en base al estado actual del problema siguiendo la politica simple clusterisada 
        Parámetros:
        estado: Estado
        return: Accion
        '''
        # determinamos la asignación de los clientes a los vehiculos según k-means
        #asignaciones_k_means = self.aplicar_k_means() # dict que tiene llave el id_vehiculo y valor una lista de ids de clientes asignados
        accion = {} # diccionario que no tiene nada asignado de momento

        # Determinamos los vehiculos que no tienen nada asignado
        vehiculos_disponibles = self.determinar_vehiculos_disponibles(estado) # lista de ids de vehiculos sin planificación
        #Si no hay vehículos disponibles entonces no se toma ninguna acción
        if vehiculos_disponibles ==  []:
            return accion
        # Determinamos los vehiculos con quiebres
        vehiculos_con_quiebre = self.determinar_vehiculos_con_quiebre(estado) # lista de ids de vehiculos con inventario bajo
        # asignamos una planificación a los vehiculos sin planificación
        # vemos los que tienen quiebre
        for idv in vehiculos_disponibles:
            # Si tiene quiebre, se asigna su ruta al depot
            if idv in vehiculos_con_quiebre:
                accion[idv] = {0 :{}} # su nueva ruta será a depot con planificación vacía
                return accion
        # Determinamos los clientes que no están asignados a ningún vehiculo
        clientes_disponibles = self.determinar_clientes_disponibles(estado) # lista de ids de clientes sin planificación

        for idv in vehiculos_disponibles:
                # Determinamos los clientes que se encuentran en el cluster del vehiculo
                clientes_disponibles_k_means = (elemento for elemento in clientes_disponibles if elemento in self.asignaciones_k_means[idv])
                if clientes_disponibles_k_means == (): # Si no quedan clientes, no se modifica la ruta y se queda 'quieto'
                    break
                # Si no tiene quiebre, se asigna su ruta al cliente crítico
                id_cliente_con_inventario_bajo = self.determinar_cliente_critico(estado, clientes_disponibles_k_means)

                if id_cliente_con_inventario_bajo == None:
                    continue 
                else:
                    planificacion_del_vehiculo = self.planificar_entrega(inventario = estado.inventarios_vehiculos[idv], demandas= self.instancia.clientes[id_cliente_con_inventario_bajo].demanda_media)
                    accion[idv] = {id_cliente_con_inventario_bajo : planificacion_del_vehiculo}
                    return accion
        return accion

    def aplicar_k_means(self):
        '''
        Método que aplica el algoritmo de k-means a los clientes
        '''
        
        # Inicializamos el númmero de clusters
        M = len(self.instancia.vehiculos.keys())
        #Obtenemos las posiciones de los clientes
        posiciones_clientes = {idc: {'x':cliente.posicion_x, 'y': cliente.posicion_y} for idc, cliente in self.instancia.clientes.items()}
        # devuelve un diccionario con la asignación de los clientes a los vehiculos. Ejemplo {1:[1,2,3], 2:[4,5,6], 3:[7,8,9]}
        asignacion = kmeans_clustering_sklearn(n_clusters=M, clientes=posiciones_clientes)
        return asignacion


class Modelos_de_aproximacion(PoliticasSimples):
    '''
    Clase que implementa los modelos de aproximación utilizados en la política MC Onpolicy, heredado de PoliticasSimples
    ''' 
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate):
        super().__init__(instancia, proceso)
        self.episodios = episodios
        self.epsilon = epsilon
        #self.epsilon = 0.5
        #self.learning_rate = 0.01
        self.learning_rate = learning_rate
        self.n_episodio = 0

    '''
    def obtener_features(self, estado, accion):
            
        Descripción:
        Método que obtiene los features de un estado dado
        Args:
            * estado: Objeto Estado
            * accion: Diccionario con la acción a aplicar
        Return:
            * features: Lista con los features del estado
                

        # Precomputar estructuras necesarias
        clientes = self.instancia.clientes
        productos = self.instancia.productos
        vehiculos = estado.posiciones_vehiculos
        
        #creamos copia del estado
        estado_copia = copy.deepcopy(estado)
        # aplicamos la acción
        self.proceso.actualizar_planificacion(estado_copia, accion)
        planificacion_post = estado_copia.planificacion

        # 1. Precalcular posiciones y capacidades
        pos_clientes = {
            idc: (clientes[idc].posicion_x, clientes[idc].posicion_y)
            for idc in self.instancia.id_clientes
        }
        

        # 2. Inicializar lista de features (eficiente para append) |1|
        features_list = [1]  # Feature inicial beta_0
        
        # 3. Feature de tiempo restante |1| +|1|
        tiempo_norm = (self.instancia.horizonte_tiempo - estado.tiempo) / self.instancia.horizonte_tiempo
        features_list.append(tiempo_norm)
        features_list.append(tiempo_norm**2)  # Feature cuadrático del tiempo restante


        # 4. Features de inventario promedio por cliente-producto |N||P| y agregamos el cuadratico |N||P|*2
        ratio_inventario_clientes_productos = {}
        for idc in self.instancia.id_clientes:
            cliente = clientes[idc]
            ratio_inventario_clientes_productos[idc] = {}
            for idp in productos:
                ratio = (estado.inventarios_clientes[idc][idp] * productos[idp].peso) / cliente.capacidad_almacenamiento
                features_list.append(ratio)
                #features_list.append(ratio**2)  # Agregar cuadrático
                ratio_inventario_clientes_productos[idc][idp] = ratio
        



        # 5. Features de depot por vehículo-producto  |M||P|
        ratios_utilizacion = self.calcular_ratios_utilizacion_vehiculos(estado)
        for idv in self.instancia.id_vehiculos:
            for idp in productos:
                if not planificacion_post[idv]:
                    features_list.append(0.0)
                elif ratios_utilizacion[idv][idp] < 0.1 and next(iter(planificacion_post[idv])) == 0:
                    features_list.append(1.0)
                else:
                    features_list.append(0.0)
        


        # 6. Features combinados: entrega base + distancia  |N||P|4|
        # Para almacenar las features de entregas por cliente-producto
        feature_entregas = {}
        for idc in self.instancia.id_clientes:
            xc, yc = pos_clientes[idc]
            capacidad = clientes[idc].capacidad_almacenamiento
            feature_entregas[idc] = {}  # Inicializar lista de features para el cliente
            
            for idp in productos:
                ratio_actual = ratio_inventario_clientes_productos[idc][idp]
                if ratio_actual >= 0.2:  # debe ser si es menor que 0.2
                    # Agregar 4 ceros si no cumple el ratio
                    feature_entregas[idc][idp] = [0.0, 0.0, 0.0, 0.0, 0.0]
                    features_list.extend([0.0, 0.0 ,  0.0, 0.0, 0.0])
                    continue
                    
                # Calcular ratio base y vehículo asignado
                ratio_peso, idv = self.calcular_peso_por_cliente_producto(planificacion_post, idc, idp)
                ratio_base = ratio_peso / capacidad  # Feature base sin distancia
                
                if idv is None or idv not in vehiculos:
                    feature_entregas[idc][idp] = [0.0, 0.0, 0.0, 0.0, 0.0]
                    features_list.extend([0.0, 0.0, 0.0, 0.0, 0.0]) # antes era ratio base al principio
                    continue
                    
                # Calcular distancia una sola vez
                xv, yv = vehiculos[idv]['x'], vehiculos[idv]['y']
                distancia = ((xc - xv)**2 + (yc - yv)**2)**0.5
                
                # Features de distancia
                f_300 = ratio_base if distancia <= 500 else 0.0
                f_1000 = ratio_base if 500 < distancia <= 1000 else 0.0
                f_1500 = ratio_base if 1000 < distancia else 0.0

                feature_entregas[idc][idp] = [ratio_base, ratio_base**2, f_300, f_1000, f_1500]         
                features_list.extend([ratio_base, ratio_base**2, f_300, f_1000, f_1500])

        
        # Agregamos el feature de entrega por cliente-producto pero multiplicado por el ratio de inventario de cliente por producto
        for idc in self.instancia.clientes:
            for idp in self.instancia.productos:
                ratio_inventario = ratio_inventario_clientes_productos[idc][idp]
                if ratio_inventario < 0.2:
                    # Multiplicar las features de entrega por el ratio de inventario del cliente-producto
                    features_list.append(feature_entregas[idc][idp][0] * ratio_inventario)
                else:
                    # Si el ratio es mayor o igual a 0.2, agregar 4 ceros
                    features_list.append(0.0)
        

        # Convertir a array numpy una sola vez
        return np.array(features_list, dtype=np.float32)
    '''

    def obtener_features(self, estado, accion):
        """
        Obtiene los features de un estado y acción dados, y aplica
        transformación de Fourier a todos los features excepto beta_0.
        
        Devuelve un vector NumPy listo para el modelo lineal.
        """
        clientes = self.instancia.clientes
        productos = self.instancia.productos
        vehiculos = estado.posiciones_vehiculos

        # Copiar el estado y aplicar la acción
        estado_copia = copy.deepcopy(estado)
        self.proceso.actualizar_planificacion(estado_copia, accion)
        planificacion_post = estado_copia.planificacion

        pos_clientes = {
            idc: (clientes[idc].posicion_x, clientes[idc].posicion_y)
            for idc in self.instancia.id_clientes
        }

        # Inicializar lista con intercepto
        features_list = [1.0]  # beta_0

        # Feature: tiempo restante normalizado
        tiempo_norm = (self.instancia.horizonte_tiempo - estado.tiempo) / self.instancia.horizonte_tiempo
        features_list.append(tiempo_norm)

        # Features: inventario promedio por cliente-producto
        ratio_inventario_clientes_productos = {}
        for idc in self.instancia.id_clientes:
            cliente = clientes[idc]
            ratio_inventario_clientes_productos[idc] = {}
            for idp in productos:
                ratio = (estado.inventarios_clientes[idc][idp] * productos[idp].peso) / cliente.capacidad_almacenamiento
                features_list.append(ratio)
                ratio_inventario_clientes_productos[idc][idp] = ratio

        # Features: depot por vehículo-producto
        ratios_utilizacion = self.calcular_ratios_utilizacion_vehiculos(estado)
        for idv in self.instancia.id_vehiculos:
            for idp in productos:
                if not planificacion_post[idv]:
                    features_list.append(0.0)
                elif ratios_utilizacion[idv][idp] < 0.1 and next(iter(planificacion_post[idv])) == 0:
                    features_list.append(1.0)
                else:
                    features_list.append(0.0)

        # Features: entregas y distancias
        feature_entregas = {}
        for idc in self.instancia.id_clientes:
            xc, yc = pos_clientes[idc]
            capacidad = clientes[idc].capacidad_almacenamiento
            feature_entregas[idc] = {}

            for idp in productos:
                ratio_actual = ratio_inventario_clientes_productos[idc][idp]
                if ratio_actual >= 0.2:
                    feature_entregas[idc][idp] = [0.0, 0.0, 0.0, 0.0]
                    features_list.extend([0.0, 0.0, 0.0, 0.0])
                    continue

                ratio_peso, idv = self.calcular_peso_por_cliente_producto(planificacion_post, idc, idp)
                ratio_base = ratio_peso / capacidad

                if idv is None or idv not in vehiculos:
                    feature_entregas[idc][idp] = [ratio_base, 0.0, 0.0, 0.0]
                    features_list.extend([ratio_base, 0.0, 0.0, 0.0])
                    continue

                xv, yv = vehiculos[idv]['x'], vehiculos[idv]['y']
                distancia = np.hypot(xc - xv, yc - yv)

                f_300 = ratio_base if distancia <= 500 else 0.0
                f_1000 = ratio_base if 500 < distancia <= 1000 else 0.0
                f_1500 = ratio_base if distancia > 1000 else 0.0

                feature_entregas[idc][idp] = [ratio_base, f_300, f_1000, f_1500]
                features_list.extend([ratio_base, f_300, f_1000, f_1500])

        # Convertir a array numpy una sola vez
        return np.array(features_list, dtype=np.float32)




    def calcular_peso_por_cliente_producto(self, accion, id_cliente, id_producto):
        '''
        Descripción:
        Método que calcula el peso de entrega de un cliente en relación a un producto
        Args:
            * accion: Diccionario con la planificación de todos los vehículos
            * id_cliente: ID del cliente
            * id_producto: ID del producto
        Return:
            * peso_entrega: Peso de entrega del cliente en relación al producto
            * id_vehículo: ID del vehículo que tiene la entrega hacia ese cliente
        '''
        total_peso = 0.
        idv = None
        for id_vehículo , vehiculo in accion.items():
            # Verificar si el vehículo tiene acción
            if not vehiculo:
                continue  # Saltar vehículos inactivos
            # Obtener destino y productos (seguro porque vehiculo no está vacío)
            id_destino, productos = next(iter(vehiculo.items()))
            if id_destino == id_cliente:
                idv = id_vehículo
                for idp, cantidad in productos.items():
                    if idp == id_producto:
                        total_peso += cantidad * self.instancia.productos[id_producto].peso
        return total_peso, idv

    def calcular_ratios_utilizacion_vehiculos(self, estado: Estado):
        """
        Calcula el ratio de utilización (peso actual vs capacidad) para cada vehículo.
        
        Args:
            estado: Objeto Estado
        
        Returns:
            dict: Diccionario con {id_vehiculo: {id_producto : ratio_utilizacion} }
                Ratio entre 0.0 (vacío) y 1.0+ (sobrecargado)
        """
        ratios = {}
        
        for id_vehiculo, inventario in estado.inventarios_vehiculos.items():
            try:
                # 1. Obtener capacidad del vehículo
                vehiculo = self.instancia.vehiculos[id_vehiculo]
                capacidad = vehiculo.capacidad
                ratios[id_vehiculo] = {}
                for id_producto, cantidad in inventario.items():
                    producto = self.instancia.productos[id_producto]
                    ratios[id_vehiculo][id_producto] = round(cantidad * producto.peso / capacidad, 2)
            except KeyError as e:
                # Manejar vehículos o productos no existentes
                print(f"Advertencia: {e} no encontrado. Vehículo {id_vehiculo} omitido")
                ratio = 0.0
        return ratios


class MonteCarlo(Modelos_de_aproximacion):
    ''' Objeto que implementará la política MC Onpolicy al problema'''

    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate):
        super().__init__(instancia, proceso,episodios, epsilon, learning_rate)
        self.betas = None
        self.mejores_betas = np.array([])
        self.optimo_mejor_betas = np.array([])
        self.registro_optimos_mejores_betas = np.array([])
        self.registro_optimos = np.array([])

    def inicializar_mejores_betas(self):
        '''
        Descripción:
        Método que inicializa los mejores betas encontrados en el entrenamiento con los betas = 0
        '''
        self.mejores_betas = copy.deepcopy(self.betas)
        # ejecutamos el algoritmo MC OnPolicy para encontrar el óptimo promedio utilizando los mejores betas
        recompensas = np.array([])
        for _ in range(10):
            trayectoria,_ = self.probar_politica_optima_MC()
            costo_episodio =  sum(estado_accion['recompensa'] for estado_accion in trayectoria)
            recompensas = np.append(recompensas,costo_episodio)
        promedio_recompensa = np.mean(recompensas)
        self.optimo_mejor_betas = promedio_recompensa

    def actualizar_mejores_betas(self):
        '''
        Actualiza los mejores parámetros beta usando 20 simulaciones Monte Carlo,
        optimizando el rendimiento y uso de memoria.
        '''
        # 1. Colección eficiente de recompensas
        recompensas = []
        for _ in range(20):
            trayectoria = self.probar_politica_optima_MC()[0]  # Evita variable no usada
            costo_episodio = sum(paso['recompensa'] for paso in trayectoria)
            recompensas.append(costo_episodio)
        
        # 2. Cálculo vectorizado del promedio
        promedio_recompensa = np.mean(recompensas)
        
        # 3. Actualización de registros optimizada
        self.registro_optimos = np.append(self.registro_optimos, promedio_recompensa)
        
        # 4. Actualización condicional de mejores betas
        if promedio_recompensa < self.optimo_mejor_betas:
            self.mejores_betas = self.betas.copy()  # Copy de numpy vs deepcopy
            self.optimo_mejor_betas = promedio_recompensa
        
        # 5. Guardado eficiente del mejor óptimo
        self.registro_optimos_mejores_betas = np.append(
            self.registro_optimos_mejores_betas,
            self.optimo_mejor_betas  # Evita copia redundante
        )

    def run(self):
        '''
        Descripción:
        Método que ejecuta el algoritmo de MC Onpolicy
        '''
        self.entrenar_modelo()
        trayectoria = self.probar_politica_optima_MC()
        return trayectoria

    def crear_betas(self):
        '''
        Crea un vector de parámetros beta que coincide exactamente con la estructura de features.
        '''
        betas = [0]  # Beta para el término constante (feature inicial)
        
        # Beta para el feature de tiempo restante (1 beta) +1 
        betas.append(0.0)
        #betas.append(0.0) ## extra
        
        # Betas para inventario por cliente-producto (|N| * |P| betas) + el cuadrado
        for _ in self.instancia.id_clientes:
            for _ in self.instancia.id_productos:
                betas.append(0.0)
                #betas.append(0.0)
        
        
        # Betas para vehículo-producto (|M| * |P| betas)
        for _ in self.instancia.id_vehiculos:
            for _ in self.instancia.id_productos:
                betas.append(0.0)
        
        # Betas para entrega con distancia (4 betas por cliente-producto: base + 3 umbrales)
        for _ in self.instancia.id_clientes:
            for _ in self.instancia.id_productos:
                # 4 betas: [base, <300, <1000, >=1500]
                betas.extend([0.0, 0.0, 0.0, 0.0]) # uno extra
        '''
        # creamos los betas asociados a la entrega por cliente-producto multiplicado por el ratio de inventario del cliente-producto |N||P|
        for _ in self.instancia.id_clientes:
            for _ in self.instancia.id_productos:
                betas.append(0.0)
        '''
        self.numero_de_features = len(betas) # obtenemos el numero de features
        self.betas = np.array(betas, dtype=np.float32)

    def ejecutar_politica_epsilon_greedy(self):
        '''Ejecuta un episodio con política epsilon-greedy optimizada.'''
        proceso = self.proceso  # Cachear para acceso rápido
        es_terminal = self.es_terminal  # Evitar búsquedas de atributo
        trayectoria = []
        
        estado = proceso.determinar_estado_inicial()
        while True:
            # Paso 1: Tomar acción y obtener features
            accion = self.tomar_accion_epsilon_greedy(estado)
            features = self.obtener_features(estado, accion)
            
            # Paso 2: Transición de estado (ignoramos variables no usadas)
            nuevo_estado, recompensa, *_ = proceso.transicion(estado, accion)
            
            # Paso 3: Almacenar datos (usamos tupla para menor overhead)
            trayectoria.append( (estado, accion, features, recompensa) )
            
            # Paso 4: Verificar condición de término
            if es_terminal(nuevo_estado):  # ¡Clave! Verificar nuevo_estado
                break
                
            estado = nuevo_estado
            
        return [{
            'estado': s,
            'accion': a,
            'features': f,
            'recompensa': r
        } for s, a, f, r in trayectoria]

    def entrenar_modelo(self):
        ''' Método que ejecuta el algoritmo MC OnPolicy'''
        self.crear_betas()
        self.inicializar_mejores_betas()
        for episodio in range(self.episodios):
            trayectoria = self.ejecutar_politica_epsilon_greedy()

            # implementamos la política epsilon greedy
            G = 0
            for t in reversed(trayectoria):
                G = G  +  t['recompensa']
                # Actualizamos los pesos
                x = t['features']
                # Ahora obtenemos el c(st,at)
                #c_st_at = self.proceso.determinar_c_st_at(t['estado'], t['accion'])
                self.SGD(x, G)# ,c_st_at)
            
            if episodio % 100 == 0:
                self.actualizar_mejores_betas()

            #if episodio == 100:
             #   self.epsilon = self.epsilon_final
              #  self.learning_rate = self.learning_rate_final

    def probar_politica_optima_MC(self):
        ''' 
        Descripción:
        Método que implementa la política entrenada y hace una simulación tomando la mejor decisión
        Args: 
            * None
        Return:
            * trayectoria: Diccionario con el detalle de lo que hizo durante la simulación
        '''
        # Comenzamos en el estado inicial
        estado = self.proceso.determinar_estado_inicial()
        trayectoria = []
        costo_traslado = 0
        while True:
            acciones = self.obtener_acciones(estado)
            accion = self.politica_optima(estado, acciones)
            nuevo_estado, recompensa,traslado,_ = self.proceso.transicion(estado,accion)
            trayectoria.append({'estado': estado, 'accion': accion, 'recompensa': recompensa})
            costo_traslado += traslado
            if self.es_terminal(estado):
                break
            else:
                estado = nuevo_estado 
        return trayectoria, costo_traslado

    def SGD(self,x,y):#, c):
            '''
            Descripción:
            *   Método que aplica SGD, recibe un array x y actualiza los pesos originados self.betas
            Parámetros:
            *   x: Array con los features de ese estado acción
            *   y: Recompensa obtenida en ese estado acción
            *   c: costo fijo del estado acción
            '''
            # Predicción y error
            prediccion = np.dot(x, self.betas) #+ c
            error = prediccion - y  # o (y - prediction) dependiendo de la convención
            # Gradiente (para una muestra)
            gradients = x * error  # Si x es un vector 1D
            # Actualización
            self.betas -= self.learning_rate * gradients

    def tomar_accion_epsilon_greedy(self, estado):
        ''' 
        Descripción:
        Método que toma una acción utilizando la política MC OnPolicy en un estado dado

        Args:
            * estado: Objeto Estado
        Return:
            * accion: Acción a tomar
        '''
        numero = np.random.random()
        acciones = self.obtener_acciones(estado)
        if numero < self.epsilon: #or self.n_episodio < 100:
            accion = np.random.choice(acciones)
        else:
            accion = self.politica_optima(estado, acciones)
        return accion

    def ejecutar_politica_mejores_betas(self):
        '''
        Descripción:
        Método que ejecuta un episodio siguiendo la mejor acción con los mejores betas encontrados en el entrenamiento
        Args:
            * None
        Return:
            * trayectoria: Diccionario con el detalle de lo que hizo durante la simulación
        '''
        # Comenzamos en el estado inicial
        estado = self.proceso.determinar_estado_inicial()
        trayectoria = []
        costo_traslado = 0
        while True:
            acciones = self.obtener_acciones(estado)
            accion = self.politica_optima_mejores_betas(estado, acciones)
            nuevo_estado, recompensa,traslado,_ = self.proceso.transicion(estado,accion)
            trayectoria.append({'estado': estado, 'accion': accion, 'recompensa': recompensa})
            costo_traslado += traslado
            if self.es_terminal(estado):
                break
            else:
                estado = nuevo_estado 
        return trayectoria, costo_traslado

    def politica_optima_mejores_betas(self,state, acciones):
            '''
            Descripción:
            *   Método que devuelve la mejor accion para un estado utilizando la función de valor con mejores betas obtenidos
            Parámetros:
            *   state: Estado
            *   acciones: Lista de acciones posibles
            Return:
            *   action: Mejor acción
            '''
            #lista de tuplas
            mejor_Q = np.inf
            mejor_accion = None
            for action in acciones: 
                x = self.obtener_features(state, action) # Lista de features
                # obtenemos el c(st,at)
                #c_st_at = self.proceso.determinar_c_st_at(state, action)
                Q = np.dot(x, self.mejores_betas) # + c_st_at 
                if Q < mejor_Q:
                    mejor_Q = Q
                    mejor_accion = action
            return mejor_accion 
    
    def politica_optima(self,state, acciones):
            '''
            Descripción:
            *   Método que devuelve la mejor accion para un estado utilizando la función de valor
            Parámetros:
            *   state: Estado
            *   acciones: Lista de acciones posibles
            Return:
            *   action: Mejor acción
            '''

            qs = []
            for action in acciones:
                features = self.obtener_features(state, action)
                #costo = self.proceso.determinar_c_st_at(state, action)
                Q = np.dot(features, self.betas) #+ costo
                qs.append((Q, action))

            # Elegir la acción con el menor Q
            return min(qs, key=lambda x: x[0])[1]



class MonteCarloRN(Modelos_de_aproximacion):
    '''
    Clase que implementa la política MC Onpolicy utilizando redes neuronales. Hereda todo de la clase padre
    '''
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate):
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate)
        self.betas = None
        self.mejores_betas = np.array([])
        self.optimo_mejor_betas = np.array([])
        self.registro_optimos_mejores_betas = np.array([])
        self.registro_optimos = np.array([])
        self.numero_de_features = None
        self.red_neuronal = None
        self.determinar_numero_de_features()
        self.inicializar_red_neuronal()

    def determinar_numero_de_features(self):
        '''
        Método que determina el número de features a utilizar en la red neuronal
        '''

        numero_features = 2 # parte con 2, el bo y el tiempo restante
        
        # Betas para inventario por cliente-producto (|N| * |P| betas)
        for _ in self.instancia.id_clientes:
            for _ in self.instancia.id_productos:
                numero_features += 1
        
        # Betas para vehículo-producto (|M| * |P| betas)
        for _ in self.instancia.id_vehiculos:
            for _ in self.instancia.id_productos:
                numero_features += 1
        
        # Betas para entrega con distancia (4 betas por cliente-producto: base + 3 umbrales)
        for _ in self.instancia.id_clientes:
            for _ in self.instancia.id_productos:
                # 4 betas: [base, <300, <1000, >=1500]
                numero_features += 4 #uno por cada distancia
        
        self.numero_de_features = numero_features

    def inicializar_red_neuronal(self):
        '''
        Inicializa la red neuronal con pesos aleatorios (sin entrenamiento previo)
        '''
        # 1. Definir arquitectura
        #self.red_neuronal = tf.keras.Sequential([
            #tf.keras.Input(shape=(self.numero_de_features,)), 
            #tf.keras.layers.Dense(64, activation='relu'),
            #tf.keras.layers.Dense(64, activation='relu'),
            #tf.keras.layers.Dense(64, activation='relu'),
            #tf.keras.layers.Dense(1)
        #])
        
        # 2. Compilar (sin entrenar aún)
        #self.red_neuronal.compile(
         #   optimizer=tf.keras.optimizers.Adam(),
          #  loss='mse'
        #)

    def entrenar_modelo(self):
        ''' Método que ejecuta el algoritmo MC OnPolicy'''
        for episodio in range(self.episodios):
            trayectoria = self.ejecutar_politica_epsilon_greedy()
            # implementamos la política epsilon greedy
            G = 0
            features = []
            retornos = []
            for t in reversed(trayectoria):
                G = G  +  t['recompensa']
                # Ahora obtenemos el c(st,at)
                c_st_at = self.proceso.determinar_c_st_at(t['estado'], t['accion'])
                # almacenamos los features que se obtuvieron en cada paso
                features.append(t['features'])
                retornos.append(G + c_st_at)

            # Convertir a arrays y reordenar
            features_array = np.array(features[::-1])  # recuperar orden original
            retornos_array = np.array(retornos[::-1]).reshape(-1, 1)


            self.red_neuronal.fit(
                features_array, 
                retornos_array, 
                epochs=1, 
                batch_size=32,
                verbose=0
)

            #if episodio % 100 == 0:
            #    self.actualizar_mejores_betas()

    def ejecutar_politica_epsilon_greedy(self):
        '''Ejecuta un episodio con política epsilon-greedy optimizada.'''
        proceso = self.proceso  # Cachear para acceso rápido
        es_terminal = self.es_terminal  # Evitar búsquedas de atributo
        trayectoria = []
        
        estado = proceso.determinar_estado_inicial()
        while True:
            # Paso 1: Tomar acción y obtener features
            accion = self.tomar_accion_epsilon_greedy(estado)
            features = self.obtener_features(estado, accion)
            
            # Paso 2: Transición de estado (ignoramos variables no usadas)
            nuevo_estado, recompensa, *_ = proceso.transicion(estado, accion)
            
            # Paso 3: Almacenar datos (usamos tupla para menor overhead)
            trayectoria.append( (estado, accion, features, recompensa) )
            
            # Paso 4: Verificar condición de término
            if es_terminal(nuevo_estado):  # ¡Clave! Verificar nuevo_estado
                break
                
            estado = nuevo_estado
            
        return [{
            'estado': s,
            'accion': a,
            'features': f,
            'recompensa': r
        } for s, a, f, r in trayectoria]

    def tomar_accion_epsilon_greedy(self, estado):
        ''' 
        Descripción:
        Método que toma una acción utilizando la política MC OnPolicy en un estado dado

        Args:
            * estado: Objeto Estado
        Return:
            * accion: Acción a tomar
        '''
        numero = np.random.random()
        acciones = self.obtener_acciones(estado)
        if numero < self.epsilon:
            accion = np.random.choice(acciones)
        else:
            accion = self.politica_optima(estado, acciones)
        return accion

    def politica_optima(self, state, acciones):
        '''
        Descripción:
        *   Devuelve la mejor acción para un estado usando la red neuronal + costo c(s,a)

        Parámetros:
        *   state: estado actual
        *   acciones: lista de acciones posibles

        Return:
        *   mejor_accion: acción con menor Q(s,a)
        '''
        mejor_Q = np.inf
        mejor_accion = np.random.choice(acciones)  # por si todas las acciones tienen el mismo Q
        for action in acciones:
            x = self.obtener_features(state, action)  # vector de features
            x = np.expand_dims(x, axis=0)             # formato (1, n_features)
            
            q_estimado = self.red_neuronal.predict(x, verbose=0)[0][0]
            c_st_at = self.proceso.determinar_c_st_at(state, action)
            
            Q = q_estimado + c_st_at  # valor total estimado
            if Q < mejor_Q:
                mejor_Q = Q
                mejor_accion = action
## agregar que se obtengan todas las predicciones antes de comparar
        return mejor_accion
    
    def ejecutar_politica_RedNeuronal(self):
        '''
        Descripción:
        Método que ejecuta un episodio siguiendo la mejor acción con los mejores betas encontrados en el entrenamiento
        Args:
            * None
        Return:
            * trayectoria: Diccionario con el detalle de lo que hizo durante la simulación
        '''
        # Comenzamos en el estado inicial
        estado = self.proceso.determinar_estado_inicial()
        trayectoria = []
        costo_traslado = 0
        while True:
            acciones = self.obtener_acciones(estado)
            accion = self.politica_optima(estado, acciones)
            nuevo_estado, recompensa,traslado,_ = self.proceso.transicion(estado,accion)
            trayectoria.append({'estado': estado, 'accion': accion, 'recompensa': recompensa})
            costo_traslado += traslado
            if self.es_terminal(estado):
                break
            else:
                estado = nuevo_estado 
        return trayectoria, costo_traslado

class Qlearning(MonteCarlo):
    '''
    Clase que implementa la política Q-learning con aproximación de función lineal. Hereda todo de la clase padre MonteCarlo
    Pero agrega la función de aproximación de Q-learning
    '''

    def __init__(self, instancia, proceso, episodios, gamma, epsilon, learning_rate):
        '''
        Descripción:
        *   Constructor de la clase Qlearning
        Parámetros:
        *   instancia: Objeto que contiene todos los elementos de la simulación
        *   proceso: Objeto que contiene el proceso de simulación
        *   episodios: Número de episodios a realizar
        *   gamma: Factor de descuento
        *   epsilon: Factor de epsilon greedy
        *   learning_rate: Factor de aprendizaje
        '''
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate)
        self.gamma = gamma
        self.buffer = deque(maxlen=10_000)  # tamaño máximo del buffer
        self.batch_size = 64  # tamaño del minibatch para entrenar
    
    def entrenar_modelo(self):
        ''' Método que ejecuta el algoritmo Q-learning sobreescribiendo el de la clase padre MonteCarlo'''
        self.crear_betas()
        self.inicializar_mejores_betas()
        proceso = self.proceso 
        es_terminal = self.es_terminal
        trayectoria = []
        
        for episodio in range(self.episodios):
            # Inicializamos el estado actual con un objeto Estado()
            estado = proceso.determinar_estado_inicial() 
            while True:
                # Tomar acción y obtener features
                accion = self.tomar_accion_epsilon_greedy(estado) #diccionario
                features = self.obtener_features(estado, accion) # np.array()
                
                # Transición de estado y almacenamos lo relevante
                nuevo_estado, recompensa, *_ = proceso.transicion(estado, accion)
                
                # obtenemos la mejor acción en el nuevo estado
                mejor_accion_nuevo_estado = self.politica_optima(nuevo_estado, self.obtener_acciones(nuevo_estado))

                # obtenemos las features del nuevo estado-acción
                features_nuevo_estado = self.obtener_features(nuevo_estado, mejor_accion_nuevo_estado)

                # obtenemos el Q valor del nuevo estado-acción
                nuevo_Q = np.dot(features_nuevo_estado, self.betas)
                #nuevo_c_s_at = self.proceso.determinar_c_st_at(nuevo_estado, mejor_accion_nuevo_estado)

                # Calcular G (recompensa + gamma * Q(nuevo_estado, mejor_accion_nuevo_estado))
                G = recompensa + self.gamma * nuevo_Q  #nuevo_c_s_at)

                # determinamos el c(st,at)
                #c_st_at = self.proceso.determinar_c_st_at(estado, accion)

                # Actualizar betas usando SGD
                self.SGD(features, G)#, c_st_at)

                # Almacenar datos (usamos tupla para menor overhead)
                trayectoria.append((estado, accion, features, recompensa) )
                
                # Verificar condición de término
                if es_terminal(nuevo_estado):
                    break
                    
                estado = nuevo_estado
                
            if episodio % 100 == 0:
                self.actualizar_mejores_betas()
            
    def entrenar_modelo_buffer(self):
        '''Método que alimenta el buffer'''
        for episodio in range(self.episodios):
            estado = self.proceso.determinar_estado_inicial() 
            while True:
                accion = self.tomar_accion_epsilon_greedy(estado)
                features = self.obtener_features(estado, accion)
                nuevo_estado, recompensa, *_ = self.proceso.transicion(estado, accion)
                terminado = self.es_terminal(nuevo_estado)

                self.buffer.append((estado, accion, features, recompensa, nuevo_estado, terminado))


                if terminado:
                    break

                estado = nuevo_estado
            
            if episodio % 100 == 0:
                self.entrenar_con_batch()
                self.actualizar_mejores_betas()

    def entrenar_con_batch(self):
        '''Método que elige un conjunto de entradas random para entrenar el modelo de Q_learning'''
        batch = random.sample(self.buffer, self.batch_size)

        for estado, accion, features, recompensa, nuevo_estado, terminado in batch:
            # Política óptima para el nuevo estado
            mejor_accion_nuevo = self.politica_optima(nuevo_estado, self.obtener_acciones(nuevo_estado))
            features_nuevo = self.obtener_features(nuevo_estado, mejor_accion_nuevo)
            Q_nuevo = np.dot(features_nuevo, self.betas)

            c_nuevo = self.proceso.determinar_c_st_at(nuevo_estado, mejor_accion_nuevo)
            c_actual = self.proceso.determinar_c_st_at(estado, accion)

            if terminado:
                G = recompensa
            else:
                G = recompensa + self.gamma * (Q_nuevo + c_nuevo)

            # SGD con las features del estado actual
            self.SGD(features, G, c_actual)

class MonteCarlo_Fourier(MonteCarlo):
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate, max_i):
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate)
        self.max_i = max_i

    def crear_betas(self):
        '''
        Crea un vector de parámetros beta que coincide exactamente con la estructura de features.
        '''
        betas = [0]  # Beta para el término constante (feature inicial)
        
        # Beta para el feature de tiempo restante (1 beta)
        betas.append(0.0)
        
        # Betas para inventario por cliente-producto (|N| * |P| betas) + el cuadrado
        for _ in self.instancia.id_clientes:
            for _ in self.instancia.id_productos:
                betas.append(0.0)
                #betas.append(0.0)
        
        
        # Betas para vehículo-producto (|M| * |P| betas)
        for _ in self.instancia.id_vehiculos:
            for _ in self.instancia.id_productos:
                betas.append(0.0)
        
        # Betas para entrega con distancia (4 betas por cliente-producto: base + 3 umbrales)
        for _ in self.instancia.id_clientes:
            for _ in self.instancia.id_productos:
                # 4 betas: [base, <300, <1000, >=1500]
                betas.extend([0.0, 0.0, 0.0, 0.0])

        # creamos los betas asociados a la entrega por cliente-producto multiplicado por el ratio de inventario del cliente-producto |N||P|
        #for _ in self.instancia.id_clientes:
         #   for _ in self.instancia.id_productos:
          #      betas.append(0.0)

        self.numero_de_features = len(betas) # obtenemos el numero de features
        self.betas = np.array(betas, dtype=np.float32)

    def aumentar_betas(self, num_features_originales, componentes_por_feature):
        """
        Descripción: 
            Devuelve una lista de betas iniciales (todos ceros),
            considerando el intercepto beta0.
        Args:
            num_features_originales: Int que dice el número de features que hay (incluye intercepto)
            componentes_por_feature: Int, número de componentes que se harán por features ( numero de índices * 2)
        """
        num_features_reales = num_features_originales - 1  # No contamos intercepto para expandir
        num_features_nuevos = num_features_reales * componentes_por_feature
        total_betas = 1 + num_features_nuevos  # 1 para intercepto

        return np.array([0] * total_betas, dtype=np.float32)

    def entrenar_modelo(self):
        ''' Método que ejecuta el algoritmo MC OnPolicy'''
        self.crear_betas()
        self.betas = self.aumentar_betas(num_features_originales= self.numero_de_features, componentes_por_feature= self.max_i * 2)
        self.inicializar_mejores_betas()
        for episodio in range(self.episodios):
            trayectoria = self.ejecutar_politica_epsilon_greedy()
            # sumamos el contador de episodios
            self.n_episodio += 1
            # implementamos la política epsilon greedy
            G = 0
            for t in reversed(trayectoria):
                G = G  +  t['recompensa']
                # Actualizamos los pesos
                x = t['features']
                # Ahora obtenemos el c(st,at)
                #c_st_at = self.proceso.determinar_c_st_at(t['estado'], t['accion'])
                self.SGD(x, G)#, c_st_at)
            
            if episodio % 100 == 0:
                self.actualizar_mejores_betas()

        # optimizado


    # optimizado
    def obtener_features(self, estado, accion):
        """
        Obtiene los features de un estado y acción dados, y aplica
        transformación de Fourier a todos los features excepto beta_0.
        
        Devuelve un vector NumPy listo para el modelo lineal.
        """
        clientes = self.instancia.clientes
        productos = self.instancia.productos
        vehiculos = estado.posiciones_vehiculos

        # Copiar el estado y aplicar la acción
        estado_copia = copy.deepcopy(estado)
        self.proceso.actualizar_planificacion(estado_copia, accion)
        planificacion_post = estado_copia.planificacion

        pos_clientes = {
            idc: (clientes[idc].posicion_x, clientes[idc].posicion_y)
            for idc in self.instancia.id_clientes
        }

        # Inicializar lista con intercepto
        features_list = [1.0]  # beta_0

        # Feature: tiempo restante normalizado
        tiempo_norm = (self.instancia.horizonte_tiempo - estado.tiempo) / self.instancia.horizonte_tiempo
        features_list.append(tiempo_norm)

        # Features: inventario promedio por cliente-producto
        ratio_inventario_clientes_productos = {}
        for idc in self.instancia.id_clientes:
            cliente = clientes[idc]
            ratio_inventario_clientes_productos[idc] = {}
            for idp in productos:
                ratio = (estado.inventarios_clientes[idc][idp] * productos[idp].peso) / cliente.capacidad_almacenamiento
                features_list.append(ratio)
                ratio_inventario_clientes_productos[idc][idp] = ratio

        # Features: depot por vehículo-producto
        ratios_utilizacion = self.calcular_ratios_utilizacion_vehiculos(estado)
        for idv in self.instancia.id_vehiculos:
            for idp in productos:
                if not planificacion_post[idv]:
                    features_list.append(0.0)
                elif ratios_utilizacion[idv][idp] < 0.1 and next(iter(planificacion_post[idv])) == 0:
                    features_list.append(1.0)
                else:
                    features_list.append(0.0)

        # Features: entregas y distancias
        feature_entregas = {}
        for idc in self.instancia.id_clientes:
            xc, yc = pos_clientes[idc]
            capacidad = clientes[idc].capacidad_almacenamiento
            feature_entregas[idc] = {}

            for idp in productos:
                ratio_actual = ratio_inventario_clientes_productos[idc][idp]
                if ratio_actual >= 0.2:
                    feature_entregas[idc][idp] = [0.0, 0.0, 0.0, 0.0]
                    features_list.extend([0.0, 0.0, 0.0, 0.0])
                    continue

                ratio_peso, idv = self.calcular_peso_por_cliente_producto(planificacion_post, idc, idp)
                ratio_base = ratio_peso / capacidad

                if idv is None or idv not in vehiculos:
                    feature_entregas[idc][idp] = [ratio_base, 0.0, 0.0, 0.0]
                    features_list.extend([ratio_base, 0.0, 0.0, 0.0])
                    continue

                xv, yv = vehiculos[idv]['x'], vehiculos[idv]['y']
                distancia = np.hypot(xc - xv, yc - yv)

                f_300 = ratio_base if distancia <= 500 else 0.0
                f_1000 = ratio_base if 500 < distancia <= 1000 else 0.0
                f_1500 = ratio_base if distancia > 1000 else 0.0

                feature_entregas[idc][idp] = [ratio_base, f_300, f_1000, f_1500]
                features_list.extend([ratio_base, f_300, f_1000, f_1500])

        # Feature: entregas multiplicadas por inventario
        #for idc in self.instancia.clientes:
         #   for idp in self.instancia.productos:
          #      ratio_inventario = ratio_inventario_clientes_productos[idc][idp]
           #     if ratio_inventario < 0.2:
            #        features_list.append(feature_entregas[idc][idp][0] * ratio_inventario)
             #   else:
              #      features_list.append(0.0)

        # Convertir lista a NumPy
        features_array = np.array(features_list, dtype=np.float32)

        # Extraer intercepto
        intercept = features_array[0]
        # Features "reales" (sin intercepto)
        real_features = features_array[1:]

        # Transformación de Fourier (vectorizada)
        transformed_features = fourier_transform_features_vectorized(real_features, max_i=5)

        # Concatenar intercepto con features transformados
        final_features = np.concatenate(([intercept], transformed_features))

        return final_features


    # adaptamos el método original para incorporar fourier
    def obtener_features2(self, estado, accion):
        '''        
        Descripción:
        Método que obtiene los features de un estado dado
        Args:
            * estado: Objeto Estado
            * accion: Diccionario con la acción a aplicar
        Return:
            * features: Lista con los features del estado
        '''        

        # Precomputar estructuras necesarias
        clientes = self.instancia.clientes
        productos = self.instancia.productos
        vehiculos = estado.posiciones_vehiculos
        
        #creamos copia del estado
        estado_copia = copy.copy(estado)
        # aplicamos la acción
        self.proceso.actualizar_planificacion(estado_copia, accion)
        planificacion_post = estado_copia.planificacion

        # 1. Precalcular posiciones y capacidades
        pos_clientes = {
            idc: (clientes[idc].posicion_x, clientes[idc].posicion_y)
            for idc in self.instancia.id_clientes
        }
        # 2. Inicializar lista de features (eficiente para append) |1|
        features_list = [1]  # Feature inicial beta_0
        # 3. Feature de tiempo restante |1| +|1|
        tiempo_norm = (self.instancia.horizonte_tiempo - estado.tiempo) / self.instancia.horizonte_tiempo
        features_list.append(tiempo_norm)
        #features_list.append(tiempo_norm**2)  # Feature cuadrático del tiempo restante
        # 4. Features de inventario promedio por cliente-producto |N||P| y agregamos el cuadratico |N||P|*2
        ratio_inventario_clientes_productos = {}
        for idc in self.instancia.id_clientes:
            cliente = clientes[idc]
            ratio_inventario_clientes_productos[idc] = {}
            for idp in productos:
                ratio = (estado.inventarios_clientes[idc][idp] * productos[idp].peso) / cliente.capacidad_almacenamiento
                features_list.append(ratio)
                #features_list.append(ratio**2)  # Agregar cuadrático
                ratio_inventario_clientes_productos[idc][idp] = ratio
        # 5. Features de depot por vehículo-producto  |M||P|
        ratios_utilizacion = self.calcular_ratios_utilizacion_vehiculos(estado)
        for idv in self.instancia.id_vehiculos:
            for idp in productos:
                if not planificacion_post[idv]:
                    features_list.append(0.0)
                elif ratios_utilizacion[idv][idp] < 0.1 and next(iter(planificacion_post[idv])) == 0:
                    features_list.append(1.0)
                else:
                    features_list.append(0.0)
        # 6. Features combinados: entrega base + distancia  |N||P|4|
        # Para almacenar las features de entregas por cliente-producto
        feature_entregas = {}
        for idc in self.instancia.id_clientes:
            xc, yc = pos_clientes[idc]
            capacidad = clientes[idc].capacidad_almacenamiento
            feature_entregas[idc] = {}  # Inicializar lista de features para el cliente
            
            for idp in productos:
                ratio_actual = ratio_inventario_clientes_productos[idc][idp]
                if ratio_actual >= 0.2:  # debe ser si es menor que 0.2
                    # Agregar 4 ceros si no cumple el ratio
                    feature_entregas[idc][idp] = [0.0, 0.0, 0.0, 0.0]
                    features_list.extend([0.0, 0.0, 0.0, 0.0])
                    continue
                    
                # Calcular ratio base y vehículo asignado
                ratio_peso, idv = self.calcular_peso_por_cliente_producto(planificacion_post, idc, idp)
                ratio_base = ratio_peso / capacidad  # Feature base sin distancia
                
                if idv is None or idv not in vehiculos:
                    feature_entregas[idc][idp] = [0.0, 0.0, 0.0, 0.0]
                    features_list.extend([ratio_base, 0.0, 0.0, 0.0])
                    continue
                    
                # Calcular distancia una sola vez
                xv, yv = vehiculos[idv]['x'], vehiculos[idv]['y']
                distancia = ((xc - xv)**2 + (yc - yv)**2)**0.5
                
                # Features de distancia
                f_300 = ratio_base if distancia <= 500 else 0.0
                f_1000 = ratio_base if 500 < distancia <= 1000 else 0.0
                f_1500 = ratio_base if 1000 < distancia else 0.0

                feature_entregas[idc][idp] = [ratio_base, f_300, f_1000, f_1500]         
                features_list.extend([ratio_base, f_300, f_1000, f_1500])


        # Agregamos el feature de entrega por cliente-producto pero multiplicado por el ratio de inventario de cliente por producto
        #for idc in self.instancia.clientes:
         #   for idp in self.instancia.productos:
          #      ratio_inventario = ratio_inventario_clientes_productos[idc][idp]
           #     if ratio_inventario < 0.2:
                    # Multiplicar las features de entrega por el ratio de inventario del cliente-producto
            #        features_list.append(feature_entregas[idc][idp][0] * ratio_inventario)
             #   else:
                    # Si el ratio es mayor o igual a 0.2, agregar 4 ceros
              #      features_list.append(0.0)

        features_list = self.fourier_transform_features(features = features_list, max_i = self.max_i)

        return features_list    

    def fourier_transform_features(self, features, max_i):
        """
        Recibe un vector de features con intercepto explícito en la primera posición.
        Devuelve nuevo vector: intercepto original + features transformados por Fourier.
        """
        intercept = features[0]
        original_features = features[1:]  # Excluye intercepto

        transformed = []

        for feature in original_features:
            for i in range(1, max_i + 1):
                transformed.append(np.cos(i * feature))
                transformed.append(np.sin(i * feature))

        # Resultado final: intercepto + features transformados
        return np.concatenate(([intercept], np.array(transformed)))