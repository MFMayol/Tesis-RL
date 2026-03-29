from .base import Politica
from src.Estado import Estado
import numpy as np
from src.FuncionesAuxiliares import kmeans_clustering_sklearn

class PoliticasSimples(Politica):
    '''
    Clase Padre que implementa las politicas de decisiones y ejecuta la simulación utilizando las politica simple
    '''
    def __init__(self, obj_instancia, obj_proceso) -> None:
        """
        Inicializa el estado de la política simple delegando la asignación
        de los atributos fundamentales a la superclase (Politica).

        Args:
            obj_instancia (Instancia): Contenedor con los datos estáticos del problema.
            obj_proceso (Proceso): Motor que maneja la dinámica de transición de estados.
            
        Returns:
            None
        """
        super().__init__(instancia=obj_instancia, proceso=obj_proceso)

    def _determinar_vehiculos_con_quiebre(self, estado: Estado) -> list:
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

    def _determinar_cliente_critico(self, estado: Estado, clientes_disponibles: list) -> int:
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
        
        min_cliente = None
        min_ratio = np.inf
        
        for idc in clientes_disponibles:
            # Obtener inventario del cliente como array NumPy
            inventario = np.array([estado.inventarios_clientes[idc][idp] for idp in productos], dtype=np.float32)
            
            # Verificar si hay al menos un producto con inventario >= 0
            if not np.any(inventario >= 0):
                continue
            
            # Extraer las demandas específicas del cliente actual
            demandas = np.array([self.instancia.clientes[idc].demanda_media[idp] for idp in productos], dtype=np.float32)
            
            # Calcular ratios y obtener el mínimo
            ratios = np.divide(inventario, demandas, out=np.full_like(inventario, np.nan), where=demandas != 0)  # Evita div por cero y rellena basura con NaN
            ratio_min = np.nanmin(ratios)  # Ignora NaNs (productos con demanda 0)
            
            if ratio_min < min_ratio:
                min_ratio = ratio_min
                min_cliente = idc
        
        return min_cliente

    def _determinar_vehiculos_disponibles(self, estado: Estado):
        '''
        Método que devuelve los vehiculos que no tienen planificación
        
        Parámetros:
            - estado: Estado actual del problema

        Return: lista con ids de vehiculos disponibles
        '''
        # Determinamos los vehiculos que no tienen nada asignado
        planificacion = estado.planificacion  # Guardar referencia
        return [idv for idv in self.instancia.id_vehiculos if not planificacion[idv]]

    def _determinar_clientes_disponibles(self, estado: Estado):
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
    
    def _obtener_4_clientes_criticos(self, estado, clientes_disponibles):
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

    def _obtener_acciones(self, estado: Estado):
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
        vehiculos_disponibles = self._determinar_vehiculos_disponibles(estado)
        vehiculos_con_quiebre = set(self._determinar_vehiculos_con_quiebre(estado))  # Convertir a set para O(1) lookups
        
        # Encontrar el primer vehículo disponible con quiebre usando next() + generador (más eficiente que loop)
        vehiculo_quiebre = next((idv for idv in vehiculos_disponibles if idv in vehiculos_con_quiebre), None)
        
        if vehiculo_quiebre is not None:
            acciones_posibles.append({vehiculo_quiebre: {0: {}}})  # Acción de redirección

        # --- Sección 2: Acciones para clientes críticos ---
        clientes_disponibles = self._determinar_clientes_disponibles(estado)
        criticos = self._obtener_4_clientes_criticos(estado, clientes_disponibles)
        
        # Precomputar demandas de clientes para evitar accesos múltiples
        demandas_clientes = {idc: self.instancia.clientes[idc].demanda_media for idc in criticos}
        
        # Generar todas las acciones evitando viajes en donde no se entregue nada
        nuevas_acciones = []
        for id_cliente in criticos:
            vehiculos_cercanos = self._determinar_2_vehiculos_mas_cercanos_disponibles(estado, id_cliente, vehiculos_disponibles)
            for idv in vehiculos_cercanos:
                plan = self.planificar_entrega(estado.inventarios_vehiculos[idv], demandas_clientes[id_cliente])
                # Solo se crea la acción si efectivamente se entregará 1 o más unidades en total
                if sum(plan.values()) > 0:
                    nuevas_acciones.append({idv: {id_cliente: plan}})
        
        acciones_posibles.extend(nuevas_acciones)
        return acciones_posibles

    def _determinar_2_vehiculos_mas_cercanos_disponibles(self, estado, id_cliente, vehiculos_disponibles):
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

        # Si no hay vehículos disponibles, retornar lista vacía.
        if not vehiculos_disponibles:
            return []

        # 1. Crear una matriz de posiciones de vehículos (N, 2)
        posiciones_vehiculos = np.array([
            [estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y']]
            for idv in vehiculos_disponibles
        ], dtype=np.float32)

        # 2. Calcular distancias al cuadrado de forma vectorizada usando broadcasting
        distancias_sq = np.sum((posiciones_vehiculos - posicion_cliente)**2, axis=1)

        # 3. Obtener los índices de los 2 vehículos más cercanos
        num_vehiculos = len(vehiculos_disponibles)
        k = min(2, num_vehiculos)  # Asegurarse de no pedir más vehículos de los que hay
        indices_mas_cercanos = np.argsort(distancias_sq)[:k]

        # 4. Mapear los índices de vuelta a los IDs de los vehículos
        vehiculos_disponibles_arr = np.array(vehiculos_disponibles)
        return vehiculos_disponibles_arr[indices_mas_cercanos].tolist()

class PoliticaSimpleClusterisada(PoliticasSimples):
    '''
    Clase que implementa la politica de decisiones y ejecuta la simulación utilizando la politica simple clusterizada
    '''
    def __init__(self, instancia, proceso):
        super().__init__(instancia, proceso)
        self.asignaciones_k_means = self._aplicar_k_means()

    def _aplicar_k_means(self):
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
        #asignaciones_k_means = self._aplicar_k_means() # dict que tiene llave el id_vehiculo y valor una lista de ids de clientes asignados
        accion = {} # diccionario que no tiene nada asignado de momento

        # Determinamos los vehiculos que no tienen nada asignado
        vehiculos_disponibles = self._determinar_vehiculos_disponibles(estado) # lista de ids de vehiculos sin planificación
        #Si no hay vehículos disponibles entonces no se toma ninguna acción
        if vehiculos_disponibles ==  []:
            return accion
        # Determinamos los vehiculos con quiebres
        vehiculos_con_quiebre = self._determinar_vehiculos_con_quiebre(estado) # lista de ids de vehiculos con inventario bajo
        # asignamos una planificación a los vehiculos sin planificación
        # vemos los que tienen quiebre
        for idv in vehiculos_disponibles:
            # Si tiene quiebre, se asigna su ruta al depot
            if idv in vehiculos_con_quiebre:
                accion[idv] = {0 :{}} # su nueva ruta será a depot con planificación vacía
                return accion
        # Determinamos los clientes que no están asignados a ningún vehiculo
        clientes_disponibles = self._determinar_clientes_disponibles(estado) # lista de ids de clientes sin planificación
        for idv in vehiculos_disponibles:
                # Convertir a lista de manera segura con .get para evitar KeyErrors
                clientes_disp_list = [e for e in clientes_disponibles if e in self.asignaciones_k_means.get(idv, [])]
                if not clientes_disp_list: # Comprobar si la lista está vacía
                    break
                # Si no tiene quiebre, se asigna su ruta al cliente crítico
                id_cliente_con_inventario_bajo = self._determinar_cliente_critico(estado, clientes_disp_list)
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
        vehiculos_disponibles = self._determinar_vehiculos_disponibles(estado) # lista de ids de vehiculos sin planificación
        #Si no hay vehículos disponibles entonces no se toma ninguna acción
        if vehiculos_disponibles ==  []:
            return accion
        # Determinamos los clientes que no están asignados
        clientes_disponibles = self._determinar_clientes_disponibles(estado) # lista de ids de clientes sin planificación
        # Determinamos los vehiculos que tengan un porcentaje de inventario menor a un umbral del 20%
        vehiculos_con_quiebre = self._determinar_vehiculos_con_quiebre(estado) # lista de ids de vehiculos con inventario bajo el umbral
        # asignamos una planificación a los vehiculos sin planificación
        for idv in vehiculos_disponibles:
            # Si tiene quiebre, se asigna su ruta al depot
            if idv in vehiculos_con_quiebre:
                accion[idv] = {0 :{}} # su nueva ruta será a depot con planificación vacía
                return accion
            
        for idv in vehiculos_disponibles:
                # Si no tiene quiebre, se asigna su ruta al cliente crítico
                id_cliente_con_inventario_bajo = self._determinar_cliente_critico(estado, clientes_disponibles)
                if id_cliente_con_inventario_bajo is None: # Si no quedan clientes, no se modifica la ruta y se queda 'quieto'
                    break
                planificacion_del_vehiculo = self.planificar_entrega(inventario = estado.inventarios_vehiculos[idv], demandas= self.instancia.clientes[id_cliente_con_inventario_bajo].demanda_media)
                # Se determina su planificación a ese cliente como entregar todo lo que tiene (PREGUNTAR COMO HACER ESTO)
                # Una vez que se asignó la planificación, se agrega al diccionario acciones
                accion[idv] = {id_cliente_con_inventario_bajo : planificacion_del_vehiculo}
                return accion 
