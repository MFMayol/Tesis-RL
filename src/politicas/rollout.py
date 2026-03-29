from .simples import PoliticasSimples
from src.Estado import Estado
import numpy as np
from src.FuncionesAuxiliares import kmeans_clustering_sklearn

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
        acciones_factibles = self._obtener_acciones(estado) 

        # Lista para almacenar (acción, costo promedio)
        valores_objetivo = []

        # Simular cada acción
        for accion in acciones_factibles:
            # Realizar 20 simulaciones y calcular el costo promedio
            recompensas_totales = [self.simular_episodio_rollout(accion, estado) for _ in range(20)] 
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
        vehiculos_disponibles = self._determinar_vehiculos_disponibles(estado) # lista de ids de vehiculos sin planificación
        #Si no hay vehículos disponibles entonces no se toma ninguna acción
        if vehiculos_disponibles ==  []:
            return accion
        # Determinamos los vehiculos que tengan un porcentaje de inventario menor a un umbral del 20%
        vehiculos_con_quiebre = self._determinar_vehiculos_con_quiebre(estado) # lista de ids de vehiculos con inventario bajo el umbral
        # asignamos una planificación a los vehiculos sin planificación
        for idv in vehiculos_disponibles:
            # Si tiene quiebre, se asigna su ruta al depot
            if idv in vehiculos_con_quiebre:
                accion[idv] = {0 :{}} # su nueva ruta será a depot con planificación vacía
                return accion

        # Determinamos los clientes que no están asignados
        clientes_disponibles = self._determinar_clientes_disponibles(estado) # lista de ids de clientes sin planificación

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

    def simular_episodio_rollout(self, accion, estado: Estado):
        '''
        Método que ejecuta la política

        Args:
        *   accion: Dict
        *   estado: Objeto Estado
        Return:
        *   recompensa_total: float
        '''
        estado, recompensa, *_ = self.proceso.transicion(estado, accion) # Hacemos la transición inicial

        trayectoria = [] # Lista que almacena los estado-accion-recompensas tomadas a lo largo de la simulación, serán guardados como diccionarios 
        costo_traslado = 0
        costos_de_insatisfecha = 0
        costos_de_almacenamiento = 0
        costo_total = recompensa
        # Se ejecuta hasta que se encuentre un estado terminal
        while True:
            accion = self.tomar_accion_politica_simple(estado)
            estado_nuevo, recompensa, costos_de_traslado, costos_de_demanda_insatisfecha, costos_almc = self.proceso.transicion(estado, accion)
            trayectoria.append({'estado': estado, 'accion': accion, 'recompensa': recompensa})
            costo_traslado += costos_de_traslado
            costos_de_insatisfecha += costos_de_demanda_insatisfecha
            costos_de_almacenamiento += costos_almc
            costo_total += recompensa
            if self.es_terminal(estado_nuevo):
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
        self.asignaciones_k_means = self._aplicar_k_means()
        self.periodos = 0

    def tomar_accion(self, estado: Estado):
        '''
        Método que determina la mejor acción según la política RollOut.
        
        Args: 
            * estado: Objeto Estado

        Return:
            * Accion (Dict): La acción con el menor costo esperado.
        '''

        # Obtenemos una lista con las acciones posibles que se pueden tomar
        acciones_factibles = self._obtener_acciones(estado) 

        # Lista para almacenar (acción, costo promedio)
        valores_objetivo = []

        # Simular cada acción
        for accion in acciones_factibles:
            # Realizar 20 simulaciones y calcular el costo promedio
            recompensas_totales = np.array([self.simular_episodio_rollout(accion, estado) for _ in range(20)])
            costo_promedio = np.mean(recompensas_totales)

            # Guardar la acción junto con su costo esperado
            valores_objetivo.append((accion, costo_promedio))

        # Seleccionar la acción con el menor costo promedio
        accion_seleccionada = min(valores_objetivo, key=lambda x: x[1])[0]

        return accion_seleccionada

    def simular_episodio_rollout(self, accion, estado: Estado):
        '''
        Método que ejecuta la política

        Args:
        *   accion: Dict
        *   estado: Objeto Estado
        Return:
        *   recompensa_total: float
        '''
        estado, recompensa, *_ = self.proceso.transicion(estado, accion) # Hacemos la transición inicial

        trayectoria = [] # Lista que almacena los estado-accion-recompensas tomadas a lo largo de la simulación, serán guardados como diccionarios 
        costo_traslado = 0
        costos_de_insatisfecha = 0
        costos_almacenamiento = 0
        costo_total = recompensa
        # Se ejecuta hasta que se encuentre un estado terminal
        while True:
            accion = self.tomar_accion_politica_cluster(estado)
            estado_nuevo, recompensa, costos_de_traslado, costos_de_demanda_insatisfecha, costos_almac  = self.proceso.transicion(estado, accion)
            trayectoria.append({'estado': estado, 'accion': accion, 'recompensa': recompensa})
            costo_traslado += costos_de_traslado
            costos_de_insatisfecha += costos_de_demanda_insatisfecha
            costos_almacenamiento += costos_almac
            costo_total += recompensa
            if self.es_terminal(estado_nuevo):
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
                # Determinamos los clientes que se encuentran en el cluster del vehiculo
                clientes_disponibles_k_means = [elemento for elemento in clientes_disponibles if elemento in self.asignaciones_k_means.get(idv, [])]
                if not clientes_disponibles_k_means: # Si no quedan clientes, no se modifica la ruta y se queda 'quieto'
                    break
                # Si no tiene quiebre, se asigna su ruta al cliente crítico
                id_cliente_con_inventario_bajo = self._determinar_cliente_critico(estado, clientes_disponibles_k_means)

                if id_cliente_con_inventario_bajo == None:
                    continue 
                else:
                    planificacion_del_vehiculo = self.planificar_entrega(inventario = estado.inventarios_vehiculos[idv], demandas= self.instancia.clientes[id_cliente_con_inventario_bajo].demanda_media)
                    accion[idv] = {id_cliente_con_inventario_bajo : planificacion_del_vehiculo}
                    return accion
        return accion

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
