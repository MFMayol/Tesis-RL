from Estado import Estado
import numpy as np
from FuncionesAuxiliares import distancia_euclidiana, kmeans_clustering_sklearn
from abc import ABC, abstractmethod
import copy

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
        planificacion = {}
        
        # Calcular la demanda total esperada y el total de inventario disponible
        total_demanda = sum(demandas.values())
        total_inventario = sum(inventario.values())
        
        # Si no hay demanda o inventario, se retorna una planificación con entrega 0 para todos los productos
        if total_demanda == 0 or total_inventario == 0:
            return {producto: 0 for producto in demandas}
        
        # Distribuir el inventario proporcionalmente según la demanda de cada producto
        for producto, demanda in demandas.items():
            ponderador = demanda / total_demanda  # Proporción de la demanda respecto a la demanda total
            entrega = round(ponderador * total_inventario)  # Cantidad a entregar basada en la proporción
            disponible = inventario.get(producto, 0)  # Obtener la cantidad disponible en inventario
            planificacion[producto] = min(entrega, disponible)  # No entregar más de lo disponible        
        return planificacion

class PoliticasSimples(Politica):
    '''
    Clase Padre que implementa las politicas de decisiones y ejecuta la simulación utilizando las politica simple
    '''
    def __init__(self, instancia, proceso):
        self.instancia = instancia
        self.proceso = proceso

    def determinar_vehiculos_con_quiebre(self, estado: Estado):
        '''
        Método que determina los vehiculos que tienen un porcentaje de inventario menor a un umbral del 20%
        Parámetros:
            - estado: Estado actual del problema
        Retorna:
            Lista con los id de vehiculos con quiebre (list)
        '''
        
        vehiculos_quiebres = []

        for idv in self.instancia.vehiculos.keys():
            inventario_utilizado = sum(cantidad * self.instancia.productos[idp].peso for idp, cantidad in estado.inventarios_vehiculos[idv].items())
            # Si su ratio es menor que el 0.2, se agrega a la lista de vehiculos quiebres
            if inventario_utilizado/ self.instancia.vehiculos[idv].capacidad < self.instancia.umbral_inventario_vehiculos:
                vehiculos_quiebres.append(idv)
        return vehiculos_quiebres
    
    def determinar_cliente_critico(self, estado: Estado, clientes_disponibles):
        '''
        Método que determina el cliente crítico

        Parámetros:
            - estado: Estado actual del problema
            - clientes_disponibles: Lista de clientes disponibles

        return:
            id del cliente con inventario más bajo (int)
        '''

        id_cliente_con_inventario_bajo = min(
                (idc for idc in clientes_disponibles if any(estado.inventarios_clientes[idc][idp] >= 0 for idp in self.instancia.productos.keys())),
                key=lambda idc: min(estado.inventarios_clientes[idc][idp] / self.instancia.demandas_medias[idp] for idp in self.instancia.productos.keys()),
                default=None
            )
        return id_cliente_con_inventario_bajo

    def determinar_vehiculos_disponibles(self, estado: Estado):
        '''
        Método que devuelve los vehiculos que no tienen planificación
        
        Parámetros:
            - estado: Estado actual del problema

        Return: lista con ids de vehiculos disponibles
        '''

        # Determinamos los vehiculos que no tienen nada asignado
        vehiculos_disponibles = []
        for idv in self.instancia.id_vehiculos:
            if estado.planificacion[idv] == {}:
                vehiculos_disponibles.append(idv)
        return vehiculos_disponibles
    
    def determinar_clientes_disponibles(self, estado: Estado):
        '''
        Método que devuelve los clientes que no están asignados a ningún vehiculo

        Return: lista con ids de clientes disponibles
        '''
        clientes_disponibles = []
        for idc in self.instancia.clientes.keys():
            idc_asignado = False # Se parte como que no está asignado y cambiará según si está asignado a algún vehiculo
            for idv in self.instancia.vehiculos.keys():
                # Si está en la planificación del vehiculo, se marca como asignado
                if idc in estado.planificacion[idv].keys(): # Si está en la planificación del vehiculo (key de ese diccionario), se marca como asignado
                    idc_asignado = True
            # Si no se asignó a ningún vehiculo, se agrega a la lista de clientes disponibles
            if not idc_asignado:
                clientes_disponibles.append(idc)
        
        return clientes_disponibles

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
                if id_cliente_con_inventario_bajo == None: # Si no quedan clientes, no se modifica la ruta y se queda 'quieto'
                    break
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
            recompensas_totales = [self.simular_episodio_rollout(accion, estado) for _ in range(3)]
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
                if id_cliente_con_inventario_bajo == None: # Si no quedan clientes, no se modifica la ruta y se queda 'quieto'
                    break
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
        costo_total = 0
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

    def obtener_acciones(self, estado:Estado):
        '''
        Descripción:
        Método que devuelve una lista con las acciones posibles para un estado
        
        Args: 
            * estado: Objeto Estado
        Return:
            * acciones_posibles: Lista con las acciones factibles
        '''
        acciones_posibles = []

        # Agregamos la acción 'nula'
        acciones_posibles.append({})

        ############################################### agregamos acción redirigir a depot######################
        # Ahora agregamos la que redirige los vehiculos con stock crítico

        vehiculos_disponibles = self.determinar_vehiculos_disponibles(estado = estado)
        vehiculos_con_quiebre = self.determinar_vehiculos_con_quiebre(estado = estado)

        accion_quiebre = {}
        for idv in vehiculos_disponibles:
            if idv in vehiculos_con_quiebre:
                accion_quiebre[idv] = {0 :{}} # su nueva ruta será a depot con planificación vacía
                break
        # Si hay vehículos con quiebre, agregamos la acción de redirigir a depot
        if accion_quiebre != {}:
            acciones_posibles.append(accion_quiebre)
        ########################################## agregamos las acciones posibles respecto a elegir los 4 clientes con menor ratio y les asignamos hasta 2 vehículos ############
        clientes_disponibles = self.determinar_clientes_disponibles(estado = estado)

        ids_4_clientes_criticos = self.obtener_4_clientes_criticos(estado = estado, clientes_disponibles = clientes_disponibles)

        # ahora determinamos los 2 vehículos más cercanos a cada cliente crítico
        for id_cliente in ids_4_clientes_criticos:
            # Determinamos los vehículos más cercanos
            vehiculos_mas_cercanos = self.determinar_2_vehiculos_mas_cercanos_disponibles(estado = estado, id_cliente = id_cliente, vehiculos_disponibles = vehiculos_disponibles) # AGREGAR VEHÍCULOS DISPONIBLES
            # Si no hay vehículos disponibles, no se agrega la acción y se pasa al siguiente vehículo crítico
            if vehiculos_mas_cercanos == []:
                pass
            else:# Para cada vehículo más cercano, se agrega la acción de asignarle el cliente crítico
                for idv in vehiculos_mas_cercanos:
                    accion = {idv :{id_cliente: self.planificar_entrega(inventario = estado.inventarios_vehiculos[idv], demandas = self.instancia.clientes[id_cliente].demanda_media)}}
                    acciones_posibles.append(accion) 

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

    def obtener_4_clientes_criticos(self, estado, clientes_disponibles):
        '''
        Descripción:
            Método que determina los 4 clientes con stock crítico considerando el peso de los productos.
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
            recompensas_totales = [self.simular_episodio_rollout(accion, estado) for _ in range(3)]
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
        costo_total = 0
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

    def obtener_4_clientes_criticos(self, estado, clientes_disponibles):
        '''
        Descripción:
            Método que determina los 4 clientes con stock crítico considerando el peso de los productos.
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

    def obtener_acciones(self, estado:Estado):
        '''
        Descripción:
        Método que devuelve una lista con las acciones posibles para un estado
        
        Args: 
            * estado: Objeto Estado
        Return:
            * acciones_posibles: Lista con las acciones factibles
        '''
        acciones_posibles = []

        # Agregamos la acción 'nula'
        acciones_posibles.append({})

        ############################################### agregamos acción redirigir a depot######################
        # Ahora agregamos la que redirige los vehiculos con stock crítico

        vehiculos_disponibles = self.determinar_vehiculos_disponibles(estado = estado)
        vehiculos_con_quiebre = self.determinar_vehiculos_con_quiebre(estado = estado)

        accion_quiebre = {}
        for idv in vehiculos_disponibles:
            if idv in vehiculos_con_quiebre:
                accion_quiebre[idv] = {0 :{}} # su nueva ruta será a depot con planificación vacía
                break
        # Si hay vehículos con quiebre, agregamos la acción de redirigir a depot
        if accion_quiebre != {}:
            acciones_posibles.append(accion_quiebre)
        ########################################## agregamos las acciones posibles respecto a elegir los 4 clientes con menor ratio y les asignamos hasta 2 vehículos ############
        clientes_disponibles = self.determinar_clientes_disponibles(estado = estado)

        ids_4_clientes_criticos = self.obtener_4_clientes_criticos(estado = estado, clientes_disponibles = clientes_disponibles)

        # ahora determinamos los 2 vehículos más cercanos a cada cliente crítico
        for id_cliente in ids_4_clientes_criticos:
            # Determinamos los vehículos más cercanos
            vehiculos_mas_cercanos = self.determinar_2_vehiculos_mas_cercanos_disponibles(estado = estado, id_cliente = id_cliente, vehiculos_disponibles = vehiculos_disponibles) # AGREGAR VEHÍCULOS DISPONIBLES
            # Si no hay vehículos disponibles, no se agrega la acción y se pasa al siguiente vehículo crítico
            if vehiculos_mas_cercanos == []:
                pass
            else:# Para cada vehículo más cercano, se agrega la acción de asignarle el cliente crítico
                for idv in vehiculos_mas_cercanos:
                    accion = {idv :{id_cliente: self.planificar_entrega(inventario = estado.inventarios_vehiculos[idv], demandas = self.instancia.clientes[id_cliente].demanda_media)}}
                    acciones_posibles.append(accion) 

        return acciones_posibles

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




class MonteCarlo(PoliticasSimples):
    ''' Objeto que implementará la política MC Onpolicy al problema'''

    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate):
        super().__init__(instancia, proceso)
        self.episodios = episodios
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.betas = None
        self.mejores_betas = np.array([])
        self.optimo_mejor_betas = np.array([])
        self.registro_optimos_mejores_betas = np.array([])
        self.registro_optimos = np.array([])
        self.crear_betas()
        self.inicializar_mejores_betas()

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
        Descripción:
        Método que actualiza los mejores betas encontrados en el entrenamiento cada 10 episodios
        '''
        # ejecutamos el algoritmo MC OnPolicy para encontrar el óptimo promedio utilizando los mejores betas
        recompensas = np.array([])
        for _ in range(10):
            trayectoria,_ = self.probar_politica_optima_MC()
            costo_episodio = sum(estado_accion['recompensa'] for estado_accion in trayectoria)
            recompensas = np.append(recompensas,costo_episodio)
        promedio_recompensa = np.mean(recompensas)
        self.registro_optimos = np.append(self.registro_optimos, promedio_recompensa)
        if promedio_recompensa < self.optimo_mejor_betas:
            self.mejores_betas = copy.deepcopy(self.betas)
            self.optimo_mejor_betas = promedio_recompensa
        mejor_optimo_betas = copy.deepcopy(self.optimo_mejor_betas)
        #guardamos el registro de los mejores betas
        self.registro_optimos_mejores_betas = np.append(self.registro_optimos_mejores_betas, mejor_optimo_betas)

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
        Descripción:
        *   Método que crea la matriz de betas
        '''
        bo = 0
        betas = np.array([])
        betas = np.append(betas, bo) # agregamos el parámetro constante

        # agregamos un beta igual a 0 asociado al primer feature de tiempo restante (1)
        betas = np.append(betas, bo)

        # agregamos los features asociados al inventario promedio que tienen de ese producto todos los vehículos (|P|)
        #for idp in self.instancia.id_productos:
         #   betas = np.append(betas, bo)

        # agregamos los features asociados al inventario que tiene cada cliente (|N|* |P|)
        for idc in self.instancia.id_clientes:
            for idp in self.instancia.id_productos:
                betas = np.append(betas, bo)
        
        # agregamos los features asociados a la entrega relativa que tiene cada cliente (|N| * |P|)
        for idc in self.instancia.id_clientes:
            for idp in self.instancia.id_productos:
                betas = np.append(betas, bo)
        
        #agregamos los features para cada vehículo (|M|* |P|)
        
        for id_vehiculo in self.instancia.id_vehiculos:
            for idp in self.instancia.id_productos:
                betas = np.append(betas, bo)

        # agregamos el feature de número de vehículos con planificación |1|
        #betas = np.append(betas, bo)

        # Porcentaje de vehículos en camino al depot si tienen menos del 20% de su inventario utilizado
        #betas = np.append(betas, bo)

        self.betas = betas

    def obtener_features(self, estado, accion):
        '''
        Descripción:
        Método que obtiene los features de un estado dado
        Args:
            * estado: Objeto Estado
            * accion: Diccionario con la acción a aplicar
        Return:
            * features: Lista con los features del estado
        '''

        ratio_inventario_clientes_productos = {} # diccionario donde almacenamos los ratios de cada cliente
        #creamos copia del estado
        estado_copia = copy.copy(estado)
        # aplicamos la acción
        self.proceso.actualizar_planificacion(estado_copia, accion)
        planificacion_post = estado_copia.planificacion
        ratios_utilizacion_vehiculos = self.calcular_ratios_utilizacion_vehiculos(estado= estado)
        #creamos una lista vacia para almacenar los features
        features = np.array([1])
        ###########################################################################################
        # Agregamos el feature de tiempo restante (1 features)
        features = np.append(features, (self.instancia.horizonte_tiempo - estado.tiempo)/self.instancia.horizonte_tiempo) # valor entre 0 y 1 
        ###########################################################################################
        # agregamos los features de producto promedio que se tiene de cada producto en los vehículos (|P| features)
        #for idp in self.instancia.id_productos:
         #   inventario_promedio = sum(estado.inventarios_vehiculos[idv][idp] for idv in self.instancia.id_vehiculos) / len(self.instancia.id_vehiculos)
          #  features = np.append(features, inventario_promedio)
        ####################################################################################################################
        # agregamos los features de inventario promedio que se tiene cada cliente por producto (|N| * |P| features)
        for id_cliente in self.instancia.id_clientes:
            ratio_inventario_clientes_productos[id_cliente] = {}
            for idp in self.instancia.id_productos:
                ratio_inventario_cliente_producto = (estado.inventarios_clientes[id_cliente][idp] * self.instancia.productos[idp].peso) / self.instancia.clientes[id_cliente].capacidad_almacenamiento
                features = np.append(features, ratio_inventario_cliente_producto)
                ratio_inventario_clientes_productos[id_cliente][idp]= ratio_inventario_cliente_producto 
        ####################################################################################################################
        # agregamos el feature de entrega relativa al ratio que tiene el cliente (|N|*|P|)
        for id_cliente in self.instancia.id_clientes:
            capacidad_cliente = self.instancia.clientes[id_cliente].capacidad_almacenamiento
            for idp in self.instancia.id_productos:
                if ratio_inventario_clientes_productos[id_cliente][idp] < 0.2:
                    ratio_peso_entrega = self.calcular_peso_por_cliente_producto(accion= planificacion_post, id_cliente= id_cliente, id_producto= idp) / capacidad_cliente
                    feature = ratio_peso_entrega
                    features = np.append(features, feature)
                else:
                    features = np.append(features, 0)
        ####################################################################################################################
        # agregamos el feature sobre si el vehículo va al depot en caso de que tenga quiebre de stock |M|* |P|
        
        for id_vehiculo in self.instancia.id_vehiculos:
            for idp in self.instancia.id_productos:
                if not planificacion_post[id_vehiculo]:
                    features = np.append(features, 0)
                    continue
                if ratios_utilizacion_vehiculos[id_vehiculo][idp] < 0.1 and next(iter(planificacion_post[id_vehiculo])) == 0:
                    features = np.append(features, 1)
                else:
                    features = np.append(features, 0)
        
                    ####################################################################################################################
        
        # beta de número de vehículos con planificación |1|
        '''
        vehículos_con_planificación = 0
        for idv in self.instancia.id_vehiculos:
            if planificacion_post[idv]:
                vehículos_con_planificación += 1
        proporcion_vehiculos_con_planificación = vehículos_con_planificación / len(self.instancia.id_vehiculos)
        features = np.append(features, proporcion_vehiculos_con_planificación)
        '''
        ####################################################################################################################
        
        # beta de número de vehículos que está en camino al depot en caso de que tenga menos del 20% de su inventario utilizado |M|
        '''
        vehículos_en_camino = 0
        for idv in self.instancia.id_vehiculos:
            if not planificacion_post[idv]:
                continue
            elif sum(ratios_utilizacion_vehiculos[idv][idp] for idp in self.instancia.id_productos) < 0.2 and next(iter(planificacion_post[idv])) == 0:
                vehículos_en_camino += 1
        proporcion_vehiculos_en_camino = vehículos_en_camino / len(self.instancia.id_vehiculos)
        features = np.append(features, proporcion_vehiculos_en_camino)
        '''
        ####################################################################################################################


        return features

    def ejecutar_politica_epsilon_greedy(self):
        '''
        Descripción:
        Método que ejecuta un episodio siguiendo la política epsilon greedy
        '''

        # Comenzamos en el estado inicial
        estado = self.proceso.determinar_estado_inicial()
        trayectoria = []
        while True:
            accion = self.tomar_accion_epsilon_greedy(estado)
            features = self.obtener_features(estado, accion)
            nuevo_estado, recompensa, _, _ = self.proceso.transicion(estado, accion)
            trayectoria.append({'estado': estado, 'accion':accion, 'features': features, 'recompensa':recompensa})
            if self.es_terminal(estado):
                break
            else:
                estado = nuevo_estado 
        return trayectoria

    def entrenar_modelo(self):
        ''' Método que ejecuta el algoritmo MC OnPolicy'''
        for episodio in range(self.episodios):
            trayectoria = self.ejecutar_politica_epsilon_greedy()
            # implementamos la política epsilon greedy
            G = 0
            for t in reversed(trayectoria):
                G = G  +  t['recompensa']
                # Actualizamos los pesos
                x = t['features']
                # Ahora obtenemos el c(st,at)
                c_st_at = self.proceso.determinar_c_st_at(t['estado'], t['accion'])
                self.SGD(x, G, c_st_at)
            
            if episodio % 100 == 0:
                self.actualizar_mejores_betas()

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

    def SGD(self,x,y, c):
            '''
            Descripción:
            *   Método que aplica SGD, recibe un array x y actualiza los pesos originados self.betas
            Parámetros:
            *   x: Array con los features de ese estado acción
            *   y: Recompensa obtenida en ese estado acción
            *   c: costo fijo del estado acción
            '''
            # Predicción y error
            prediccion = np.dot(x, self.betas) + c
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
        if numero < self.epsilon:
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
            mejor_accion = np.random.choice(acciones)
            for action in acciones: 
                x = self.obtener_features(state, action) # Lista de features
                # obtenemos el c(st,at)
                c_st_at = self.proceso.determinar_c_st_at(state, action)
                Q = np.dot(x, self.mejores_betas) + c_st_at # Predicción FALTA SUMAR C(St,At)
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
            #lista de tuplas
            mejor_Q = np.inf
            mejor_accion = np.random.choice(acciones)
            for action in acciones: 
                x = self.obtener_features(state, action) # Lista de features
                # obtenemos el c(st,at)
                c_st_at = self.proceso.determinar_c_st_at(state, action)
                Q = np.dot(x, self.betas) + c_st_at # Predicción FALTA SUMAR C(St,At)
                if Q < mejor_Q:
                    mejor_Q = Q
                    mejor_accion = action
            return mejor_accion

    def calcular_peso_por_cliente_producto(self, accion, id_cliente, id_producto):
        '''
        Descripción:
        Método que calcula el peso de entrega de un cliente en relación a un producto
        Args:
            * accion: Acción
            * id_cliente: ID del cliente
            * id_producto: ID del producto
        Return:
            * peso_entrega: Peso de entrega del cliente en relación al producto
        '''
        total_peso = 0.0
        for vehiculo in accion.values():
            # Verificar si el vehículo tiene acción
            if not vehiculo:
                continue  # Saltar vehículos inactivos
            # Obtener destino y productos (seguro porque vehiculo no está vacío)
            id_destino, productos = next(iter(vehiculo.items()))
            if id_destino == id_cliente:
                for idp, cantidad in productos.items():
                    if idp == id_producto:
                        total_peso += cantidad * self.instancia.productos[id_producto].peso
        return total_peso

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

    def obtener_acciones(self, estado:Estado):
        '''
        Descripción:
        Método que devuelve una lista con las acciones posibles para un estado
        
        Args: 
            * estado: Objeto Estado
        Return:
            * acciones_posibles: Lista con las acciones factibles
        '''
        acciones_posibles = []

        # Agregamos la acción 'nula'
        acciones_posibles.append({})

        ############################################### agregamos acción redirigir a depot######################
        # Ahora agregamos la que redirige los vehiculos con stock crítico

        vehiculos_disponibles = self.determinar_vehiculos_disponibles(estado = estado)
        vehiculos_con_quiebre = self.determinar_vehiculos_con_quiebre(estado = estado)

        accion_quiebre = {}
        for idv in vehiculos_disponibles:
            if idv in vehiculos_con_quiebre:
                accion_quiebre[idv] = {0 :{}} # su nueva ruta será a depot con planificación vacía
                break
        # Si hay vehículos con quiebre, agregamos la acción de redirigir a depot
        if accion_quiebre != {}:
            acciones_posibles.append(accion_quiebre)
        ########################################## agregamos las acciones posibles respecto a elegir los 4 clientes con menor ratio y les asignamos hasta 2 vehículos ############
        clientes_disponibles = self.determinar_clientes_disponibles(estado = estado)

        ids_4_clientes_criticos = self.obtener_4_clientes_criticos(estado = estado, clientes_disponibles = clientes_disponibles)

        # ahora determinamos los 2 vehículos más cercanos a cada cliente crítico
        for id_cliente in ids_4_clientes_criticos:
            # Determinamos los vehículos más cercanos
            vehiculos_mas_cercanos = self.determinar_2_vehiculos_mas_cercanos_disponibles(estado = estado, id_cliente = id_cliente, vehiculos_disponibles = vehiculos_disponibles) # AGREGAR VEHÍCULOS DISPONIBLES
            # Si no hay vehículos disponibles, no se agrega la acción y se pasa al siguiente vehículo crítico
            if vehiculos_mas_cercanos == []:
                pass
            else:# Para cada vehículo más cercano, se agrega la acción de asignarle el cliente crítico
                for idv in vehiculos_mas_cercanos:
                    accion = {idv :{id_cliente: self.planificar_entrega(inventario = estado.inventarios_vehiculos[idv], demandas = self.instancia.clientes[id_cliente].demanda_media)}}
                    acciones_posibles.append(accion) 
        return acciones_posibles

    def obtener_4_clientes_criticos(self, estado, clientes_disponibles):
        '''
        Descripción:
            Método que determina los 4 clientes con stock crítico considerando el peso de los productos.
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
