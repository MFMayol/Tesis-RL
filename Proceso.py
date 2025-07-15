from Instancia import Instancia
from Estado import Estado
import math
import numpy as np
from FuncionesAuxiliares import distancia_euclidiana
import copy
from collections import defaultdict

class Proceso:
    '''
    Clase que implementará la actualización de los estados aplicando las acciones mediante las transiciónes de estados
    Atributos:
        - instancia (Instancia): Objeto instancia 
    Métodos:
        - determinar_estado_inicial: Lee la instancia y devuelve el estado inicial (Estado())
    '''
    def __init__(self, instancia: Instancia):
        self.instancia = instancia # Instancia del problema que obtiene todos los valores fijos del inicio

    def determinar_estado_inicial(self) -> None:
        '''
        Método que inicializa el estado actual del problema. La planificación es vacia, las posiciones de los vehiculos como clientes se leen de la instancia, los inventarios de los clientes como los vehículos se leen 
        
        - Retorna:
            estado : Estado inicial del sistema (Estado())
        '''

        estado = Estado()
        # guardamos los inventarios de los vehiculos
        for idv, vehiculo in self.instancia.vehiculos.items():
            estado.inventarios_vehiculos[idv] = vehiculo.inventario

        # Guardamos los inventarios de los clientes
        for idc, cliente in self.instancia.clientes.items():
            estado.inventarios_clientes[idc] = cliente.inventarios_iniciales

        # Inicializamos las posiciones de los vehiculos
        for idv in self.instancia.vehiculos.keys():
            estado.posiciones_vehiculos[idv] = {'x': self.instancia.depot_X, 'y': self.instancia.depot_Y} 

        # iniciamos el tiempo en 0
        estado.tiempo = 0

        # inicializamos los planes de los vehiculos
        for idv in self.instancia.vehiculos.keys():
            estado.planificacion[idv] = {}
        
        return estado

    def transicion(self, estado_original: Estado, accion):
        '''
        Método que devuelve el siguiente estado del problema asociado a un estado-accion

        Parámetros:
            - estado_original (Estado): Estado actual del problema
            - accion (Accion): Acción que se aplicará al estado
        - Retorna:
            estado_nuevo (Estado): Estado nuevo del problema
            recompensa (float): Recompensa obtenida por
        '''
        estado = copy.deepcopy(estado_original) # creamos una copia para que no afecte a la referencia del estado original
        tiempo_inicial = estado.tiempo # esto se utilizará para hacer que se genere un estado en un intervalo de tiempo determinado
        intervalo_de_tiempo = 0 # se actualizará y cada 5 minutos se creará un estado si no ocurre nada
        costos_de_demanda_insatisfecha = 0 # aquí se alargará el costo de la demanda insatisfecha
        costos_de_almacenamiento = 0 # aquí se alargará el costo de almacenamiento
        costos_de_traslado = 0 # aquí se alargará el costo de traslado
        # Variable para determinar si se generó un estado
        Es_estado = False
        # Actualizamos la planificación según la acción que se toma, 'sobreescribiendo' las planificaciones anteriores y agregando los que no tenían planificación
        self.actualizar_planificacion(estado, accion)
        # aquí se generará el paso del tiempo y se actualizarán los 'estados' entre estados
        #Debemos actualizar los inventarios de los clientes y vehiculos, el tiempo, la planificación y las posiciones de los vehiculos
        while True:
            # Primero aumentamos en una unidad el tiempo
            estado.tiempo += 1
            intervalo_de_tiempo += 1
            # Si el tiempo es mayor al horizonte de tiempo, se termina el ciclo y se redirigen todos al depot
            #if estado.tiempo >= self.instancia.horizonte_tiempo:
             #   for idv in self.instancia.id_vehiculos:
              #      estado.planificacion[idv] = {0: {}}                
               # break
            # Ahora, actualizamos las posiciones de los vehiculos según la velocidad que tengan y actualizamos los inventarios de los clientes/vehiculos según la planificación
            for idv, vehiculo in self.instancia.vehiculos.items():
                # Si el vehiculo no tiene planificación, no se actualiza nada 
                if not estado.planificacion[idv]:
                    continue
                else:
                    # Si tiene planificación, se actualiza todo respecto a la planificación de ese vehiculo
                    posicion_actual = {'x': estado.posiciones_vehiculos[idv]['x'] , 'y': estado.posiciones_vehiculos[idv]['y']}  # Obtenemos la posición actual del vehiculo
                    # Determinamos el id del destiono del vehiculo
                    id_destino = next(iter(estado.planificacion[idv])) # Aquí se obtiene el id del cliente al que se dirige el vehiculo
                    # Primero actualizaré el caso de los que se dirigen al depot
                    if id_destino == 0:
                        destino = (self.instancia.depot_X, self.instancia.depot_Y) # Se actualiza la posición al depot
                        velocidad = vehiculo.velocidad_media  # vehiculo.velocidad_media # podría reemplazarse por una distribución normal por ejemplo
                        estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y'] = self.actualizar_posicion_numpy((posicion_actual['x'],posicion_actual['y']), destino, velocidad, 1)
                        # Si el vehiculo llegó al depot, se actualiza el inventario del vehiculo
                        if (estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y']) == destino:
                            Es_estado = True # Genera que sea un estado la actualización
                            estado.inventarios_vehiculos[idv] = self.instancia.vehiculos[idv].inventario # Se actualiza el inventario del vehiculo segun la instancia (PREGUNTAR)
                            estado.planificacion[idv] = {} # Se elimina la planificación del vehiculo
                    # Ahora, actualizamos el caso de los que se dirigen a un cliente
                    else:
                        destino = (self.instancia.clientes[id_destino].posicion_x, self.instancia.clientes[id_destino].posicion_y) # Recupera las coordenadas de ese cliente al que se dirige
                        velocidad = abs(np.random.normal(vehiculo.velocidad_media, vehiculo.desv_est_velocidad))  # podría reemplazarse por una distribución normal por ejemplo 
                        # Actualizamos la posición del vehiculo según la velocidad y el tiempo transcurrido (1 unidad de tiempo)
                        estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y'] = self.actualizar_posicion_numpy((posicion_actual['x'],posicion_actual['y']), destino, velocidad, 1)
                        # Ahora, determinamos si el vehiculo llegó al destino
                        if (estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y']) == destino:
                            Es_estado = True # Se genera un estado
                            # Determinamos el peso de la entrega planificada
                            peso_entrega = sum(cantidad * self.instancia.productos[idp].peso for idp, cantidad in estado.planificacion[idv][id_destino].items())
                            # Ahora determinaré el inventario utilizado por el cliente
                            inventario_utilizado_cliente = sum(estado.inventarios_clientes[id_destino][idp] * self.instancia.productos[idp].peso for idp in self.instancia.id_productos)
                            # Si el peso de la entrega se puede entregar al cliente, se le entrega toda la planificación y se actualizan los inventarios
                            if peso_entrega <= self.instancia.clientes[id_destino].capacidad_almacenamiento - inventario_utilizado_cliente:
                                for idp, cantidad in estado.planificacion[idv][id_destino].items():
                                    # actualizamos los inventarios de los vehiculos
                                    inventario_vehiculo_referencial = estado.inventarios_vehiculos[idv][idp] - cantidad
                                    estado.inventarios_vehiculos[idv][idp] = max(0,inventario_vehiculo_referencial)
                                    # Actualizamos los inventarios de los clientes acorde a lo que efectivamente le pudo llegar
                                    estado.inventarios_clientes[id_destino][idp] += cantidad
                            else:
                                #if peso_entrega== 0:
                                #    continue
                                # Si el peso de la entrega no se puede entregar al cliente, se entrega lo máximo posible al multiplicar por un Beta que sea como ponderador
                                #else:
                                beta = (self.instancia.clientes[id_destino].capacidad_almacenamiento - inventario_utilizado_cliente) / peso_entrega#sum(estado.planificacion[idv][id_destino][idp]* self.instancia.productos[idp].peso for idp in estado.planificacion[idv][id_destino]) 
                                for idp, cantidad in estado.planificacion[idv][id_destino].items():
                                    inventario_vehiculo_referencial = estado.inventarios_vehiculos[idv][idp] - int(cantidad*beta)
                                    estado.inventarios_vehiculos[idv][idp] = max(0,inventario_vehiculo_referencial)
                                    estado.inventarios_clientes[id_destino][idp] += int(cantidad*beta)  
                                # Se elimina la planificación del vehiculo
                                estado.planificacion[idv] = {}

            # Ahora se determinan los costos de demanda insatisfecha como de inventario
            for idc in self.instancia.id_clientes:
                cliente = self.instancia.clientes[idc]
                inventarios_cliente = estado.inventarios_clientes[idc]
                for idp in self.instancia.id_productos:
                    media = cliente.demanda_media[idp]
                    desviacion = cliente.demanda_desv_est[idp]
                    demanda = int(abs(np.random.normal(media, desviacion)))
                    inventario = inventarios_cliente[idp]
                    demanda_insatisfecha = max(demanda - inventario, 0)
                    exceso_inventario = max(inventario - demanda, 0)
                    costos_de_demanda_insatisfecha += demanda_insatisfecha * cliente.costos_penalizacion[idp]
                    costos_de_almacenamiento += exceso_inventario * cliente.costos_inventario[idp]                    
                    inventarios_cliente[idp] = max(0, inventario - demanda)


                # si el inventario del clientes está bajo el umbral se genera un estado
                #inventario_utilizado_cliente = sum(estado.inventarios_clientes[idc][idp] * self.instancia.productos[idp].peso for idp in self.instancia.id_productos)
                #if estado.inventarios_clientes[idc][idp] < self.instancia.umbral_inventario_clientes:
                    #Es_estado = True

            # Si el tiempo es mayor al horizonte de tiempo, se termina el ciclo y se redirigen todos al depot
            if estado.tiempo >= self.instancia.horizonte_tiempo:
                for idv in self.instancia.id_vehiculos:
                    estado.planificacion[idv] = {0: {}}                
                break
            # Ahora se determina si pasaron más de 5 periodos sin que se genere un estado
            if intervalo_de_tiempo >= 3:
                Es_estado = True

            # Si paso alguna de las condiciones se genera un estado
            if Es_estado:
                break

        # calculamos el costo de traslado de los vehiculos de el estado actual al nuevo estado
        for idv in self.instancia.id_vehiculos:
            posicion_anterior = estado_original.posiciones_vehiculos[idv]
            posicion_nueva = estado.posiciones_vehiculos[idv]
            # calculamos el costo de traslado de los vehiculos de el estado actual al nuevo estado
            costos_de_traslado += np.linalg.norm(np.array([posicion_anterior['x'], posicion_anterior['y']]) - np.array([posicion_nueva['x'], posicion_nueva['y']])) #distancia_euclidiana( punto1 = np.array([posicion_anterior['x'], posicion_anterior['y']]), punto2 = np.array([posicion_nueva['x'], posicion_nueva['y']]))


        # Calculamos la recompensa de la transición
        recompensa = costos_de_traslado + costos_de_demanda_insatisfecha + costos_de_almacenamiento

        return estado, recompensa, costos_de_traslado, costos_de_demanda_insatisfecha



    def determinar_c_st_at(self, estado_original, accion):
        '''método que determina el costo fijo de tomar ese estado acción. Es dado por la planificación post estado
        Args: 
            * estado: Objeto estado
            * accion: Dict que tiene la acción tomada
        Returna:
            * costo: costo fijo de la planificación post estado '''
        estado = copy.deepcopy(estado_original) # creamos una copia para que no afecte a la referencia del estado original
        self.actualizar_planificacion(estado, accion) # actualizamos la planificación
        costo = 0
        # obtenemos los costos de traslado hasta que lleguen al cliente cada vehículo {1: {2: {1:20,2:10} }, 2: {6 : {1:10,2:20}}} key= id_vehiculo, value =plani(dict), plani.key = id_cliente
        for id_vehiculo, planificacion in estado.planificacion.items():
            # vemos el caso de que tenga planificación vacia y le asignamos costo 0
            if planificacion == {}:
                costo += 0
            else:
                # obtenemos el id del cliente
                id_cliente = next(iter(planificacion)) 
                posicion_actual_x, posicion_actual_y = estado.posiciones_vehiculos[id_vehiculo]['x'], estado.posiciones_vehiculos[id_vehiculo]['y']
                if  id_cliente == 0:
                    #ahora calculamos el traslado hacia ese id
                    costo += distancia_euclidiana( punto1 = np.array([posicion_actual_x, posicion_actual_y]), punto2 = np.array([self.instancia.depot_X, self.instancia.depot_Y]))
                else:
                    #ahora calculamos el traslado hacia ese id de cliente
                    costo += distancia_euclidiana( punto1 = np.array([posicion_actual_x, posicion_actual_y]), punto2 = np.array([self.instancia.clientes[id_cliente].posicion_x, self.instancia.clientes[id_cliente].posicion_y]))
                    #costo += np.linalg.norm(np.array([posicion_actual_x, posicion_actual_y]) - np.array([self.instancia.clientes[id_cliente].posicion_x, self.instancia.clientes[id_cliente].posicion_y])) #distancia_euclidiana( punto1 = np.array([posicion_anterior['x'], posicion_anterior['y']]), punto2 = np.array([posicion_nueva['x'], posicion_nueva['y']]))

            return costo


#NO SE USA DE MOMENTO
    def actualizar_datos_vehiculos_a_depot(self, estado: Estado, idv: int, vehiculo, posicion_actual: dict):
        '''
        Método que actualiza los datos de los vehiculos cuando van al depot y devuelve los costos de traslado del vehiculo
        '''
        destino = (self.instancia.depot_X, self.instancia.depot_Y)
        velocidad = vehiculo.velocidad_media
        estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y'] = self.actualizar_posicion(posicion_actual, destino, velocidad, 1) # actualizamos la posicion del vehiculo

        #Se agrega el costo de traslado
        costos_de_traslado += distancia_euclidiana(np.array([posicion_actual['x'], posicion_actual['y']]), np.array([estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y']]))

        return costos_de_traslado, 


    def actualizar_planificacion(self, estado: Estado, accion):
        '''
        Método que actualiza la planificacion
        '''
            # Actualizamos la planificación según la acción que se toma, 'sobreescribiendo' la planificación anterior
        for idv, planificacion in accion.items():
            estado.planificacion[idv] = planificacion

    def actualizar_posicion(self, posicion_inicial, destino, velocidad, tiempo):
        """
        Actualiza la posición de un vehículo en movimiento rectilíneo uniforme,
        asegurando que no sobrepase el destino.

        Args:
            posicion_inicial: Tuple con las coordenadas iniciales (x, y).
            destino: Tuple con las coordenadas del destino (x, y).
            velocidad: Velocidad del vehículo en unidades de distancia por tiempo.
            tiempo: Tiempo transcurrido desde la posición inicial.

        Returns:
            Tuple con las nuevas coordenadas (x, y).
        """
        # Calcula el vector de dirección (destino - origen)
        vector_direccion = (destino[0] - posicion_inicial[0], destino[1] - posicion_inicial[1])

        # Calcula la magnitud del vector de dirección (distancia total hasta el destino)
        distancia_total = math.sqrt(vector_direccion[0]**2 + vector_direccion[1]**2)

        # Si ya estamos en el destino, retornamos la misma posición
        if distancia_total == 0:
            return posicion_inicial

        # Calcula el desplazamiento en base a la velocidad y el tiempo
        desplazamiento = velocidad * tiempo

        # Si el desplazamiento es mayor que la distancia total,
        # el vehículo llegará exactamente al destino
        if desplazamiento >= distancia_total:
            return destino

        # Normaliza el vector de dirección (para obtener un vector unitario)
        vector_unitario = (vector_direccion[0] / distancia_total, 
                        vector_direccion[1] / distancia_total)

        # Calcula las nuevas coordenadas sumando el desplazamiento al vector unitario
        nueva_x = posicion_inicial[0] + desplazamiento * vector_unitario[0]
        nueva_y = posicion_inicial[1] + desplazamiento * vector_unitario[1]

        return (nueva_x, nueva_y)
    
    def actualizar_posicion_numpy(self,posicion_inicial, destino, velocidad, tiempo):
        """
        Actualiza la posición de un vehículo en movimiento rectilíneo uniforme,
        asegurando que no sobrepase el destino (versión NumPy).

        Args:
            posicion_inicial: Array de NumPy con las coordenadas iniciales (x, y).
            destino: Array de NumPy con las coordenadas del destino (x, y).
            velocidad: Velocidad del vehículo en unidades de distancia por tiempo.
            tiempo: Tiempo transcurrido desde la posición inicial.

        Returns:
            Array de NumPy con las nuevas coordenadas (x, y).
        """
    
        # Convertimos las posiciones a arrays de NumPy para poder operar fácilmente con vectores
        posicion_inicial = np.array(posicion_inicial)
        destino = np.array(destino)
        
        # Calculamos el vector desde la posición actual hasta el destino
        vector_direccion = destino - posicion_inicial

        # Calculamos la distancia total entre la posición actual y el destino
        distancia_total = np.linalg.norm(vector_direccion)

        # Si ya estamos en el destino (distancia cero), no hay que moverse
        if distancia_total == 0:
            return posicion_inicial

        # Calculamos cuánto se puede avanzar en este tiempo, según la velocidad
        distancia_a_recorrer = velocidad * tiempo

        # Si el vehículo puede llegar o pasar el destino en este paso, se ubica exactamente en el destino
        if distancia_a_recorrer >= distancia_total:
            return destino

        # Calculamos el vector unitario en la dirección al destino (dirección normalizada)
        vector_unitario = vector_direccion / distancia_total

        # Avanzamos desde la posición actual en la dirección correcta, una distancia proporcional
        nueva_posicion = posicion_inicial + vector_unitario * distancia_a_recorrer

        # Devolvemos la nueva posición como array de NumPy
        return nueva_posicion