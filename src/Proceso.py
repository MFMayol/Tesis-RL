from src.Instancia import Instancia
from src.Estado import Estado
import math
import numpy as np
from src.FuncionesAuxiliares import distancia_euclidiana
import copy
import random
from collections import defaultdict

class Proceso:
    '''
    Clase que implementará la actualización de los estados aplicando las acciones mediante las transiciónes de estados
    Atributos:
        - instancia (Instancia): Objeto instancia 
    Métodos:
        - determinar_estado_inicial: Lee la instancia y devuelve el estado inicial (Estado())
        - transicion: toma un estado accion y devuelve el nuevo estado + recompensa
        - determinar c_st_at: determina el "c" que utilizamos para ayudar a obtener el costo futuro dado el estado y la acción
        - actualizar_planificacion: actualiza la planificación según la acción tomada
        - actualizar_posicion_numpy: actualiza la posición de los vehículos según la acción tomada
    '''
    def __init__(self, instancia: Instancia):
        self.instancia = instancia # Instancia del problema que obtiene todos los valores fijos del inicio

    def determinar_estado_inicial(self) -> Estado:
        '''
        Método que inicializa el estado actual del problema. La planificación es vacia, las posiciones de los vehiculos como clientes se leen de la instancia, los inventarios de los clientes como los vehículos se leen 
        
        - Retorna:
            estado : Estado inicial del sistema (Estado())
        '''

        estado = Estado()
        # guardamos los inventarios de los vehiculos
        for idv, vehiculo in self.instancia.vehiculos.items():
            estado.inventarios_vehiculos[idv] = vehiculo.inventario.copy()

        # Guardamos los inventarios de los clientes
        for idc, cliente in self.instancia.clientes.items():
            estado.inventarios_clientes[idc] = cliente.inventarios_iniciales.copy()

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
        # Creamos una copia manual optimizada encapsulada en la clase Estado
        estado = estado_original.copiar()

        intervalo_de_tiempo = 0 # se actualizará y cada 5 minutos se creará un estado si no ocurre nada
        costos_de_demanda_insatisfecha = 0 # aquí se alargará el costo de la demanda insatisfecha
        costos_de_almacenamiento = 0 # aquí se alargará el costo de almacenamiento
        costos_de_traslado = 0 # aquí se alargará el costo de traslado
        # Variable para determinar si se generó un estado
        Es_estado = False
        # Actualizamos la planificación según la acción que se toma, 'sobreescribiendo' las planificaciones anteriores y agregando los que no tenían planificación
        self.actualizar_planificacion(estado, accion)

        # Pre-computar pesos y posiciones repetidas
        pesos_productos = {idp: self.instancia.productos[idp].peso for idp in self.instancia.id_productos}
        depot_destino = (self.instancia.depot_X, self.instancia.depot_Y)

        # aquí se generará el paso del tiempo y se actualizarán los 'estados' entre estados
        #Debemos actualizar los inventarios de los clientes y vehiculos, el tiempo, la planificación y las posiciones de los vehiculos
        while True:
            # Primero aumentamos en una unidad el tiempo
            estado.tiempo += 1
            intervalo_de_tiempo += 1 # esto es para ver si pasan los 5 periodos sin generar estado

            # Ahora, actualizamos las posiciones de los vehiculos según la velocidad que tengan y actualizamos los inventarios de los clientes/vehiculos según la planificación
            for idv, vehiculo in self.instancia.vehiculos.items():
                # Si el vehiculo no tiene planificación, no se actualiza nada 
                if not estado.planificacion[idv]:
                    continue
                else:
                    # Si tiene planificación, se actualiza todo respecto a la planificación de ese vehiculo
                    posicion_actual = (estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y'])
                    # Determinamos el id del destino del vehiculo
                    id_destino = next(iter(estado.planificacion[idv])) # Aquí se obtiene el id del destino al que se dirige el vehiculo
                    
                    # random.gauss es mucho más rápido que np.random.normal para números escalares
                    velocidad = abs(random.gauss(vehiculo.velocidad_media, vehiculo.desv_est_velocidad))

                    # Primero actualizaré el caso de los que se dirigen al depot
                    if id_destino == 0:
                        destino = depot_destino
                        nueva_x, nueva_y = self.actualizar_posicion(posicion_actual, destino, velocidad)
                        estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y'] = nueva_x, nueva_y
                        # Si el vehiculo llegó al depot, se actualiza el inventario del vehiculo
                        if (nueva_x, nueva_y) == destino:
                            Es_estado = True # Genera que sea un estado la actualización
                            estado.inventarios_vehiculos[idv] = self.instancia.vehiculos[idv].inventario.copy() # Ojo, se agrega copy para prevenir mutación referencial
                            estado.planificacion[idv] = {} # Se elimina la planificación del vehiculo ya que llegó a destino
                    # Ahora, actualizamos el caso de los que se dirigen a un cliente
                    else:
                        cliente_destino = self.instancia.clientes[id_destino]
                        destino = (cliente_destino.posicion_x, cliente_destino.posicion_y)
                        nueva_x, nueva_y = self.actualizar_posicion(posicion_actual, destino, velocidad)
                        estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y'] = nueva_x, nueva_y
                        
                        if (nueva_x, nueva_y) == destino:
                            Es_estado = True # Se genera un estado
                            peso_entrega = sum(cantidad * pesos_productos[idp] for idp, cantidad in estado.planificacion[idv][id_destino].items())
                            inventario_utilizado_cliente = sum(estado.inventarios_clientes[id_destino][idp] * pesos_productos[idp] for idp in self.instancia.id_productos)
                            
                            if peso_entrega <= cliente_destino.capacidad_almacenamiento - inventario_utilizado_cliente:
                                for idp, cantidad in estado.planificacion[idv][id_destino].items():
                                    # actualizamos los inventarios de los vehiculos
                                    estado.inventarios_vehiculos[idv][idp] = estado.inventarios_vehiculos[idv][idp] - cantidad
                                    # Actualizamos los inventarios de los clientes acorde a lo que efectivamente le pudo llegar
                                    estado.inventarios_clientes[id_destino][idp] += cantidad
                            else:
                                peso_entrega_seguro = max(peso_entrega, 1e-6)
                                beta = (cliente_destino.capacidad_almacenamiento - inventario_utilizado_cliente) / peso_entrega_seguro 
                                for idp, cantidad in estado.planificacion[idv][id_destino].items():
                                    cant_entregar = int(cantidad * beta)
                                    estado.inventarios_vehiculos[idv][idp] -= cant_entregar
                                    estado.inventarios_clientes[id_destino][idp] += cant_entregar
                                # Se elimina la planificación del vehiculo
                                estado.planificacion[idv] = {}

            # Ahora ocurre la demanda y se determinan los costos de inventario/demanda insatisfecha para cada cliente
            for idc in self.instancia.id_clientes:
                cliente = self.instancia.clientes[idc] # guardamos el objeto cliente
                inventarios_cliente = estado.inventarios_clientes[idc] # el inventario también
                for idp in self.instancia.id_productos: # recorremos los productos de ese cliente
                    media = cliente.demanda_media[idp] # media del producto-cliente
                    desviacion = cliente.demanda_desv_est[idp] # desviación del producto-cliente
                    demanda = int(abs(random.gauss(media, desviacion))) # se genera la demanda de manera más rápida
                    inventario = inventarios_cliente[idp] # guardo el inventario de producto
                    
                    if demanda > inventario:
                        costos_de_demanda_insatisfecha += (demanda - inventario) * cliente.costos_penalizacion[idp]
                        inventarios_cliente[idp] = 0
                    else:
                        costos_de_almacenamiento += (inventario - demanda) * cliente.costos_inventario[idp]
                        inventarios_cliente[idp] = inventario - demanda

                ## si el inventario del producto-cliente está bajo el umbral se genera un estado
                inventario_utilizado_cliente = sum(estado.inventarios_clientes[idc][idp] * pesos_productos[idp] for idp in self.instancia.id_productos)
                if inventario_utilizado_cliente < (cliente.capacidad_almacenamiento * self.instancia.umbral_inventario_clientes):
                    Es_estado = True

            # Si el tiempo es mayor al horizonte de tiempo, se termina el ciclo y se redirigen todos al depot
            if estado.tiempo >= self.instancia.horizonte_tiempo:
                for idv in self.instancia.id_vehiculos:
                    estado.planificacion[idv] = {0: {}}                
                Es_estado = True

            # Ahora se determina si pasaron más de 5 periodos sin que se genere un estado
            if intervalo_de_tiempo >= 5:
                Es_estado = True

            # Si paso alguna de las condiciones se deja de pasar el "tiempo" (sale del while) y genera un estado
            if Es_estado:
                break

        # calculamos el costo de traslado de los vehiculos de el estado actual al nuevo estado
        for idv in self.instancia.id_vehiculos:
            posicion_anterior = estado_original.posiciones_vehiculos[idv] # donde comenzó (map)
            posicion_nueva = estado.posiciones_vehiculos[idv] # donde llegó pasado el tiempo (map)
            # math.hypot es significativamente más veloz que np.linalg.norm
            costos_de_traslado += math.hypot(posicion_anterior['x'] - posicion_nueva['x'], posicion_anterior['y'] - posicion_nueva['y'])
            
        # Calculamos la recompensa de la transición
        recompensa = costos_de_traslado + costos_de_demanda_insatisfecha + costos_de_almacenamiento
        return estado, recompensa, costos_de_traslado, costos_de_demanda_insatisfecha, costos_de_almacenamiento

    def determinar_c_st_at(self, estado_original, accion):
        '''método que determina el costo fijo de tomar ese estado acción. Es dado por la planificación post estado
        Args: 
            * estado: Objeto estado
            * accion: Dict que tiene la acción tomada
        Returna:
            * costo: costo fijo de la planificación post estado que se utiliza para calcular el costo futuro'''
        costo = 0
        # Combinar la acción temporalmente sin hacer una copia completa del estado (mejora de rendimiento enorme)
        planificacion_eval = {idv: estado_original.planificacion.get(idv, {}) for idv in self.instancia.id_vehiculos}
        for idv, plan in accion.items():
            planificacion_eval[idv] = plan
            
        for id_vehiculo, planificacion in planificacion_eval.items():
            if not planificacion:
                continue
                
            id_cliente = next(iter(planificacion)) 
            pos_x, pos_y = estado_original.posiciones_vehiculos[id_vehiculo]['x'], estado_original.posiciones_vehiculos[id_vehiculo]['y']
            
            if id_cliente == 0:
                costo += math.hypot(pos_x - self.instancia.depot_X, pos_y - self.instancia.depot_Y)
            else:
                cliente = self.instancia.clientes[id_cliente]
                costo += math.hypot(pos_x - cliente.posicion_x, pos_y - cliente.posicion_y)

        return costo

    def actualizar_planificacion(self, estado: Estado, accion):
        '''
        Método que actualiza la planificacion
        '''
            # Actualizamos la planificación según la acción que se toma, 'sobreescribiendo' la planificación anterior
        for idv, planificacion in accion.items():
            estado.planificacion[idv] = planificacion

    def actualizar_posicion(self,posicion_inicial, destino, velocidad):
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
    
        x1, y1 = posicion_inicial
        x2, y2 = destino
        
        dx = x2 - x1
        dy = y2 - y1
        distancia_total = math.hypot(dx, dy)

        # Si ya estamos en el destino (distancia cero), no hay que moverse
        if distancia_total == 0:
            return x1, y1


        # Calculamos cuánto se puede avanzar en este tiempo, según la velocidad
        distancia_a_recorrer = velocidad

        # Si el vehículo puede llegar o pasar el destino en este paso, se ubica exactamente en el destino
        if distancia_a_recorrer >= distancia_total:
            return x2, y2

        ratio = distancia_a_recorrer / distancia_total
        return x1 + dx * ratio, y1 + dy * ratio