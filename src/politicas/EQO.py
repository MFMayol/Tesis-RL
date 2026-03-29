from .simples import PoliticaSimple
import numpy as np
from src.Estado import Estado
from scipy.stats import norm

class EOQSimple(PoliticaSimple):
    '''
    Implementa la política simple de ruteo, pero utiliza el modelo EOQ Estocástico (Q, R)
    para decidir la cantidad a entregar, en lugar de la distribución proporcional.
    '''
    def __init__(self, instancia, proceso, nivel_servicio=0.95):
        super().__init__(instancia, proceso)
        # Calculamos el valor Z para el nivel de servicio (ej: 0.95 -> 1.645)
        self.z_score = norm.ppf(nivel_servicio)

    def tomar_accion(self, estado: Estado):
        '''
        Sobreescribimos tomar_accion para poder pasar el 'id_cliente' y 'estado' 
        a la función de planificación, datos necesarios para el cálculo de EOQ.
        '''
        accion = {} 
        vehiculos_disponibles = self._determinar_vehiculos_disponibles(estado) # Lista de ids
        
        if not vehiculos_disponibles:
            return accion
            
        clientes_disponibles = self._determinar_clientes_disponibles(estado) # Lista de ids
        vehiculos_con_quiebre = self._determinar_vehiculos_con_quiebre(estado) # Lista de ids

        # 1. Asignar retorno a depot si hay quiebre de stock en vehículo
        for idv in vehiculos_disponibles:
            if idv in vehiculos_con_quiebre:
                accion[idv] = {0: {}}
                return accion
            
        # 2. Asignar rutas a clientes críticos
        for idv in vehiculos_disponibles:
            id_cliente_con_inventario_bajo = self._determinar_cliente_critico(estado, clientes_disponibles)
            
            # Si no hay clientes disponibles o críticos, no hacemos nada
            if id_cliente_con_inventario_bajo is None:
                break
                
            # --- CAMBIO PRINCIPAL AQUÍ ---
            # Llamamos al nuevo planificar_entrega pasando el contexto necesario para EOQ
            planificacion_del_vehiculo = self.planificar_entrega(
                estado=estado,
                id_vehiculo=idv,
                id_cliente=id_cliente_con_inventario_bajo
            )
            
            accion[idv] = {id_cliente_con_inventario_bajo: planificacion_del_vehiculo}
            return accion 
            
        return accion

    def planificar_entrega(self, estado: Estado, id_vehiculo: int, id_cliente: int) -> dict:
        """
        Calcula la cantidad a entregar usando la política de revisión continua (Q, R).

        Argumentos:
        - estado: Estado actual del problema.
        - id_vehiculo: ID del vehículo.
        - id_cliente: ID del cliente.
        
        Regla:
        - Si Inventario_Cliente < R: Entregar min(Q*, Capacidad_Vehiculo, Espacio_Cliente)
        - Si Inventario_Cliente >= R: Entregar 0
        """
        cliente = self.instancia.clientes[id_cliente] # objeto cliente
        vehiculo = self.instancia.vehiculos[id_vehiculo] # objeto vehiculo
        inv_vehiculo = estado.inventarios_vehiculos[id_vehiculo] # dict de id:cantidad
        inv_cliente = estado.inventarios_clientes[id_cliente] # dict de id:cantidad
        
        planificacion = {}
        
        # Estimación del Lead Time (L). 
        # En simulación dinámica L es el tiempo de viaje. Usamos la distancia / velocidad media.
        pos_v = np.array([estado.posiciones_vehiculos[id_vehiculo]['x'], estado.posiciones_vehiculos[id_vehiculo]['y']])
        pos_c = np.array([cliente.posicion_x, cliente.posicion_y])
        distancia = np.linalg.norm(pos_v - pos_c)
        
        # Evitar división por cero si velocidad es 0 (raro) o muy baja
        velocidad = max(vehiculo.velocidad_media, 0.1)
        L = distancia / velocidad # Tiempo de viaje estimado (Lead Time)
        
        # Costo de preparación (K). Asumimos costo proporcional a la distancia recorrida para llegar.
        # K = Costo de ir al cliente. (Podría ser fijo, pero usamos distancia como proxy del costo)
        K = distancia # O multiplicar por un costo_por_km si existe en la instancia
        
        for idp in self.instancia.productos: #recorremos los productos 
            # Datos del producto
            mu = cliente.demanda_media[idp]
            sigma = cliente.demanda_desv_est[idp]
            h = cliente.costos_inventario[idp]
            
            # Evitar división por cero en h
            h = max(h, 0.001)
            
            # 1. Calcular Q* (Cantidad Económica de Pedido)
            if mu > 0:
                Q_star = np.sqrt((2 * K * mu) / h)
            else:
                Q_star = 0
                
            # 2. Calcular R (Punto de Reorden)
            stock_seguridad = self.z_score * sigma * np.sqrt(L)
            R = (mu * L) + stock_seguridad
            
            cantidad_a_entregar = 0
            stock_actual = inv_cliente[idp]
            
            # 3. Regla de Decisión
            if stock_actual < R:
                cantidad_ideal = Q_star
                # Ajustamos por lo disponible en el vehículo:
                disponible_vehiculo = inv_vehiculo.get(idp, 0)
                
                cantidad_a_entregar = min(cantidad_ideal, disponible_vehiculo)
                
                # Opcional: No desbordar al cliente (si capacidad es por producto o check global)
                # cantidad_a_entregar = min(cantidad_a_entregar, espacio_en_cliente)
                
                cantidad_a_entregar = max(0, int(round(cantidad_a_entregar)))
            else:
                cantidad_a_entregar = 0
            
            if cantidad_a_entregar > 0:
                planificacion[idp] = cantidad_a_entregar
                
        return planificacion