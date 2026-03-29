from .simples import PoliticaSimple
from src.Estado import Estado
import numpy as np
from scipy.stats import norm

class EOQEstocastico(PoliticaSimple):
    '''
    Política (Q, R) Estocástica Dinámica.
    Calcula Q y R en tiempo real considerando que el Lead Time y el Costo de Ordenar (K)
    dependen de la posición actual del vehículo respecto al cliente.
    '''
    def __init__(self, instancia, proceso, max_iter=15, tol=1e-3):
        super().__init__(instancia, proceso)
        self.max_iter = max_iter
        self.tol = tol

    def normal_loss_function(self, z):
        """
        Función de pérdida normal L(z).
        Optimizado: norm.pdf y cdf son algo lentos, para ultra-rendimiento 
        se podría usar una aproximación racional, pero esto es estándar.
        """
        return norm.pdf(z) - z * (1 - norm.cdf(z))

    def tomar_accion(self, estado: Estado):
        '''
        Toma decisiones calculando (Q, R) dinámicamente según la posición del vehículo.
        '''
        accion = {} 
        vehiculos_disponibles = self._determinar_vehiculos_disponibles(estado)
        
        if not vehiculos_disponibles:
            return accion
            
        clientes_disponibles = self._determinar_clientes_disponibles(estado)
        vehiculos_con_quiebre = self._determinar_vehiculos_con_quiebre(estado)

        # 1. Retorno a Depot si necesario (Prioridad 1)
        for idv in vehiculos_disponibles:
            if idv in vehiculos_con_quiebre:
                accion[idv] = {0: {}}
                return accion
            
        # 2. Evaluación Dinámica (Q, R)
        for idv in vehiculos_disponibles:
            # Identificamos el candidato más crítico (heurística de la clase padre)
            id_cliente = self._determinar_cliente_critico(estado, clientes_disponibles)
            
            if id_cliente is None:
                continue

            # Calculamos la entrega óptima para ESTE vehículo en ESTA posición
            planificacion = self.calcular_entrega_qr_dinamico(estado, idv, id_cliente)
            
            if sum(planificacion.values()) > 0:
                accion[idv] = {id_cliente: planificacion}
                return accion 
            
        return accion

    def calcular_entrega_qr_dinamico(self, estado: Estado, id_vehiculo: int, id_cliente: int) -> dict:
        """
        Calcula Q* y R* iterando hasta convergencia.
        Usa L y K basados en la distancia REAL actual.
        """
        cliente = self.instancia.clientes[id_cliente]
        vehiculo = self.instancia.vehiculos[id_vehiculo]
        inv_vehiculo = estado.inventarios_vehiculos[id_vehiculo]
        inv_cliente = estado.inventarios_clientes[id_cliente]
        
        planificacion = {}
        
        # --- 1. Calcular Parámetros Dinámicos (L y K) ---
        pos_v = np.array([estado.posiciones_vehiculos[id_vehiculo]['x'], estado.posiciones_vehiculos[id_vehiculo]['y']])
        pos_c = np.array([cliente.posicion_x, cliente.posicion_y])
        
        distancia = np.linalg.norm(pos_v - pos_c)
        velocidad = max(vehiculo.velocidad_media, 1.0) # Evitar div/0
        
        # Lead Time Dinámico: Tiempo de viaje actual
        L = max(distancia / velocidad, 0.05) # Mínimo L pequeño para evitar singularidades
        
        # Costo de Ordenar Dinámico (K): Proxy = Distancia
        # Si estoy lejos, el "costo de setup" es alto.
        K = max(distancia, 1.0)

        for idp in self.instancia.productos:
            # Datos del producto
            D = cliente.demanda_media[idp]
            sigma_D = cliente.demanda_desv_est[idp]
            h = max(cliente.costos_inventario[idp], 0.001)
            p = cliente.costos_penalizacion[idp]

            if D <= 0:
                planificacion[idp] = 0
                continue

            # Parámetros en Lead Time Dinámico
            mu_L = D * L
            sigma_L = sigma_D * np.sqrt(L)

            # --- 2. Algoritmo Iterativo (Q, R) ---
            # Inicialización: EOQ Clásico
            Q = np.sqrt((2 * K * D) / h)
            R = 0
            
            # Iteración (Limitada para eficiencia)
            prev_Q = -1.0
            
            for _ in range(self.max_iter):
                if abs(Q - prev_Q) < self.tol:
                    break
                prev_Q = Q
                
                # A. Probabilidad de déficit (P_stockout)
                # Condición de optimalidad: P(x > R) = hQ / pD
                if p * D > 0:
                    prob_stockout = (h * Q) / (p * D)
                else:
                    prob_stockout = 0.0 # Si no hay costo penalización, no permitimos stockout
                
                # Clipping: La probabilidad debe estar en (0, 1)
                # Si hQ > pD, significa que mantener es muy caro o penalizar muy barato -> stockout seguro
                prob_stockout = min(max(prob_stockout, 1e-6), 1.0 - 1e-6)  # determinamos el 1-F(R) = prob
                
                # B. Nuevo R
                z = norm.ppf(1 - prob_stockout) # determinamos el z dada la inversa de F(R)
                R = mu_L + z * sigma_L # Calculamos el R
                
                # C. Déficit esperado n(R)
                Lz = self.normal_loss_function(z)
                n_R = sigma_L * Lz # determinamos el n(R)
                
                # D. Nuevo Q (EOQ ajustado por costo de déficit)
                costo_ajustado = K + (p * n_R)
                Q = np.sqrt((2 * D * costo_ajustado) / h)
            
            # --- 3. Decisión de Entrega ---
            stock_actual = inv_cliente[idp]
            
            # Regla de Reorden Dinámica:
            # Si el vehículo está lejos (L grande), R es grande -> "Pido antes"
            # Si el vehículo está cerca (L pequeño), R es pequeño -> "Espero más"
            if stock_actual < R:
                cantidad_ideal = Q
                disponible_vehiculo = inv_vehiculo.get(idp, 0)
                
                cantidad_real = min(cantidad_ideal, disponible_vehiculo)
                planificacion[idp] = max(0, int(cantidad_real))
            else:
                planificacion[idp] = 0

        return planificacion