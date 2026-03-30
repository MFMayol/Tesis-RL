from .modelos_avazados import MonteCarlo_Fourier
import numpy as np
from src.Estado import Estado
import math
from typing import Dict, List, Set, Tuple, Any
import itertools

class MCFourierH(MonteCarlo_Fourier):
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate, max_i):
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate, max_i)

    def _generar_planes_parciales(self, plan_completo: dict):
            """
            Método auxiliar para generar planes de entrega combinados (100%, 75%, 50%, 25%).
            
            A partir de un plan de entrega completo, genera combinaciones
            escalando las cantidades de manera INDEPENDIENTE por producto.
            Redondea las cantidades y elimina planes vacíos o duplicados.
            NOTA: Es costoso computacionalmente
            
            Args:
                * plan_completo (dict): {'id_producto': cantidad_total, ...}
                
            Return:
                * planes_generados (list): Lista de diccionarios [plan1, plan2, ...]
            """
            planes_generados = []
            planes_unicos = set() # Para evitar duplicados por redondeo
            factores_entrega = [1.0, 0.75, 0.5, 0.25]
            
            productos = list(plan_completo.keys())
            
            # itertools.product genera todas las combinaciones posibles de factores.
            # Ej: para 2 productos -> (1.0, 1.0), (1.0, 0.75), ... (0.25, 0.25)
            for combinacion_factores in itertools.product(factores_entrega, repeat=len(productos)):
                plan_parcial = {}
                for i, producto in enumerate(productos):
                    cantidad_parcial = round(plan_completo[producto] * combinacion_factores[i])
                    if cantidad_parcial > 0:
                        plan_parcial[producto] = cantidad_parcial
                        
                if not plan_parcial:
                    continue

                plan_hashable = tuple(sorted(plan_parcial.items()))
                
                if plan_hashable not in planes_unicos:
                    planes_unicos.add(plan_hashable)
                    planes_generados.append(plan_parcial)
                    
            return planes_generados
    
    def _obtener_acciones(self, estado: Estado):
        '''
        Descripción:
        Método que devuelve una lista con las acciones posibles para un estado
        Genera todas las acciones posibles para el estado actual, optimizando:
        1. Reducción de llamadas redundantes a métodos.
        2. Uso de estructuras de datos eficientes (sets, generadores).
        3. Minimización de operaciones costosas.
        
        (MODIFICADO): Ahora también genera acciones parciales (75%, 50%, 25%) 
                      para cada acción de entrega.
        
        Args: 
            * estado: Objeto Estado
        Return:
            * acciones_posibles: Lista con las acciones factibles

        '''
        acciones_posibles = [{}]  # Acción nula

        # --- Sección 1: Acción de redirigir vehículos con quiebre de stock al depot ---
        # (Esta sección no cambia)
        vehiculos_disponibles = self._determinar_vehiculos_disponibles(estado)
        vehiculos_con_quiebre = set(self._determinar_vehiculos_con_quiebre(estado))
        
        vehiculo_quiebre = next((idv for idv in vehiculos_disponibles if idv in vehiculos_con_quiebre), None)
        
        if vehiculo_quiebre is not None:
            acciones_posibles.append({vehiculo_quiebre: {0: {}}})

        # --- Sección 2: Acciones para clientes críticos (CON MODIFICACIÓN) ---
        clientes_disponibles = self._determinar_clientes_disponibles(estado)
        criticos = self._obtener_4_clientes_criticos(estado, clientes_disponibles)
        
        demandas_clientes = {idc: self.instancia.clientes[idc].demanda_media for idc in criticos}
        
        # Reemplazamos la list comprehension por bucles explícitos
        # para poder generar las acciones parciales.
        nuevas_acciones = []
        for id_cliente in criticos:
            # Determinamos los vehículos más cercanos para este cliente
            vehiculos_cercanos = self._determinar_2_vehiculos_mas_cercanos_disponibles(
                estado, id_cliente, vehiculos_disponibles
            )
            
            for idv in vehiculos_cercanos:
                # 1. Calcular el plan de entrega COMPLETO (100%)
                plan_completo = self.planificar_entrega(
                    estado.inventarios_vehiculos[idv], 
                    demandas_clientes[id_cliente]
                )
                
                # Si el plan completo está vacío (no se puede entregar nada), 
                # saltamos esta combinación.
                if not plan_completo:
                    continue
                
                # 2. <<< AQUÍ ESTÁ LA MODIFICACIÓN >>>
                # Generar todos los planes (100%, 75%, 50%, 25%)
                # usando el nuevo método auxiliar.
                planes_de_entrega = self._generar_planes_parciales(plan_completo)
                
                # 3. Crear una acción por cada plan de entrega generado
                for plan in planes_de_entrega:
                    accion = {idv: {id_cliente: plan}}
                    nuevas_acciones.append(accion)
        
        acciones_posibles.extend(nuevas_acciones)
        return acciones_posibles





class MCFourierHCORRREGIDO(MonteCarlo_Fourier1D):
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate, max_i):
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate, max_i)

    def _generar_planes_filtrados_por_criticidad(
        self, 
        dict_plan_completo: Dict[int, int],
        dict_costos_penalizacion: Dict[int, float],
        dict_costos_almacenamiento: Dict[int, float]
    ) -> List[Dict[int, int]]:
        """
        Genera hasta 4 planes de entrega alternativos filtrando los productos
        según su ratio de criticidad (penalización / costo de almacenamiento).
        
        A partir de un plan de entrega completo, aplica umbrales de corte 
        (100%, 75%, 50%, 25%) sobre el ranking de productos más críticos.
        
        Args:
            dict_plan_completo: Diccionario con la planificación original {id_producto: cantidad}.
            dict_costos_penalizacion: Diccionario con el costo de penalización por producto.
            dict_costos_almacenamiento: Diccionario con el costo unitario de almacenamiento por producto.
            
        Returns:
            Lista de diccionarios con los planes generados (sin duplicados ni vacíos).
        """
        list_planes_generados: List[Dict[int, int]] = []
        set_planes_unicos: Set[Tuple[Tuple[int, int], ...]] = set()
        list_acciones_agente: List[float] = [1.0, 0.75, 0.5, 0.25]
        
        # Calcular el ratio de criticidad para los productos en el plan
        dict_ratios_criticidad: Dict[int, float] = {}
        for int_id_producto in dict_plan_completo.keys():
            float_penalizacion: float = dict_costos_penalizacion.get(int_id_producto, 0.0)
            float_almacenamiento: float = dict_costos_almacenamiento.get(int_id_producto, 1.0)
            
            # Evitar división por cero si el almacenamiento es gratuito
            if float_almacenamiento <= 0:
                dict_ratios_criticidad[int_id_producto] = float('inf')
            else:
                dict_ratios_criticidad[int_id_producto] = float_penalizacion / float_almacenamiento
                
        # Ordenar productos de mayor a menor criticidad (Rango 1, 2, 3...)
        list_productos_ordenados: List[int] = sorted(
            dict_ratios_criticidad.keys(),
            key=lambda id_prod: dict_ratios_criticidad[id_prod],
            reverse=True
        )
        
        int_total_productos: int = len(list_productos_ordenados)

        # Generar los planes basados en los umbrales de acción
        for float_accion in list_acciones_agente:
            int_limite_corte: int = math.ceil(float_accion * int_total_productos)
            set_productos_aprobados: Set[int] = set(list_productos_ordenados[:int_limite_corte])
            
            dict_plan_parcial: Dict[int, int] = {}
            for int_id_producto, int_cantidad in dict_plan_completo.items():
                if int_id_producto in set_productos_aprobados and int_cantidad > 0:
                    dict_plan_parcial[int_id_producto] = int_cantidad
                    
            # No añadir planes vacíos
            if not dict_plan_parcial:
                continue

            # Evitar duplicados (ej. si 50% y 25% resultan en el mismo corte para 2 productos)
            tuple_plan_hashable = tuple(sorted(dict_plan_parcial.items()))
            
            if tuple_plan_hashable not in set_planes_unicos:
                set_planes_unicos.add(tuple_plan_hashable)
                list_planes_generados.append(dict_plan_parcial)
                
        return list_planes_generados
    
    def _obtener_acciones(self, estado: Any) -> List[Dict[Any, Any]]:
        """
        Genera todas las acciones posibles para el estado actual.
        
        Incorpora la lógica de entregas parciales filtradas por criticidad
        para clientes que requieren reabastecimiento.
        
        Args: 
            estado: Objeto Estado actual del MDP.
            
        Returns:
            Lista de diccionarios con las acciones factibles de ruteo y entrega.
        """
        list_acciones_posibles: List[Dict[Any, Any]] = [{}]  # Acción nula

        # --- Sección 1: Acción de redirigir vehículos con quiebre de stock al depot ---
        list_vehiculos_disponibles = self._determinar_vehiculos_disponibles(estado)
        set_vehiculos_con_quiebre = set(self._determinar_vehiculos_con_quiebre(estado))
        
        int_vehiculo_quiebre = next((idv for idv in list_vehiculos_disponibles if idv in set_vehiculos_con_quiebre), None)
        
        if int_vehiculo_quiebre is not None:
            list_acciones_posibles.append({int_vehiculo_quiebre: {0: {}}})

        # --- Sección 2: Acciones para clientes críticos ---
        list_clientes_disponibles = self._determinar_clientes_disponibles(estado)
        list_clientes_criticos = self._obtener_4_clientes_criticos(estado, list_clientes_disponibles)
        
        dict_demandas_clientes = {idc: self.instancia.clientes[idc].demanda_media for idc in list_clientes_criticos}
        
        list_nuevas_acciones: List[Dict[Any, Any]] = []
        
        for int_id_cliente in list_clientes_criticos:
            # Extraer costos del cliente (ADVERTENCIA: Verifica estos nombres en tu clase Cliente)
            obj_cliente = self.instancia.clientes[int_id_cliente]
            dict_costos_penalizacion = obj_cliente.costos_penalizacion 
            dict_costos_almacenamiento = obj_cliente.costos_inventario
            
            list_vehiculos_cercanos = self._determinar_2_vehiculos_mas_cercanos_disponibles(
                estado, int_id_cliente, list_vehiculos_disponibles
            )
            
            for int_id_vehiculo in list_vehiculos_cercanos:
                # 1. Calcular el plan de entrega COMPLETO (100%)
                dict_plan_completo = self.planificar_entrega(
                    estado.inventarios_vehiculos[int_id_vehiculo], 
                    dict_demandas_clientes[int_id_cliente]
                )
                
                if not dict_plan_completo:
                    continue
                
                # 2. Generar todos los planes aplicando el filtro de criticidad
                list_planes_de_entrega = self._generar_planes_filtrados_por_criticidad(
                    dict_plan_completo,
                    dict_costos_penalizacion,
                    dict_costos_almacenamiento
                )
                
                # 3. Crear una acción por cada plan de entrega generado
                for dict_plan in list_planes_de_entrega:
                    dict_accion = {int_id_vehiculo: {int_id_cliente: dict_plan}}
                    list_nuevas_acciones.append(dict_accion)
        
        list_acciones_posibles.extend(list_nuevas_acciones)
        return list_acciones_posibles
