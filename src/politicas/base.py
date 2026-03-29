from src.Estado import Estado
from abc import ABC, abstractmethod

class Politica:
    '''
    Clase padre que implementa la politica de decisiones y ejecuta la simulación del Proceso
    '''
    def __init__(self, instancia, proceso):
        self.instancia = instancia
        self.proceso = proceso
        self.periodos = 0

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
        costo_almacenamiento = 0
        periodos = 0
        # Se ejecuta hasta que se encuentre un estado terminal
        while True:
            accion = self.tomar_accion(estado)
            estado_nuevo, recompensa, costos_de_traslado, costos_de_demanda_insatisfecha, costos_almac = self.proceso.transicion(estado, accion)
            trayectoria.append({'estado': estado, 'accion': accion, 'recompensa': recompensa})
            costo_traslado += costos_de_traslado
            costos_de_insatisfecha += costos_de_demanda_insatisfecha
            costo_almacenamiento += costos_almac
            costo_total += recompensa
            periodos += 1
            if self.es_terminal(estado_nuevo):
                break
            else:
                estado = estado_nuevo
        return trayectoria , costo_traslado, costos_de_insatisfecha, costo_almacenamiento, periodos

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
