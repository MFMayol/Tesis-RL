from Politica import PoliticaSimple, PoliticaSimpleClusterisada, RollOutSimple, RollOutCluster, MonteCarlo
import pandas as pd
import os
from Instancia import Instancia
from Proceso import Proceso

class Simulacion:
    """
    Clase para ejecutar una simulación que compara dos políticas: una política simple y una política clusterizada.
    Se encarga de procesar varias instancias, ejecutar las políticas y generar un archivo con los resultados.
    """
    def __init__(self, carpeta_instancias, n_instancias, n_repeticiones):
        """
        Inicializa la simulación con los parámetros especificados.
        
        :param carpeta_instancias: Ruta a la carpeta que contiene las instancias.
        :param n_instancias: Número total de instancias a procesar.
        :param n_repeticiones: Número de veces que se repite cada instancia.
        """
        self.carpeta_instancias = carpeta_instancias
        self.n_instancias = n_instancias
        self.n_repeticiones = n_repeticiones
        self.resultados = {}

    def ejecutar_politicas(self, instancia, proceso):
        """
        Ejecuta las políticas simple y clusterizada sobre una instancia y devuelve los costos asociados.
        
        :param instancia: Objeto de la clase Instancia.
        :param proceso: Objeto de la clase Proceso asociado a la instancia.
        :return: Tupla con los costos de ambas políticas (simple y clusterizada).
        """

        politica_simple = PoliticaSimple(instancia, proceso)
        politica_clusterizada = PoliticaSimpleClusterisada(instancia, proceso)
        PoliticaRolloutSimple = RollOutSimple( instancia= instancia, proceso= proceso)
        PoliticaRolloutCluster = RollOutCluster(instancia, proceso)
        PoliticiaMonteCarlo = MonteCarlo(instancia = instancia, proceso = proceso, episodios = 5000, epsilon=0.01, learning_rate=0.001) # para que converja debe estar en 10^-6  episodios = 10000, epsilon=0.3, learning_rate=0.00001) Esta da bien
        
        PoliticiaMonteCarlo.entrenar_modelo()

        trayectoria_simple, costo_traslado, costo_insatisfecha = politica_simple.run()
        trayectoria_clusterizada, costo_traslado_cluster, costo_insatisfecha_cluster = politica_clusterizada.run()
        trayectoria_rollout_simple,costo_traslado_RolloutSimple,costo_insatisfecha_RolloutSimple = PoliticaRolloutSimple.run()
        trayectoria_rollout_cluster, costo_traslado_RolloutCluster, costo_insatisfecha_RolloutCluster = PoliticaRolloutCluster.run()
        trayectoria_MC, _ = PoliticiaMonteCarlo.run()


        resultado_simple = sum(estado_accion['recompensa'] for estado_accion in trayectoria_simple)
        resultado_cluster = sum(estado_accion['recompensa'] for estado_accion in trayectoria_clusterizada)
        resultado_rollout_simple = sum(estado_accion['recompensa'] for estado_accion in trayectoria_rollout_simple)
        resultado_rollout_cluster = sum(estado_accion['recompensa'] for estado_accion in trayectoria_rollout_cluster)
        resultado_MC = sum(estado_accion['recompensa'] for estado_accion in trayectoria_MC)

        return resultado_simple, costo_traslado, costo_insatisfecha, resultado_cluster, costo_traslado_cluster, costo_insatisfecha_cluster,resultado_rollout_simple, costo_traslado_RolloutSimple, costo_insatisfecha_RolloutSimple, resultado_rollout_cluster, costo_traslado_RolloutCluster, costo_insatisfecha_RolloutCluster
        

    def procesar_instancias(self):
        """
        Procesa cada instancia y ejecuta las políticas en múltiples repeticiones, almacenando los resultados.
        """
        for i in range(self.n_instancias):
            ruta_completa = os.path.join(self.carpeta_instancias, f'instancia{i}.xml')
            self.resultados[f'instancia{i}'] = {
                'costo_total': [], 'costo_traslado': [], 'costo_insatisfecha': [],
                'costo_total_cluster': [], 'costo_traslado_cluster': [], 'costo_insatisfecha_cluster': [], 'Costo_Total_RollOutSimple': [], 'Costo_Traslado_RollOutSimple': [], 'Costo_insatisfecha_RollOutSimple': [],
                'Costo_total_RollOut_Cluster': [], 'Costo_Traslado_RollOut_Cluster': [], 'Costo_Insatisfecha_Rollout_Cluster': []
            }
            
            for _ in range(self.n_repeticiones):
                instancia = Instancia(ruta_archivo=ruta_completa, umbral_inventario_clientes= 0.2, umbral_inventario_vehiculos= 0.2)
                proceso = Proceso(instancia)
                
                (resultado_simple, costo_traslado, costo_insatisfecha,
                resultado_cluster, costo_traslado_cluster, costo_insatisfecha_cluster, resultado_RollOut_simple,costo_traslado_RolloutSimple,costo_insatisfecha_RolloutSimple, costo_rollout_cluster, costo_traslado_rollout_cluster, costo_insatisfecha_rollout_cluster ) = self.ejecutar_politicas(instancia, proceso)
                
                print(f'instancia{i} terminada')
                self.resultados[f'instancia{i}']['costo_total'].append(resultado_simple)
                self.resultados[f'instancia{i}']['costo_traslado'].append(costo_traslado)
                self.resultados[f'instancia{i}']['costo_insatisfecha'].append(costo_insatisfecha)
                
                self.resultados[f'instancia{i}']['costo_total_cluster'].append(resultado_cluster)
                self.resultados[f'instancia{i}']['costo_traslado_cluster'].append(costo_traslado_cluster)
                self.resultados[f'instancia{i}']['costo_insatisfecha_cluster'].append(costo_insatisfecha_cluster)

                self.resultados[f'instancia{i}']['Costo_Total_RollOutSimple'].append(resultado_RollOut_simple)
                self.resultados[f'instancia{i}']['Costo_Traslado_RollOutSimple'].append(costo_traslado_RolloutSimple)
                self.resultados[f'instancia{i}']['Costo_insatisfecha_RollOutSimple'].append(costo_insatisfecha_RolloutSimple)

                self.resultados[f'instancia{i}']['Costo_total_RollOut_Cluster'].append(costo_rollout_cluster)
                self.resultados[f'instancia{i}']['Costo_Traslado_RollOut_Cluster'].append(costo_traslado_rollout_cluster)
                self.resultados[f'instancia{i}']['Costo_Insatisfecha_Rollout_Cluster'].append(costo_insatisfecha_rollout_cluster)

    def generar_dataframe(self):
        """
        Crea un DataFrame de pandas con los promedios de los costos obtenidos en cada instancia.
        
        :return: DataFrame con los resultados organizados por instancia y política.
        """
        resumen_resultados = []
        for instancia, valores in self.resultados.items():
            resumen_resultados.append([
                instancia,
                sum(valores['costo_total']) / self.n_repeticiones,
                sum(valores['costo_traslado']) / self.n_repeticiones,
                sum(valores['costo_insatisfecha']) / self.n_repeticiones,
                sum(valores['costo_total_cluster']) / self.n_repeticiones,
                sum(valores['costo_traslado_cluster']) / self.n_repeticiones,
                sum(valores['costo_insatisfecha_cluster']) / self.n_repeticiones,
                sum(valores['Costo_Total_RollOutSimple']) / self.n_repeticiones,
                sum(valores['Costo_Traslado_RollOutSimple']) / self.n_repeticiones,
                sum(valores['Costo_insatisfecha_RollOutSimple']) / self.n_repeticiones,
                sum(valores['Costo_total_RollOut_Cluster']) / self.n_repeticiones,
                sum(valores['Costo_Traslado_RollOut_Cluster']) / self.n_repeticiones,
                sum(valores['Costo_Insatisfecha_Rollout_Cluster']) / self.n_repeticiones
            ])
        
        return pd.DataFrame(resumen_resultados, columns=[
            "Instancia", "Costo Total Simple", "Costo Traslado Simple", "Costo Demanda Insatisfecha Simple",
            "Costo Total Clusterizado", "Costo Traslado Clusterizado", "Costo Demanda Insatisfecha Clusterizada",
            "Costo total Rollout Simple", "Costo traslado Rollout Simple", "costo demanda insatisfecha Rollout Simple",
            "Costo total RollOut Clusterizado", "Costo traslado Rollout Clusterizado", "Costo demanda insatisfecha Rollout Clusterizado"
        ])

    def ejecutar(self):
        """
        Ejecuta todo el proceso: procesa instancias, genera el DataFrame y guarda los resultados en un archivo Excel.
        """
        self.procesar_instancias()
        df_resultados = self.generar_dataframe()
        df_resultados.to_excel("resultadosPoliticasSimpleYCluster.xlsx", index=False)
        print("Resultados guardados en 'resultados.xlsx'")
        return df_resultados
