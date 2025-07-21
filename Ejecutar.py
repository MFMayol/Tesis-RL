import os
import numpy as np
import pandas as pd
from Creador import generar_instancia_irp  
from Instancia import Instancia
from Proceso import Proceso
from Politica import PoliticaSimple, PoliticaSimpleClusterisada, RollOutSimple, RollOutCluster, MonteCarlo, MonteCarloRN, Qlearning, MonteCarlo_Fourier, Qlearning_Fourier
from FuncionesAuxiliares import kmeans_clustering, visualizar_clusters, kmeans_clustering_sklearn
from Generador_de_instancias import n_instancias, generar_instancias, semillas
import os

if __name__ == "__main__":
    print("Comenzamos a ejecutar las instancias...")
    instancias = 3
    carpeta_instancias = "Instancias demandas bajas 25C 400"
    resultados = []

    for i in range(1,instancias+1):
        resultados_instancia = {"Instancia": i}
        ruta_completa = os.path.join(carpeta_instancias, f'instancia{i}.xml')
        instancia = Instancia(ruta_archivo=ruta_completa, umbral_inventario_clientes=0.2, umbral_inventario_vehiculos=0.1)
        proceso = Proceso(instancia)

        # -------------------- Política Simple --------------------
        politicaSimple = PoliticaSimple(instancia, proceso)
        costos_simple = []
        for _ in range(100):
            trayectoriaSimple, _, _ = politicaSimple.run()
            costo = sum(estado_accion['recompensa'] for estado_accion in trayectoriaSimple)
            costos_simple.append(costo)
        resultados_instancia["FO S"] = np.mean(costos_simple)

        # -------------------- Política Clusterizada --------------------
        politicaCluster = PoliticaSimpleClusterisada(instancia, proceso)
        costos_cluster = []
        for _ in range(100):
            trayectoriaCluster, _, _ = politicaCluster.run()
            costo = sum(estado_accion['recompensa'] for estado_accion in trayectoriaCluster)
            costos_cluster.append(costo)
        resultados_instancia["FO C"] = np.mean(costos_cluster)

        # -------------------- Monte Carlo Clásico --------------------
        MC = MonteCarlo(instancia=instancia, proceso=proceso, episodios=3000, epsilon=0.05, learning_rate=0.0001)
        MC.entrenar_modelo()
        resultados_instancia["FO MC"] = MC.optimo_mejor_betas

        # -------------------- Monte Carlo Fourier --------------------
        MC_fourier = MonteCarlo_Fourier(instancia=instancia, proceso=proceso, episodios=2000, epsilon=0.05, learning_rate=0.0001, max_i=5)
        MC_fourier.entrenar_modelo()
        resultados_instancia["FO MC fourier"] = MC_fourier.optimo_mejor_betas

        # ----------------------Qlearning --------------------------------------------------------------
        Q_learning = Qlearning(instancia = instancia, proceso = proceso, episodios = 2000, epsilon = 0.05, learning_rate = 0.001, gamma = 0.96)
        Q_learning.entrenar_modelo()
        resultados_instancia["FO Q learning Normal"] = Q_learning.optimo_mejor_betas

        print(f"instancia {i} terminada")


        # -----------------------QlearningFourier ------------------------------------------------------------
        Qfourier = Qlearning_Fourier(instancia = instancia, proceso = proceso, episodios= 2000, epsilon = 0.05, learning_rate = 0.001, gamma = 0.96, max_i=5)
        Qfourier.entrenar_modelo()
        resultados_instancia["QL F"] = Qfourier.optimo_mejor_betas

        # -------------------- Guardar resultados en lista --------------------
        resultados.append(resultados_instancia)

    # -------------------- Crear DataFrame --------------------
    df_resultados = pd.DataFrame(resultados)

    # -------------------- Guardar en Excel --------------------
    nombre_archivo = "Resultados_Cluster_Qlearning.xlsx"
    df_resultados.to_excel(nombre_archivo, index=False)

    print(f"Archivo '{nombre_archivo}' creado exitosamente con {len(df_resultados)} instancias.")