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
    instancias_inicio = 1
    instancias_final = 5
    carpeta_instancias = "Instancias demandas bajas 25C 400"
    resultados = []

    for i in range(instancias_inicio,instancias_final+1):
        resultados_instancia = {"Instancia": i}
        ruta_completa = os.path.join(carpeta_instancias, f'instancia{i}.xml')
        instancia = Instancia(ruta_archivo=ruta_completa, umbral_inventario_clientes=0.2, umbral_inventario_vehiculos=0.1)
        proceso = Proceso(instancia)


        # -----------------------QlearningFourier ------------------------------------------------------------
        Qfourier = Qlearning_Fourier(instancia = instancia, proceso = proceso, episodios= 2000, epsilon = 0.05, learning_rate = 0.001, gamma = 0.96, max_i=5)
        Qfourier.entrenar_modelo()
        resultados_instancia["QL F"] = Qfourier.optimo_mejor_betas
        resultados_instancia["T train"] = Qfourier.tiempo_entrenamiento

        print(f"instancia {i} terminada")

        # -------------------- Guardar resultados en lista --------------------
        resultados.append(resultados_instancia)



    # -------------------- Crear DataFrame --------------------
    df_resultados = pd.DataFrame(resultados)

    # -------------------- Guardar en Excel --------------------
    nombre_archivo = "Resultados_Cluster_QLF.xlsx"
    df_resultados.to_excel(nombre_archivo, index=False)

    print(f"Archivo '{nombre_archivo}' creado exitosamente con {len(df_resultados)} instancias.")