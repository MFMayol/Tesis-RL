from Creador import generar_instancia_irp
import numpy as np

# Creación de 10 instancias base
n_instancias= 20

semillas = range(n_instancias)

n_clientes = [  15 for _ in range(n_instancias)] # int(np.random.uniform(9,14)

n_vehiculos = [4 for N in n_clientes ] # max(int(N/3), int(N/4)+1)   hay que considerar que debe ser partido en 3 con tal de que en el peor de los casos se tienen 6 clientes y 2 vehículos con tal de que existan mínimo 4 clientes sin atenderse.

n_productos = [ 2 for _ in range(n_instancias) ]  # int(np.random.uniform(1,4))

def generar_instancias(semillas):
    for n in semillas:
        generar_instancia_irp(
            ancho_zona = 2000, # 20 km radio de la zona
            largo_zona = 2000, # 20 km radio de la zona
            horizonte_tiempo = 300, # 480 acorde al paper del profesor
            num_productos = n_productos[n], 
            num_clientes = n_clientes[n], #4
            num_vehiculos = n_vehiculos[n], #2
            semilla = n, # la 5 es para explicar facil 
            nombre_archivo= f'instancia{n}.xml',
            carpeta_instancias= "Instancias demandas bajas"
        )
    print('Se generaron los archivos de instancia correctamente')

