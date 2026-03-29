import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import os
import math

def generar_instancia_irp(
        ancho_zona: int,
        largo_zona: int,
        horizonte_tiempo: int,
        num_productos: int,
        num_clientes: int,
        num_vehiculos: int,
        semilla: int,
        carpeta_instancias: str,
        nombre_archivo="instancia_irp.xml",
        distribucion_geo: str = "GU",
    ):
    """
    Función principal para generar una instancia del problema IRP multiproducto.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    
    root = ET.Element("InstanciaIRP")
    
    info_general = ET.SubElement(root, "InformacionGeneral")
    ET.SubElement(info_general, "AnchoZona").text = str(ancho_zona)
    ET.SubElement(info_general, "LargoZona").text = str(largo_zona)
    ET.SubElement(info_general, "Depot_x").text = str(ancho_zona / 2)
    ET.SubElement(info_general, "Depot_y").text = str(largo_zona / 2)
    ET.SubElement(info_general, "HorizonteTiempo").text = str(horizonte_tiempo)
    
    productos = ET.SubElement(root, "Productos")
    for i in range(num_productos):
        producto = ET.SubElement(productos, "Producto")
        producto.set("id", str(i + 1))
        ET.SubElement(producto, "Peso").text = str(random.choice([1]))

    clientes = ET.SubElement(root, "Clientes")

    centroides = []
    if distribucion_geo == "GC":
        num_centroides = math.floor(num_clientes / 5)
        if num_centroides == 0: num_centroides = 1
        
        for _ in range(num_centroides):
            centroide_x = random.randint(0, ancho_zona)
            centroide_y = random.randint(0, largo_zona)
            centroides.append((centroide_x, centroide_y))

    for i in range(num_clientes):
        cliente = ET.SubElement(clientes, "Cliente")
        cliente.set("id", str(i + 1))
        
        pos_x, pos_y = 0, 0

        if distribucion_geo == "GU":
            pos_x = random.randint(0, ancho_zona)
            pos_y = random.randint(0, largo_zona)
        
        elif distribucion_geo == "GC":
            centroide_asignado = centroides[i % num_centroides]
            offset_x = random.randint(-100, 100)
            offset_y = random.randint(-100, 100)
            pos_x = max(0, min(ancho_zona, centroide_asignado[0] + offset_x))
            pos_y = max(0, min(largo_zona, centroide_asignado[1] + offset_y))

        ET.SubElement(cliente, "PosicionX").text = str(pos_x)
        ET.SubElement(cliente, "PosicionY").text = str(pos_y)

        # --- BLOQUE DE CÓDIGO FALTANTE QUE HA SIDO REINSERTADO ---
        ET.SubElement(cliente, "CapacidadAlmacenamiento").text = str(int(random.uniform(50, 150) * num_productos))
        
        inventarios = ET.SubElement(cliente, "Inventarios")
        costos = ET.SubElement(cliente, "Costos")
        demandas = ET.SubElement(cliente, "Demandas")
        penalti = np.random.uniform(1, 120)
        almacenamiento = np.random.uniform(1, 3)
        
        for j in range(num_productos):
            inv = ET.SubElement(inventarios, f"Producto{j+1}")
            inv.text = str(0) #inventario inicial
            
            costo_prod = ET.SubElement(costos, f"Producto{j+1}")
            ET.SubElement(costo_prod, "CostoInventario").text = str(round(almacenamiento))
            ET.SubElement(costo_prod, "CostoPenalizacion").text = str(round(penalti))
            
            demanda_prod = ET.SubElement(demandas, f"Producto{j+1}")
            ET.SubElement(demanda_prod, "Media").text = str(round(np.random.uniform(1, 2)))
            ET.SubElement(demanda_prod, "DesvEst").text = str(round(random.uniform(1, 2), 2))
        # --- FIN DEL BLOQUE REINSERTADO ---

    vehiculos = ET.SubElement(root, "Vehiculos")
    for i in range(num_vehiculos):
        vehiculo = ET.SubElement(vehiculos, "Vehiculo")
        vehiculo.set("id", str(i+1))
        ET.SubElement(vehiculo, "Capacidad").text = str(300 * num_productos)
        ET.SubElement(vehiculo, "VelocidadMedia").text = "50"
        ET.SubElement(vehiculo, "DesvEstVelocidad").text = '5'
    
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    
    os.makedirs(carpeta_instancias, exist_ok=True)
    
    ruta_completa = os.path.join(carpeta_instancias, nombre_archivo)
    
    with open(ruta_completa, "w", encoding="utf-8") as f:
        f.write(xmlstr)
    
    return f"Archivo '{ruta_completa}' generado exitosamente."



def generar_lote_instancias(lista_configuraciones, num_repeticiones=5, semilla_base=1):
    """
    Genera un conjunto de instancias a partir de una lista de nombres de configuración,
    creando un número específico de repeticiones para cada una.
    """
    total_instancias = len(lista_configuraciones) * num_repeticiones
    print(f"Iniciando generación de {total_instancias} instancias ({num_repeticiones} por configuración)...")

    # Mapeo de configuraciones (sin cambios)
    clientes_map = {"C1": (5, 12), "C2": (13, 20), "C3": (21, 28)}
    productos_map = {"P1": 2, "P2": 4, "P3": 6}

    # Bucle principal que itera sobre cada TIPO de configuración (ej: "C1M2P1GU")
    for i, config_nombre in enumerate(lista_configuraciones):
        print(f"\n--- Procesando configuración: {config_nombre} ---")

        # --- INICIO DEL CAMBIO ---
        # Bucle anidado para generar las 10 repeticiones de la misma configuración
        for j in range(num_repeticiones):
            
            # 1. Se calcula una SEMILLA ÚNICA para cada instancia generada.
            #    Esto garantiza que las 10 repeticiones sean diferentes entre sí.
            semilla_actual = semilla_base + (i * num_repeticiones) + j
            
            # Parseo del nombre (sin cambios)
            codigo_c, codigo_m, codigo_p, codigo_g = config_nombre[0:2], config_nombre[2:4], config_nombre[4:6], config_nombre[6:8]
            
            # --- Clientes (N) ---
            # Es importante volver a calcular N para cada repetición para tener variedad
            rango_clientes = clientes_map[codigo_c]
            N = random.randint(rango_clientes[0], rango_clientes[1])

            # --- Vehículos (M) ---
            if codigo_m == "M1":
                M = math.floor(N / 4)
            elif codigo_m == "M2":
                M = max(math.floor(N / 3), math.floor(N / 4) + 1)
            else: # M3
                M = max(math.floor(N / 2), math.floor(N / 3) + 1)
            if M == 0: M = 1
                
            # --- Productos (P) ---
            P = productos_map[codigo_p]

            # Se crea un NOMBRE DE ARCHIVO ÚNICO que incluye el número de la repetición.
            nombre_archivo_final = f"instancia_{config_nombre}Almacenamiento_rep{j+1}.xml"
            print(f"  Generando: {nombre_archivo_final} con semilla {semilla_actual}")

            # Se llama a la función generadora original con los parámetros de esta repetición.
            generar_instancia_irp(
                ancho_zona=1000,
                largo_zona=1000,
                horizonte_tiempo=300,
                num_productos=P,
                num_clientes=N,
                num_vehiculos=M,
                semilla=semilla_actual,
                carpeta_instancias="INSTANCIAS_ALMACENAMIENTO",
                distribucion_geo=codigo_g,
                nombre_archivo=nombre_archivo_final
            )
    
    print("\n¡Proceso completado! Todas las instancias han sido generadas.")

if __name__ == "__main__":
    # Define aquí todas las configuraciones que quieres crear
    configs_a_generar = [
        "C1M1P1GU"
        ]
    
    # Llama a la función controladora para iniciar el proceso
    generar_lote_instancias(configs_a_generar, num_repeticiones=10, semilla_base=1)