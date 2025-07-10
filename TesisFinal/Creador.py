import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import os

def generar_instancia_irp(
        ancho_zona: int,
        largo_zona: int,
        horizonte_tiempo: int,
        num_productos: int,
        num_clientes: int,
        num_vehiculos: int,
        semilla: int,
        carpeta_instancias: str,
        nombre_archivo = "instancia_irp.xml"
    ):
    """
    Función principal para generar una instancia del problema IRP multiproducto
    
    Parámetros:
    - ancho_zona: Ancho del área de operación
    - largo_zona: Largo del área de operación
    - horizonte_tiempo: Número de períodos de tiempo
    - num_productos: Cantidad de productos diferentes
    - num_clientes: Número de clientes
    - num_vehiculos: Número de vehículos disponibles
    - nombre_archivo: Nombre del archivo XML a generar
    """
    random.seed(semilla) #fijamos la semilla
    np.random.seed(semilla) #fijamos la semilla
    
    # Crear el elemento raíz del XML
    root = ET.Element("InstanciaIRP")
    
    # Añadir información general de la instancia
    info_general = ET.SubElement(root, "InformacionGeneral")
    ET.SubElement(info_general, "AnchoZona").text = str(ancho_zona)
    ET.SubElement(info_general, "LargoZona").text = str(largo_zona)
    ET.SubElement(info_general, "Depot_x").text = str(ancho_zona/2)
    ET.SubElement(info_general, "Depot_y").text = str(largo_zona/2)
    ET.SubElement(info_general, "HorizonteTiempo").text = str(horizonte_tiempo)
    
    # Generar información de productos
    productos = ET.SubElement(root, "Productos")
    for i in range(num_productos):
        producto = ET.SubElement(productos, "Producto")
        producto.set("id", str(i+1))
        # Generar peso aleatorio entre 0.5 y 2 unidades
        ET.SubElement(producto, "Peso").text = str(random.choice([1])) # Lo fijé entre 0.5 o 1 por mientras. str(round(random.uniform(0.5, 1), 2))
    
    # Generar información de clientes
    clientes = ET.SubElement(root, "Clientes")
    for i in range(num_clientes):
        cliente = ET.SubElement(clientes, "Cliente")
        cliente.set("id", str(i+1))
        
        # Generar ubicación aleatoria dentro de la zona
        ET.SubElement(cliente, "PosicionX").text = str(random.randint(0, ancho_zona))
        ET.SubElement(cliente, "PosicionY").text = str(random.randint(0, largo_zona))
        
        # Capacidad total de almacenamiento (en peso)
        ET.SubElement(cliente, "CapacidadAlmacenamiento").text = str(int( random.uniform(50,150) *num_productos))    #str(int(random.randint(100, 100)))
        
        # Información por producto
        inventarios = ET.SubElement(cliente, "Inventarios")
        costos = ET.SubElement(cliente, "Costos")
        demandas = ET.SubElement(cliente, "Demandas")
        penalti = np.random.uniform(1, 120)
        
        for j in range(num_productos):

            # Inventario inicial
            inv = ET.SubElement(inventarios, f"Producto{j+1}")
            inv.text = str(10) # str(int(random.randint(50, 50))) # 
            
            # Costos
            costo_prod = ET.SubElement(costos, f"Producto{j+1}")
            ET.SubElement(costo_prod, "CostoInventario").text = '0'  
            ET.SubElement(costo_prod, "CostoPenalizacion").text = str(round(penalti)) #
            
            # Demandas por período
            demanda_prod = ET.SubElement(demandas, f"Producto{j+1}")
            ET.SubElement(demanda_prod, "Media").text = str(round(np.random.uniform(1,2) )) #
            ET.SubElement(demanda_prod, "DesvEst").text = str(round(random.uniform(1,2), 2))   # instancias features (entre 1 y 3 con desviación de 2)

    # Generar información de vehículos
    vehiculos = ET.SubElement(root, "Vehiculos")
    for i in range(num_vehiculos):
        vehiculo = ET.SubElement(vehiculos, "Vehiculo")
        vehiculo.set("id", str(i+1))
        
        # Capacidad del vehículo (en peso)
        ET.SubElement(vehiculo, "Capacidad").text = str(300*num_productos)  #
        
        # Velocidad del vehículo
        ET.SubElement(vehiculo, "VelocidadMedia").text = "50" 
        ET.SubElement(vehiculo, "DesvEstVelocidad").text = '5' 
    
    # Convertir el XML a string con formato agradable
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    
    # Crear la carpeta INSTANCIAS si no existe
    os.makedirs(carpeta_instancias, exist_ok=True)
    
    # Construir la ruta completa para guardar el archivo
    ruta_completa = os.path.join(carpeta_instancias, nombre_archivo)
    
    # Guardar el archivo en la carpeta INSTANCIAS
    with open(ruta_completa, "w", encoding="utf-8") as f:
        f.write(xmlstr)
    
    return f"Archivo '{ruta_completa}' generado exitosamente."