from dataclasses import dataclass, field
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from pathlib import Path
from FuncionesAuxiliares import distancia_euclidiana
import numpy as np 

@dataclass
class Producto:
    """
    Representa un producto en el problema IRP.
    """
    id: int
    peso: float

    def __post_init__(self):
        # Validar que el peso del producto sea positivo
        if self.peso <= 0:
            raise ValueError(f"El peso del producto {self.id} debe ser positivo")

@dataclass
class Vehiculo:
    """
    Representa un vehículo en el problema IRP.
    """
    id: int
    capacidad: float
    velocidad_media: float
    desv_est_velocidad: float
    inventario: Dict[int, float] = field(default_factory=dict)

@dataclass
class Cliente:
    """
    Representa un cliente en el problema IRP.
    """
    id: int
    posicion_x: float
    posicion_y: float
    capacidad_almacenamiento: float
    inventarios_iniciales: Dict[int, float]
    costos_inventario: Dict[int, float]
    costos_penalizacion: Dict[int, float]
    demanda_media: Dict[int, float]
    demanda_desv_est: Dict[int, float]

    def __post_init__(self):
        # Validar que la capacidad de almacenamiento sea positiva
        if self.capacidad_almacenamiento <= 0:
            raise ValueError(f"La capacidad de almacenamiento del cliente {self.id} debe ser positiva")

        # Verificar que todos los diccionarios tengan las mismas claves (productos)
        productos = set(self.inventarios_iniciales.keys())
        if not all(set(d.keys()) == productos for d in [
            self.costos_inventario,
            self.costos_penalizacion,
            self.demanda_media,
            self.demanda_desv_est
        ]):
            raise ValueError(f"Todos los diccionarios del cliente {self.id} deben tener los mismos productos")



class Instancia:
    """
    Clase para leer y gestionar una instancia del problema IRP desde un archivo XML.

    Atributos:
        ruta_archivo (Path): Ruta al archivo XML que contiene la información.
        ancho_zona (float): Ancho de la zona de distribución.
        largo_zona (float): Largo de la zona de distribución.
        depot_X (float): Coordenada X del depósito.
        depot_Y (float): Coordenada Y del depósito.
        horizonte_tiempo (int): Horizonte de tiempo del problema.
        productos (Dict[int, Producto]): Diccionario de productos.
        clientes (Dict[int, Cliente]): Diccionario de clientes.
        vehiculos (Dict[int, Vehiculo]): Diccionario de vehículos.
        distancia_entre_nodos (Dict): Diccionadio con la distancia eucleidiana entre nodos
    
    Métodos:
            __init__(self, ruta_archivo: str): Inicializa la instancia leyendo los datos desde un archivo XML.
            leer_datos_desde_xml(self
            _cargar_desde_xml(self): Carga los datos desde un elemento XML.
    """
    def __init__(self, ruta_archivo: str, umbral_inventario_clientes, umbral_inventario_vehiculos):
        """
        Inicializa la instancia leyendo los datos desde un archivo XML.

        Parámetros:
            ruta_archivo (str): Ruta al archivo XML que contiene la información.
        """
        self.ruta_archivo = Path(ruta_archivo)
        if not self.ruta_archivo.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

        # Inicializar atributos
        self.ancho_zona: float = 0
        self.largo_zona: float = 0
        self.depot_X: float = 0
        self.depot_Y: float = 0
        self.horizonte_tiempo: int = 0
        self.productos: Dict[int, Producto] = {}
        self.clientes: Dict[int, Cliente] = {}
        self.vehiculos: Dict[int, Vehiculo] = {}
        self.demandas_medias = None
        self.distancias_entre_nodos = {}
        self.id_clientes = []
        self.id_vehiculos = []
        self.id_productos = []
        self.umbral_inventario_clientes = umbral_inventario_clientes
        self.umbral_inventario_vehiculos = umbral_inventario_vehiculos
        # Cargar datos desde el archivo XML
        self._cargar_desde_xml()
        self.inicializar_inventarios()
        # self.cargar_distancias()
        self.cargar_ids()

    def cargar_ids(self):
        '''Método que almacena los ids de la instancia
        Parámetros:
            - None
        Retorna:
            - None
        '''
        self.id_clientes = list(self.clientes.keys())
        self.id_vehiculos = list(self.vehiculos.keys())
        self.id_productos = list(self.productos.keys())
    
    def cargar_distancias(self):
        '''
        Método que lee la instancia y devuevle un diccionario con la distancia entre los nodos del problema
        Parámetros:
            - None
        Retorna:
            - distancias_entre_nodos: diccionario con la distancia entre los nodos del problema
        '''

        # Crear un diccionario para almacenar las distancias entre nodos
        # Partiré definiendo desde el depot al resto de los nodos y viseversa
        self.distancias_entre_nodos[0] = {}
        for id, cliente in self.clientes.items():
            #Parto con el depot
            self.distancias_entre_nodos[0][id] = distancia_euclidiana(np.array([self.depot_X, self.depot_Y]), np.array([cliente.posicion_x, cliente.posicion_y]) )
            self.distancias_entre_nodos[id] = {}
            self.distancias_entre_nodos[id][0] = self.distancias_entre_nodos[0][id]
            for id2, cliente2 in self.clientes.items():
                self.distancias_entre_nodos[id][id2]  = distancia_euclidiana(np.array([cliente.posicion_x, cliente.posicion_y]), np.array([cliente2.posicion_x, cliente2.posicion_y]) )

    def _cargar_desde_xml(self) -> None:
        """
        Carga la información de la instancia desde el archivo XML.
        """
        try:
            tree = ET.parse(self.ruta_archivo)
            root = tree.getroot()

            # Cargar información general
            info_general = root.find('InformacionGeneral')
            if info_general is None:
                raise ValueError("No se encontró la sección InformacionGeneral")

            self.ancho_zona = float(info_general.find('AnchoZona').text)
            self.largo_zona = float(info_general.find('LargoZona').text)
            self.depot_X = float(info_general.find('Depot_x').text)
            self.depot_Y = float(info_general.find('Depot_y').text)
            self.horizonte_tiempo = int(info_general.find('HorizonteTiempo').text)

            if any(dim <= 0 for dim in [self.ancho_zona, self.largo_zona, self.horizonte_tiempo]):
                raise ValueError("Las dimensiones y el horizonte de tiempo deben ser positivos")

            # Cargar productos
            self._cargar_productos(root)

            # Cargar vehículos
            self._cargar_vehiculos(root)

            # Cargar clientes
            self._cargar_clientes(root)

        except ET.ParseError as e:
            raise ET.ParseError(f"Error al parsear el archivo XML: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error en los datos del archivo XML: {e}")

    def _cargar_productos(self, root: ET.Element) -> None:
        """
        Carga la información de los productos desde el XML.
        """
        productos_elem = root.find('Productos')
        if productos_elem is None:
            raise ValueError("No se encontró la sección Productos")

        for prod_elem in productos_elem.findall('Producto'):
            id_prod = int(prod_elem.get('id'))
            self.productos[id_prod] = Producto(
                id=id_prod,
                peso=float(prod_elem.find('Peso').text)
            )

    def _cargar_vehiculos(self, root: ET.Element) -> None:
        """
        Carga la información de los vehículos desde el XML.
        """
        vehiculos_elem = root.find('Vehiculos')
        if vehiculos_elem is None:
            raise ValueError("No se encontró la sección Vehiculos")

        for veh_elem in vehiculos_elem.findall('Vehiculo'):
            id_veh = int(veh_elem.get('id'))
            self.vehiculos[id_veh] = Vehiculo(
                id=id_veh,
                capacidad=float(veh_elem.find('Capacidad').text),
                velocidad_media=float(veh_elem.find('VelocidadMedia').text),
                desv_est_velocidad=float(veh_elem.find('DesvEstVelocidad').text)
            )

    def _cargar_clientes(self, root: ET.Element) -> None:
        """
        Carga la información de los clientes desde el XML.
        """
        clientes_elem = root.find('Clientes')
        if clientes_elem is None:
            raise ValueError("No se encontró la sección Clientes")

        for cli_elem in clientes_elem.findall('Cliente'):
            id_cli = int(cli_elem.get('id'))

            inventarios = self._cargar_datos_producto(cli_elem, 'Inventarios')
            costos_inv = self._cargar_datos_producto(cli_elem, 'Costos', 'CostoInventario')
            costos_pen = self._cargar_datos_producto(cli_elem, 'Costos', 'CostoPenalizacion')
            demanda_media = self._cargar_datos_producto(cli_elem, 'Demandas', 'Media')
            demanda_desv = self._cargar_datos_producto(cli_elem, 'Demandas', 'DesvEst')

            self.clientes[id_cli] = Cliente(
                id=id_cli,
                posicion_x=float(cli_elem.find('PosicionX').text),
                posicion_y=float(cli_elem.find('PosicionY').text),
                capacidad_almacenamiento=float(cli_elem.find('CapacidadAlmacenamiento').text),
                inventarios_iniciales=inventarios,
                costos_inventario=costos_inv,
                costos_penalizacion=costos_pen,
                demanda_media=demanda_media,
                demanda_desv_est=demanda_desv
            )
        
    def _cargar_datos_producto(self, elemento: ET.Element, seccion: str, subseccion: Optional[str] = None) -> Dict[int, float]:
        """
        Carga datos relacionados con productos desde una sección específica del XML.

        Parámetros:
            - elemento (ET.Element): Elemento XML del cliente.
            - seccion (str): Nombre de la sección principal (e.g., 'Inventarios').
            - subseccion (str, opcional): Nombre de la subsección (e.g., 'CostoInventario').

        Retorna:
            Dict[int, float]: Diccionario con los datos cargados por producto.
        """
        datos = {}
        seccion_elem = elemento.find(seccion)
        if seccion_elem is None:
            raise ValueError(f"No se encontró la sección {seccion}")

        for prod_id in self.productos.keys():
            if subseccion:
                valor_elem = seccion_elem.find(f'Producto{prod_id}/{subseccion}')
            else:
                valor_elem = seccion_elem.find(f'Producto{prod_id}')

            if valor_elem is None:
                raise ValueError(f"Faltan datos para el producto {prod_id}")

            datos[prod_id] = float(valor_elem.text)

        return datos

    def inicializar_inventarios(self) -> None:
        """
        Inicializa los inventarios de los vehiculos con los valores iniciales según la polítia que considera la demanda promedio de los clientes.
        """
        
        # Primero se obtienen la demandas medias de cada producto
        self.demandas_medias = {}

        for id in self.productos.keys():
            demandas = []
            for cliente in self.clientes.values():
                demandas.append(cliente.demanda_media[id]) 
            self.demandas_medias[id] = sum(demandas) / len(demandas)

        # Ahora implementamos la política de inventario inicial considerndo la demanda promedio de cada producto y las capacidades de los vehiculos según lo conversado con el profesor
        for idp,producto in self.productos.items():
            for idv,vehiculo in self.vehiculos.items():
                if idp == 1:
                    numerador = vehiculo.capacidad
                    denominador = sum(self.demandas_medias[j]/self.demandas_medias[idp] * self.productos[j].peso for j in self.productos.keys())
                    vehiculo.inventario[idp] = int(numerador / denominador)
                else: 
                    vehiculo.inventario[idp] = int( (vehiculo.inventario[1] / self.demandas_medias[1]) * self.demandas_medias[idp])