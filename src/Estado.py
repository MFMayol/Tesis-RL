from src.Instancia import Instancia
import copy

class Estado:
    '''
    Objeto que representa un estado del problema.

    Atributos:
    - inventarios_clientes: diccionario con llave id cliente y valor diccionario con llave id producto y valor cantidad Ej: {id_cliente: {id_producto: cantidad}}
    - inventarios_vehiculos: diccionario con llave id vehiculo y valor igual a diccionario con llave id producto y valor cantidad {id_vehiculo: {id_producto : cantidad}}
    - planificacion: diccionario que tiene el diccionario llave igual a los id de los vehiculos y valor otro diccionario que tiene llave cliente y valor diccionario con planificación a ese cliente {id_vehiculo: {id_destino: {id_producto :cantidad} } }}
    - tiempo: valor que representa el tiempo de la simulación
    - posiciones_vehiculos: diccionario con llave id vehiculo y valor lista de posiciones
    '''

    def __init__(self):
        self.inventarios_clientes = {} # diccionario con llave id cliente y valor diccionario con llave id producto y valor cantidad Ej: {1 (cliente): {1 (producto):(Cantidad) 10, 2:5 }, ...... }
        self.inventarios_vehiculos = {} # diccionario con llave id vehiculo y valor igual a diccionario con llave id producto y valor cantidad
        self.planificacion = {} # diccionario que tiene el diccionario llave igual a los id de los vehiculos y valor otro diccionario que tiene llave cliente y valor diccionario con planificación a ese cliente {1:{1:{id_producto1:10, id_producto2:5}, 2:{4:{1:20 ,2:5}}}}
        self.tiempo = 0.0 # valor que representa el tiempo de la simulación
        self.posiciones_vehiculos = {} # diccionario con llave id vehiculo y diccionario con llave x:valor_x y y: valor_y  {"x": coordenada x, "y":coordenada y}

    def __str__(self):
            return f"""
    Estado del sistema:
        Inventarios de clientes: 
        {self.inventarios_clientes}
        Inventarios de vehículos: 
        {self.inventarios_vehiculos}
        Planificación de entregas:
        {self.planificacion}
        Tiempo transcurrido: {self.tiempo:.2f} unidades de tiempo
        Posiciones de vehículos:
        {self.posiciones_vehiculos}
        """

    def copiar(self):
        '''Retorna una copia profunda optimizada del estado actual.'''
        nuevo_estado = Estado()
        nuevo_estado.inventarios_clientes = {k: v.copy() for k, v in self.inventarios_clientes.items()}
        nuevo_estado.inventarios_vehiculos = {k: v.copy() for k, v in self.inventarios_vehiculos.items()}
        nuevo_estado.planificacion = {k: {k2: v2.copy() for k2, v2 in self.planificacion.items()} for k, v in self.planificacion.items()}
        nuevo_estado.posiciones_vehiculos = {k: v.copy() for k, v in self.posiciones_vehiculos.items()}
        nuevo_estado.tiempo = self.tiempo
        return nuevo_estado