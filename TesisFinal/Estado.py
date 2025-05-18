from Instancia import Instancia

class Estado:
    '''
    Objeto que representa un estado del problema.

    Atributos:
    - inventarios_clientes: diccionario con llave id cliente y valor diccionario con llave id producto y valor cantidad Ej: {1 (cliente): {1 (producto):(Cantidad) 10, 2:5 }, ...... }
    - inventarios_vehiculos: diccionario con llave id vehiculo y valor igual a diccionario con llave id producto y valor cantidad
    - planificacion: diccionario que tiene el diccionario llave igual a los id de los vehiculos y valor otro diccionario que tiene llave cliente y valor diccionario con planificación a ese cliente {1:{1:{id_producto1:10, id_producto2:5}, 2:{4:{1:20 ,2:5}}}}
    - tiempo: valor que representa el tiempo de la simulación
    - posiciones_vehiculos: diccionario con llave id vehiculo y valor lista de posiciones
    '''

    def __init__(self):
        self.inventarios_clientes = {} # diccionario con llave id cliente y valor diccionario con llave id producto y valor cantidad Ej: {1 (cliente): {1 (producto):(Cantidad) 10, 2:5 }, ...... }
        self.inventarios_vehiculos = {} # diccionario con llave id vehiculo y valor igual a diccionario con llave id producto y valor cantidad
        self.planificacion = {} # diccionario que tiene el diccionario llave igual a los id de los vehiculos y valor otro diccionario que tiene llave cliente y valor diccionario con planificación a ese cliente {1:{1:{id_producto1:10, id_producto2:5}, 2:{4:{1:20 ,2:5}}}}
        self.tiempo = float # valor que representa el tiempo de la simulación
        self.posiciones_vehiculos = {} # diccionario con llave id vehiculo y diccionario con llave x:valor_x y y: valor_y
        self.inventario_utilizado_clientes = {}
        
        
    def __copy__(self):
        """Método que crea una copia eficiente del estado actual"""
        nuevo_estado = Estado()
        
        # Copiar tiempo (valor simple)
        nuevo_estado.tiempo = self.tiempo
        
        # Copiar inventarios_clientes (diccionario anidado)
        for idc, productos in self.inventarios_clientes.items():
            nuevo_estado.inventarios_clientes[idc] = productos.copy()
        
        # Copiar inventarios_vehiculos (diccionario anidado)
        for idv, productos in self.inventarios_vehiculos.items():
            nuevo_estado.inventarios_vehiculos[idv] = productos.copy()
        
        # Copiar planificacion (diccionario doblemente anidado)
        for idv, clientes in self.planificacion.items():
            nuevo_estado.planificacion[idv] = {}
            for idc, productos in clientes.items():
                nuevo_estado.planificacion[idv][idc] = productos.copy()
        
        # Copiar posiciones_vehiculos (diccionario con subdicionarios)
        for idv, posicion in self.posiciones_vehiculos.items():
            nuevo_estado.posiciones_vehiculos[idv] = posicion.copy()
        
        # Copiar inventario_utilizado_clientes (diccionario)
        if hasattr(self, 'inventario_utilizado_clientes'):
            nuevo_estado.inventario_utilizado_clientes = self.inventario_utilizado_clientes.copy()
        
        return nuevo_estado


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