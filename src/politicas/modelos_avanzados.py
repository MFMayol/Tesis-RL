from .simples import PoliticasSimples
from src.Estado import Estado
import numpy as np
from src.FuncionesAuxiliares import fourier_transform_features_vectorized, fourier_transform_interactions_vectorized
import copy
import time

class Modelos_de_aproximacion(PoliticasSimples):
    '''
    Clase que implementa los modelos de aproximación utilizados en la política MC Onpolicy, heredado de PoliticasSimples
    ''' 
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate):
        super().__init__(instancia, proceso)
        self.episodios = episodios
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.n_episodio = 0
        self.periodos = 0

    def _obtener_features(self, estado, accion):
        """
        Obtiene los features de un estado y acción dados.
        Esta versión está optimizada para reducir bucles de Python y usar operaciones
        vectorizadas donde sea posible, mejorando el rendimiento.
        """
        # --- 1. Pre-cálculos y Preparación ---
        clientes = self.instancia.clientes
        productos = self.instancia.productos
        id_clientes = self.instancia.id_clientes
        id_productos = self.instancia.id_productos
        id_vehiculos = self.instancia.id_vehiculos
        
        estado_copia = estado.copiar()
        self.proceso.actualizar_planificacion(estado_copia, accion)
        planificacion_post = estado_copia.planificacion

        # --- 2. Construcción de Grupos de Features ---

        # Grupo 1: Intercepto y Tiempo
        feature_intercepto = np.array([1.0], dtype=np.float32)
        tiempo_norm = (self.instancia.horizonte_tiempo - estado.tiempo) / self.instancia.horizonte_tiempo
        feature_tiempo = np.array([tiempo_norm], dtype=np.float32)

        # Grupo 2: Ratios de Inventario de Clientes (parcialmente vectorizado)
        features_ratios_inv_list = []
        ratio_inventario_clientes_productos = {}
        for idc in id_clientes:
            cliente = clientes[idc]
            ratio_inventario_clientes_productos[idc] = {}
            capacidad_cliente = cliente.capacidad_almacenamiento if cliente.capacidad_almacenamiento > 0 else 1.0
            for idp in id_productos:
                ratio = (estado.inventarios_clientes[idc][idp] * productos[idp].peso) / capacidad_cliente
                features_ratios_inv_list.append(ratio)
                ratio_inventario_clientes_productos[idc][idp] = ratio
        features_ratios_inv = np.array(features_ratios_inv_list, dtype=np.float32)

        # Grupo Extra (Faltante): Inventario promedio de cada producto en la flota (Feature 3 del paper)
        features_flota_list = []
        num_vehiculos = len(id_vehiculos)
        for idp in id_productos:
            inv_total_producto = sum(estado.inventarios_vehiculos[idv].get(idp, 0) for idv in id_vehiculos)
            features_flota_list.append(inv_total_producto / num_vehiculos if num_vehiculos > 0 else 0.0)
        features_flota_arr = np.array(features_flota_list, dtype=np.float32)

        # Grupo 3: Features de Depot (bucle necesario por lógica compleja)
        features_depot = []
        ratios_utilizacion = self._calcular_ratios_utilizacion_vehiculos(estado)
        for idv in id_vehiculos:
            for idp in id_productos:
                if not planificacion_post[idv]:
                    features_depot.append(0.0)
                elif ratios_utilizacion[idv].get(idp, 0.0) < 0.1 and next(iter(planificacion_post[idv])) == 0:
                    features_depot.append(1.0)
                else:
                    features_depot.append(0.0)
        features_depot_arr = np.array(features_depot, dtype=np.float32)

        # Grupo 4: Features de Entregas y Distancias (bucle necesario)
        features_entregas = []
        pos_clientes = {
            idc: (clientes[idc].posicion_x, clientes[idc].posicion_y)
            for idc in id_clientes
        }
        for idc in id_clientes:
            xc, yc = pos_clientes[idc]
            capacidad = clientes[idc].capacidad_almacenamiento
            if capacidad == 0: capacidad = 1.0 # Evitar división por cero

            for idp in id_productos:
                ratio_actual = ratio_inventario_clientes_productos[idc][idp]
                if ratio_actual >= 0.2:
                    features_entregas.extend([0.0, 0.0, 0.0, 0.0, 0.0]) # Corregido a 5 ceros
                    continue

                ratio_peso, idv = self._calcular_peso_por_cliente_producto(planificacion_post, idc, idp)
                ratio_entrega = ratio_peso / capacidad
                # Corregido: Multiplicación explícita descrita en el Feature 5 del paper
                ratio_base = ratio_entrega * ratio_actual 

                if idv is None or idv not in estado.posiciones_vehiculos:
                    features_entregas.extend([ratio_base, 0.0, 0.0, 0.0, 0.0])
                    continue

                xv, yv = estado.posiciones_vehiculos[idv]['x'], estado.posiciones_vehiculos[idv]['y']
                distancia = np.hypot(xc - xv, yc - yv)

                f_500 = ratio_base if distancia <= 500 else 0.0
                f_1000 = ratio_base if 500 < distancia <= 1000 else 0.0
                f_1500 = ratio_base if 1000 < distancia <= 1500 else 0.0
                f_mas_1500 = ratio_base if distancia > 1500 else 0.0

                features_entregas.extend([ratio_base, f_500, f_1000, f_1500, f_mas_1500])
        features_entregas_arr = np.array(features_entregas, dtype=np.float32)

        # --- 3. Concatenación Final ---
        # Unimos todos los grupos de features en un único array de NumPy
        return np.concatenate([
            feature_intercepto,
            feature_tiempo,
            features_flota_arr,
            features_ratios_inv,
            features_depot_arr,
            features_entregas_arr
        ])

    def _crear_betas(self):
        '''
        Crea un vector de parámetros beta que coincide exactamente con la estructura de features.
        Este método es heredado por MCNN, QLNN y Modelos_aprox_lineal.
        '''
        betas = [0]  # Beta para el término constante (feature inicial)
        
        # Beta para el feature de tiempo restante (1 beta) +1 
        betas.append(0.0)
        
        # Betas para el inventario promedio de la flota por producto (Feature 3)
        for _ in self.instancia.id_productos:
            betas.append(0.0)
        
        # Betas para inventario por cliente-producto (|N| * |P| betas) + el cuadrado
        for _ in self.instancia.id_clientes:
            for _ in self.instancia.id_productos:
                betas.append(0.0)
        
        # Betas para vehículo-producto (|M| * |P| betas)
        for _ in self.instancia.id_vehiculos:
            for _ in self.instancia.id_productos:
                betas.append(0.0)
        
        # Betas para entrega con distancia (5 betas por cliente-producto: base + 4 umbrales)
        for _ in self.instancia.id_clientes:
            for _ in self.instancia.id_productos:
                betas.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        self.numero_de_features = len(betas) # obtenemos el numero de features
        self.betas = np.array(betas, dtype=np.float32)

    def _calcular_peso_por_cliente_producto(self, accion, id_cliente, id_producto):
        '''
        Descripción:
        Método que calcula el peso de entrega de un cliente en relación a un producto
        Args:
            * accion: Diccionario con la planificación de todos los vehículos
            * id_cliente: ID del cliente
            * id_producto: ID del producto
        Return:
            * peso_entrega: Peso de entrega del cliente en relación al producto
            * id_vehículo: ID del vehículo que tiene la entrega hacia ese cliente
        '''
        total_peso = 0.
        idv = None
        for id_vehículo , vehiculo in accion.items():
            # Verificar si el vehículo tiene acción
            if not vehiculo:
                continue  # Saltar vehículos inactivos
            # Obtener destino y productos (seguro porque vehiculo no está vacío)
            id_destino, productos = next(iter(vehiculo.items()))
            if id_destino == id_cliente:
                idv = id_vehículo
                for idp, cantidad in productos.items():
                    if idp == id_producto:
                        total_peso += cantidad * self.instancia.productos[id_producto].peso
        return total_peso, idv

    def _calcular_ratios_utilizacion_vehiculos(self, estado: Estado):
        """
        Calcula el ratio de utilización (peso actual vs capacidad) para cada vehículo.
        
        Args:
            estado: Objeto Estado
        
        Returns:
            dict: Diccionario con {id_vehiculo: {id_producto : ratio_utilizacion} }
                Ratio entre 0.0 (vacío) y 1.0+ (sobrecargado)
        """
        ratios = {}
        
        for id_vehiculo, inventario in estado.inventarios_vehiculos.items():
            try:
                # 1. Obtener capacidad del vehículo
                vehiculo = self.instancia.vehiculos[id_vehiculo]
                capacidad = vehiculo.capacidad
                ratios[id_vehiculo] = {}
                for id_producto, cantidad in inventario.items():
                    producto = self.instancia.productos[id_producto]
                    ratios[id_vehiculo][id_producto] = round(cantidad * producto.peso / capacidad, 2)
            except KeyError as e:
                # Manejar vehículos o productos no existentes
                print(f"Advertencia: {e} no encontrado. Vehículo {id_vehiculo} omitido")
                ratio = 0.0
        return ratios

class Modelos_aprox_lineal(Modelos_de_aproximacion):
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate):
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate)
        self.betas = None
        self.mejores_betas = np.array([])
        self.optimo_mejor_betas = None # valor float asociado al mejor promedio FO
        self.traslado_mejor_betas = None
        self.insatisfecha_mejor_betas = None
        self.almacenamiento_mejor_betas = None
        self.registro_optimos_mejores_betas = np.array([])
        self.registro_optimos = np.array([])
        self.tiempo_entrenamiento = None
        self.mejores_periodos = None


    def _inicializar_mejores_betas(self):
        '''
        Descripción:
        Método que inicializa los mejores betas encontrados en el entrenamiento con los betas = 0
        '''
        self.mejores_betas = self.betas.copy()
        # ejecutamos el algoritmo MC OnPolicy para encontrar el óptimo promedio utilizando los mejores betas
        recompensas = np.array([])
        costo_traslado = np.array([])
        costo_insat = np.array([])
        costo_almacenamiento = np.array([])
        periodos = np.array([])
        for _ in range(10):
            costo_episodio, tras, insat, alm, n_periodos = self._probar_politica_actual()
            recompensas = np.append(recompensas,costo_episodio)
            costo_traslado = np.append(costo_traslado, tras)
            costo_insat = np.append(costo_insat, insat)
            costo_almacenamiento = np.append(costo_almacenamiento, alm)
            periodos = np.append(periodos,n_periodos)

        promedio_recompensa = np.mean(recompensas)
        promedio_traslado = np.mean(costo_traslado)
        promedio_insatisfecha = np.mean(costo_insat)
        promedio_almacenamiento = np.mean(costo_almacenamiento)
        promedio_periodos = np.mean(periodos)

        self.optimo_mejor_betas = promedio_recompensa
        self.traslado_mejor_betas = promedio_traslado
        self.insatisfecha_mejor_betas = promedio_insatisfecha
        self.almacenamiento_mejor_betas = promedio_almacenamiento
        self.periodos = promedio_periodos

        self.registro_optimos = np.append(self.registro_optimos, promedio_recompensa)
        
        self.registro_optimos_mejores_betas = np.append(
            self.registro_optimos_mejores_betas,
            self.optimo_mejor_betas
        )

    def _probar_politica_actual(self):
        ''' 
        Descripción:
        Método que implementa la política entrenada y hace una simulación tomando la mejor decisión según los betas actuales
        Args: 
            * None
        Return:
                * fo, costo_traslado, insatisfecha, almacenamiento
            '''
        # Comenzamos en el estado inicial
        estado = self.proceso.determinar_estado_inicial()
        costo_traslado = 0
        costo_insatisfecha = 0
        costo_almacenamiento = 0
        fo = 0
        periodos = 0
        while True:
            acciones = self._obtener_acciones(estado)
            accion = self._politica_optima(estado, acciones)
            nuevo_estado, recompensa,traslado, insatisfecha, costo_almacenamiento = self.proceso.transicion(estado,accion)
            costo_traslado += traslado
            costo_insatisfecha += insatisfecha
            costo_almacenamiento += costo_almacenamiento
            fo += recompensa
            periodos += 1

            if self.es_terminal(nuevo_estado):
                self.periodos = periodos
                break
            else:
                estado = nuevo_estado 
        return fo, costo_traslado, costo_insatisfecha, costo_almacenamiento, periodos

    def _actualizar_mejores_betas(self):
        '''
        Actualiza los mejores parámetros beta usando 20 simulaciones Monte Carlo,
        optimizando el rendimiento y uso de memoria.
        '''
        # 1. Colección eficiente de recompensas
        recompensas = []
        traslados = []
        insat = []
        almac = []
        periodos = []
        for _ in range(20):
            costo_episodio, tras, ins, alm, periodo = self._probar_politica_actual() 
            recompensas.append(costo_episodio)
            traslados.append(tras)
            insat.append(ins)
            almac.append(alm)
            periodos.append(periodo)
        
        # 2. Cálculo vectorizado del promedio
        promedio_recompensa = np.mean(recompensas)
        promedio_traslado = np.mean(traslados)
        promedio_insatisfecha = np.mean(insat)
        promedio_almacenamiento = np.mean(almac)
        promedio_periodos = np.mean(periodos)


        # 3. Actualización de registros optimizada
        self.registro_optimos = np.append(self.registro_optimos, promedio_recompensa)
        
        # 4. Actualización condicional de mejores betas
        if promedio_recompensa < self.optimo_mejor_betas:
            self.mejores_periodos = self.periodos
            self.mejores_betas = self.betas.copy()  # Copy de numpy vs deepcopy
            self.optimo_mejor_betas = promedio_recompensa
            self.traslado_mejor_betas = promedio_traslado
            self.insatisfecha_mejor_betas = promedio_insatisfecha
            self.almacenamiento_mejor_betas = promedio_almacenamiento
            self.periodos = promedio_periodos


        # 5. Guardado eficiente del mejor óptimo
        self.registro_optimos_mejores_betas = np.append(
            self.registro_optimos_mejores_betas,
            self.optimo_mejor_betas  # Evita copia redundante
        )

    def _SGD(self,x,y):
            '''
            Descripción:
            *   Método que aplica SGD, recibe un array x y actualiza los pesos originados self.betas
            Parámetros:
            *   x: Array con los features de ese estado acción
            *   y: Recompensa obtenida en ese estado acción
            *   c: costo fijo del estado acción
            '''
            # Predicción y error
            prediccion = np.dot(x, self.betas) # predice el Q valor
            error = prediccion - y  # o (y - prediction) dependiendo de la convención
            # Gradiente (para una muestra)
            gradients = x * error  # Si x es un vector 1D
            # Actualización
            self.betas -= self.learning_rate * gradients

    def _tomar_accion_epsilon_greedy(self, estado):
        ''' 
        Descripción:
        Método que toma una acción utilizando la política MC OnPolicy en un estado dado

        Args:
            * estado: Objeto Estado
        Return:
            * accion: Acción a tomar
        '''
        numero = np.random.random()
        acciones = self._obtener_acciones(estado)
        if numero < self.epsilon:
            accion = np.random.choice(acciones)
        else:
            accion = self._politica_optima(estado, acciones)
        return accion

    def ejecutar_politica_mejores_betas(self):
        '''
        Descripción:
        Método que ejecuta un episodio siguiendo la mejor acción con los mejores betas encontrados en el entrenamiento
        Args:
            * None
        Return:
            * trayectoria: Diccionario con el detalle de lo que hizo durante la simulación
        '''
        # Comenzamos en el estado inicial
        estado = self.proceso.determinar_estado_inicial()
        trayectoria = []
        costo_traslado = 0
        while True:
            acciones = self._obtener_acciones(estado)
            accion = self._politica_optima_mejores_betas(estado, acciones)
            nuevo_estado, recompensa, traslado, *_ = self.proceso.transicion(estado,accion)
            trayectoria.append({'estado': estado, 'accion': accion, 'recompensa': recompensa})
            costo_traslado += traslado
            if self.es_terminal(estado):
                break
            else:
                estado = nuevo_estado 
        return trayectoria, costo_traslado

    def _politica_optima_mejores_betas(self,state, acciones):
            '''
            Descripción:
            *   Método que devuelve la mejor accion para un estado utilizando la función de valor con mejores betas obtenidos
            Parámetros:
            *   state: Estado
            *   acciones: Lista de acciones posibles
            Return:
            *   action: Mejor acción
            '''
            #lista de tuplas
            mejor_Q = np.inf
            mejor_accion = None
            for action in acciones: 
                x = self._obtener_features(state, action) # Lista de features
                # obtenemos el c(st,at)
                c_st_at = self.proceso.determinar_c_st_at(state, action)
                Q = np.dot(x, self.mejores_betas) + c_st_at 
                if Q < mejor_Q:
                    mejor_Q = Q
                    mejor_accion = action
            return mejor_accion 
    
    def _politica_optima(self,state, acciones):
            '''
            Descripción:
            *   Método que devuelve la mejor accion para un estado utilizando la función de valor
            Parámetros:
            *   state: Estado
            *   acciones: Lista de acciones posibles
            Return:
            *   action: Mejor acción
            '''

            qs = []
            for action in acciones:
                features = self._obtener_features(state, action)
                c_st_at = self.proceso.determinar_c_st_at(state, action)
                Q = np.dot(features, self.betas) + c_st_at
                qs.append((Q, action))

            # Elegir la acción con el menor Q
            return min(qs, key=lambda x: x[0])[1]

    def _ejecutar_politica_epsilon_greedy(self):
        '''Ejecuta un episodio con política epsilon-greedy optimizada.'''
        proceso = self.proceso  # Cachear para acceso rápido
        es_terminal = self.es_terminal  # Evitar búsquedas de atributo
        trayectoria = []
        
        estado = proceso.determinar_estado_inicial()
        while True:
            # Paso 1: Tomar acción y obtener features
            accion = self._tomar_accion_epsilon_greedy(estado)
            features = self._obtener_features(estado, accion)
            c_st_at = proceso.determinar_c_st_at(estado, accion)
            
            # Paso 2: Transición de estado (ignoramos variables no usadas)
            nuevo_estado, recompensa, *_ = proceso.transicion(estado, accion)
            
            # Paso 3: Almacenar datos (usamos tupla para menor overhead)
            trayectoria.append( (estado, accion, features, recompensa, c_st_at) )
            
            # Paso 4: Verificar condición de término
            if es_terminal(nuevo_estado):  # ¡Clave! Verificar nuevo_estado
                break
                
            estado = nuevo_estado
            
        return [{
            'estado': s,
            'accion': a,
            'features': f,
            'recompensa': r,
            'c_st_at': c
        } for s, a, f, r, c in trayectoria]

class MonteCarlo(Modelos_aprox_lineal):
    ''' Objeto que implementará la política MC Onpolicy al problema'''

    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate):
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate)


    def entrenar_modelo(self):
        ''' Método que ejecuta el algoritmo MC OnPolicy'''
        self._crear_betas()
        self._inicializar_mejores_betas()
        inicio = time.time()  # Empieza a medir tiempo

        for episodio in range(self.episodios):
            trayectoria = self._ejecutar_politica_epsilon_greedy() # devuelve una lista de diccionarios con el estado, accion, recompensa, features

            # implementamos la política epsilon greedy
            G = 0
            for t in reversed(trayectoria):
                G = G  +  t['recompensa']
                # Actualizamos los pesos
                x = t['features'] 
                c_st_at = t['c_st_at']

                self._SGD(x, G - c_st_at)
            
            if episodio % 100 == 0:
                self._actualizar_mejores_betas()
        
        fin = time.time()  # Termina de medir tiempo
        self.tiempo_entrenamiento = round(fin - inicio)

class Fourier(Modelos_aprox_lineal):
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate, max_i):
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate)
        self.max_i = max_i
        # la interacción usa n1=5 (inventario) y n2=3 (entrega)
        self.max_i_entrega = 3 

    def _obtener_features(self, estado, accion):
        """
        Obtiene los features de un estado y acción dados.
        Aplica Fourier 1D a las variables individuales.
        Aplica Fourier 2D (Interacción) al par Inventario-Entrega.
        """
        clientes = self.instancia.clientes
        productos = self.instancia.productos
        vehiculos = estado.posiciones_vehiculos

        # Copiar el estado y aplicar la acción
        estado_copia = estado.copiar()
        self.proceso.actualizar_planificacion(estado_copia, accion)
        planificacion_post = estado_copia.planificacion

        pos_clientes = {
            idc: (clientes[idc].posicion_x, clientes[idc].posicion_y)
            for idc in self.instancia.id_clientes
        }

        # --- LISTAS PARA ALMACENAR DATOS CRUDOS ---
        features_1d_raw = [] # Para transformación estándar
        interacciones_raw = [] # Para transformación 2D (pares x, y)

        # 1. Feature: tiempo restante normalizado
        tiempo_norm = (self.instancia.horizonte_tiempo - estado.tiempo) / self.instancia.horizonte_tiempo
        features_1d_raw.append(tiempo_norm)

        # Feature 3: Inventario promedio de cada producto en la flota
        num_vehiculos = len(self.instancia.id_vehiculos)
        for idp in productos:
            inv_total = sum(estado.inventarios_vehiculos[idv].get(idp, 0) for idv in self.instancia.id_vehiculos)
            features_1d_raw.append(inv_total / num_vehiculos if num_vehiculos > 0 else 0.0)

        # inventario promedio por cliente-producto
        # Guardamos estos valores en un diccionario para usarlos luego en la interacción
        ratio_inventario_map = {} 
        
        for idc in self.instancia.id_clientes:
            cliente = clientes[idc]
            for idp in productos:
                capacidad_cliente = cliente.capacidad_almacenamiento if cliente.capacidad_almacenamiento > 0 else 1.0
                ratio = (estado.inventarios_clientes[idc][idp] * productos[idp].peso) / capacidad_cliente
                
                # Agregamos a 1D (Efecto principal del inventario)
                features_1d_raw.append(ratio)
                
                # Guardamos para el cruce 2D
                ratio_inventario_map[(idc, idp)] = ratio

        # Features: depot por vehículo-producto
        ratios_utilizacion = self._calcular_ratios_utilizacion_vehiculos(estado)
        for idv in self.instancia.id_vehiculos:
            for idp in productos:
                if not planificacion_post[idv]:
                    features_1d_raw.append(0.0)
                elif ratios_utilizacion[idv].get(idp, 0.0) < 0.1 and next(iter(planificacion_post[idv])) == 0:
                    features_1d_raw.append(1.0)
                else:
                    features_1d_raw.append(0.0)

        # Features: entregas y distancias + INTERACCIÓN 2D
        for idc in self.instancia.id_clientes:
            xc, yc = pos_clientes[idc]
            capacidad = clientes[idc].capacidad_almacenamiento if clientes[idc].capacidad_almacenamiento > 0 else 1.0

            for idp in productos:
                ratio_inventario = ratio_inventario_map[(idc, idp)] # Recuperamos el 'x'
                
                # Lógica original de cálculo de entrega
                if ratio_inventario >= 0.2:
                    # Caso: Inventario alto, no entregamos (ratio 0)
                    ratio_base = 0.0
                    features_1d_raw.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    ratio_peso, idv = self._calcular_peso_por_cliente_producto(planificacion_post, idc, idp)
                    ratio_entrega = ratio_peso / capacidad
                    ratio_base = ratio_entrega * ratio_inventario # Multiplicación del paper

                    if idv is None or idv not in vehiculos:
                        features_1d_raw.extend([ratio_base, 0.0, 0.0, 0.0, 0.0])
                    else:
                        xv, yv = vehiculos[idv]['x'], vehiculos[idv]['y']
                        distancia = np.hypot(xc - xv, yc - yv)

                        f_500 = ratio_base if distancia <= 500 else 0.0
                        f_1000 = ratio_base if 500 < distancia <= 1000 else 0.0
                        f_1500 = ratio_base if 1000 < distancia <= 1500 else 0.0
                        f_mas_1500 = ratio_base if distancia > 1500 else 0.0

                        features_1d_raw.extend([ratio_base, f_500, f_1000, f_1500, f_mas_1500])

                # --- AQUÍ AGREGAMOS LA INTERACCIÓN ---
                # Guardamos el par (Inventario, Entrega) para la expansión 2D
                # Nota: Usamos ratio_base (la entrega total) para la interacción, 
                # tal como sugiere el paper (Entrega vs Inventario).
                interacciones_raw.append((ratio_inventario, ratio_base))

        # --- TRANSFORMACIONES ---
        
        # Transformación 1D (Vectorizada)
        arr_1d = np.array(features_1d_raw, dtype=np.float32)
        trans_1d = fourier_transform_features_vectorized(arr_1d, max_i=self.max_i)

        # Transformación 2D (Interacciones)
        # Iteramos sobre los pares guardados y aplicamos la función 2D corregida
        lista_interacciones_transformadas = []
        for inv_val, del_val in interacciones_raw:
            # Usamos max_i (5) para inventario y max_i_entrega (3) para entrega
            res_2d = fourier_transform_interactions_vectorized(
                inv_val, del_val, max_i1=self.max_i, max_i2=self.max_i_entrega
            )
            lista_interacciones_transformadas.append(res_2d)
            
        trans_2d = np.concatenate(lista_interacciones_transformadas)

        # Concatenación Final: Intercepto + 1D + 2D
        final_features = np.concatenate(([1.0], trans_1d, trans_2d))

        return final_features

    def _crear_betas(self):
        '''
        Calcula dinámicamente el tamaño del vector de betas necesario
        considerando expansiones 1D y 2D.
        '''
        # Contamos cuántos features crudos (raw) tenemos de cada tipo
        
        # --- Conteo 1D ---
        num_1d_raw = 0
        num_1d_raw += 1 # Tiempo
        num_1d_raw += len(self.instancia.id_productos) # Feature 3: Inventario flota
        num_1d_raw += len(self.instancia.id_clientes) * len(self.instancia.id_productos) # Inventarios
        num_1d_raw += len(self.instancia.id_vehiculos) * len(self.instancia.id_productos) # Utilización
        num_1d_raw += len(self.instancia.id_clientes) * len(self.instancia.id_productos) * 5 # Entregas (Base + 4 distancias)
        
        # Expansión 1D: Cada feature genera (2 * max_i) nuevos features
        total_1d_expanded = num_1d_raw * (2 * self.max_i)
        
        # --- Conteo 2D (Interacciones) ---
        # Hay 1 interacción por cada par Cliente-Producto
        num_pares_interaccion = len(self.instancia.id_clientes) * len(self.instancia.id_productos)
        
        # Expansión 2D: Sumatoria doble i=1..n1, j=1..n2. 
        # Cantidad = n1 * n2 * 2 (seno y coseno)
        # n1 = self.max_i (5), n2 = self.max_i_entrega (3)
        params_per_interaction = self.max_i * self.max_i_entrega * 2
        total_2d_expanded = num_pares_interaccion * params_per_interaction
        
        # --- Total ---
        total_betas = 1 + total_1d_expanded + total_2d_expanded # +1 por el Bias (Intercepto)
        
        self.numero_de_features = total_betas
        self.betas = np.zeros(total_betas, dtype=np.float32)

        # return np.array([0] * total_betas, dtype=np.float32)

class MonteCarlo_Fourier(Fourier):
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate, max_i):
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate, max_i)

    def entrenar_modelo(self):
        ''' Método que ejecuta el algoritmo MC OnPolicy'''
        self._crear_betas()
        self._inicializar_mejores_betas()
        inicio = time.time()
        for episodio in range(self.episodios):
            trayectoria = self._ejecutar_politica_epsilon_greedy()
            # sumamos el contador de episodios
            self.n_episodio += 1
            # implementamos la política epsilon greedy
            G = 0
            for t in reversed(trayectoria):
                G = G  +  t['recompensa']
                # Actualizamos los pesos
                x = t['features']
                c_st_at = t['c_st_at']
                self._SGD(x, G - c_st_at)
            
            if episodio % 100 == 0:
                self._actualizar_mejores_betas()
        fin = time.time()  # ⏱️ Termina de medir tiempo
        self.tiempo_entrenamiento = round(fin - inicio)
