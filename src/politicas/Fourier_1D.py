from .modelos_avanzados import Fourier
import copy
import numpy as np
from src.FuncionesAuxiliares import fourier_transform_features_vectorized
import time

class Fourier1D(Fourier):
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate, max_i):
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate, max_i)

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

        # 2. Features: inventario promedio por cliente-producto
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

        # 3. Features: depot por vehículo-producto
        ratios_utilizacion = self._calcular_ratios_utilizacion_vehiculos(estado)
        for idv in self.instancia.id_vehiculos:
            for idp in productos:
                if not planificacion_post[idv]:
                    features_1d_raw.append(0.0)
                elif ratios_utilizacion[idv].get(idp, 0.0) < 0.1 and next(iter(planificacion_post[idv])) == 0:
                    features_1d_raw.append(1.0)
                else:
                    features_1d_raw.append(0.0)

        # 4. Features: entregas y distancias + INTERACCIÓN 2D
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
                    ratio_base = ratio_entrega * ratio_inventario

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

        # --- TRANSFORMACIONES ---        
        # Transformación 1D (Vectorizada)
        arr_1d = np.array(features_1d_raw, dtype=np.float32)
        trans_1d = fourier_transform_features_vectorized(arr_1d, max_i=self.max_i)


        # Concatenación Final: Intercepto + 1D + 2D
        final_features = np.concatenate(([1.0], trans_1d))

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
        
        # --- Total ---
        total_betas = 1 + total_1d_expanded
        
        self.numero_de_features = total_betas
        self.betas = np.zeros(total_betas, dtype=np.float32)

        # return np.array([0] * total_betas, dtype=np.float32)

class MonteCarlo_Fourier1D(Fourier1D):
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
