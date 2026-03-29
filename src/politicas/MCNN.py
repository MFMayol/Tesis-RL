"""
Deep Monte Carlo Neural Network (MCNN) Implementation
=====================================================

Este módulo implementa un algoritmo de Monte Carlo con aproximación de función mediante redes neuronales
para resolver problemas de optimización secuencial. El algoritmo aprende una función de valor que estima
el costo esperado desde cualquier estado dado, utilizando episodios completos para el entrenamiento.

Características principales:
- Aproximación de función valor con redes neuronales profundas
- Política epsilon-greedy para exploración vs explotación  
- Entrenamiento basado en retornos de Monte Carlo (episodios completos)
- Seguimiento automático del mejor modelo encontrado
- Evaluación periódica del rendimiento

Autores: [Tu nombre]
Fecha: [Fecha actual]
Versión: 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from collections import deque
import random
from .modelos_avanzados import Modelos_de_aproximacion
from src.Estado import Estado
from src.Proceso import Proceso


class QNetwork(nn.Module):
    """
    Red Neuronal para aproximación de la función de valor Q.
    
    Esta red neuronal feedforward estima el valor Q(s,a) - el costo esperado
    de tomar una acción específica desde un estado dado y seguir la política óptima.
    
    Arquitectura:
    - Capa de entrada: dimensión variable según las características del problema
    - 2 capas ocultas: 64 y 32 neuronas respectivamente  
    - Funciones de activación: ReLU
    - Capa de salida: 1 neurona (valor Q estimado)
    
    Args:
        input_dim (int): Dimensión del vector de características de entrada
    """
    
    def __init__(self, input_dim, hidden_layers):
        """
        Inicializa la arquitectura de la red neuronal.
        
        Args:
            input_dim (int): Número de características de entrada (features)
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1)) 
        self.model = nn.Sequential(*layers)

    
    def forward(self, x):
        """
        Propagación hacia adelante de la red.
        
        Args:
            x (torch.Tensor): Tensor de características de entrada [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Valores Q estimados [batch_size, 1]
        """
        return self.model(x)


class MCNN(Modelos_de_aproximacion):
    """
    Deep Monte Carlo Neural Network para optimización secuencial.
    
    Esta clase implementa un algoritmo de Monte Carlo que utiliza una red neuronal
    para aproximar la función de valor Q(s,a). El algoritmo:
    
    1. Ejecuta episodios completos usando una política epsilon-greedy
    2. Calcula retornos G_t para cada paso del episodio
    3. Entrena la red neuronal para predecir estos retornos
    4. Mejora iterativamente la política basada en los valores Q aprendidos
    
    El método es especialmente útil para problemas donde:
    - Los episodios tienen duración finita
    - Se requiere optimización de costo total
    - El espacio de estados es grande (requiere aproximación de función)
    
    Atributos:
        instancia: Instancia del problema a resolver
        proceso: Definición de la dinámica del problema (transiciones, recompensas)
        episodios (int): Número total de episodios para entrenar
        epsilon (float): Probabilidad de exploración en política epsilon-greedy
        learning_rate (float): Tasa de aprendizaje para el optimizador
        net (QNetwork): Red neuronal para aproximar función Q
        optimizer: Optimizador Adam para entrenar la red
        mejor_fo (float): Mejor función objetivo encontrada hasta el momento
        mejor_red (QNetwork): Copia de la mejor red entrenada
    """
    
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate, estructura):
        """
        Inicializa el algoritmo Deep Monte Carlo.
        
        Args:
            instancia: Instancia del problema que define clientes, productos, vehículos, etc.
            proceso: Objeto que define las transiciones y recompensas del MDP
            episodios (int): Número de episodios para el entrenamiento
            epsilon (float): Probabilidad de exploración (0.0 = solo explotación, 1.0 = solo exploración)
            learning_rate (float): Tasa de aprendizaje para la red neuronal (típicamente 0.001-0.01)
        """
        # Llamada al constructor de la clase padre
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate)
        
        # Crear vector de parámetros beta (representación de características)
        self._crear_betas()
        self.input_dim = len(self.betas)
        
        # Inicializar red neuronal y optimizador
        self.net = QNetwork(self.input_dim, hidden_layers=estructura)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        # Función de pérdida: Error cuadrático medio para regresión
        self.loss_fn = nn.MSELoss()
        # Variables para seguimiento del rendimiento
        self.mejor_fo = None              # Mejor función objetivo encontrada
        self.mejor_red = None             # Copia de la mejor red neuronal
        self.registro_fo = []             # Historial de función objetivo por evaluación
        self.registro_mejores_fo = []     # Historial de mejores FO encontradas
        # Establecer baseline inicial
        self.mejor_red_inicial()

    def tomar_accion_epsilon_greedy(self, estado):
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        La política epsilon-greedy balancea exploración y explotación:
        - Con probabilidad epsilon: selecciona acción aleatoria (exploración)
        - Con probabilidad (1-epsilon): selecciona mejor acción según Q (explotación)
        
        Args:
            estado: Estado actual del problema
            
        Returns:
            dict: Acción seleccionada según la política epsilon-greedy
        """
        numero = np.random.random()  # Número aleatorio [0,1)
        acciones = self._obtener_acciones(estado)  # Acciones válidas desde este estado
        
        if numero < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.choice(acciones)
        else:
            # Explotación: mejor acción según política actual
            return self.politica_optima(estado, acciones)

    def ejecutar_episodio(self):
        """
        Selecciona la acción óptima según los valores Q actuales de la red.
        
        Para cada acción válida:
        1. Extrae características del par (estado, acción)
        2. Evalúa la red neuronal para obtener valor Q estimado
        3. Selecciona la acción con menor valor Q (problema de minimización de costo)
        
        Args:
            state: Estado actual
            acciones: Lista de acciones válidas desde el estado
            
        Returns:
            dict: Acción con el menor valor Q estimado
        """        
        proceso = self.proceso 
        es_terminal = self.es_terminal
        estado = proceso.determinar_estado_inicial() # objeto estado con las características relevantes del problema
        costo = 0
        c_tras = 0
        c_ins = 0
        c_alm = 0
        while True:
            acciones = self._obtener_acciones(estado)
            accion = self.politica_optima(estado,acciones) # diccionario con la acción a elegir
            nuevo_estado, recompensa, tras, ins,alm = proceso.transicion(estado, accion) # objeto estado al ejecutar la acción a en el estado e
            done = es_terminal(nuevo_estado) # booleano si terminó o no
            costo += recompensa
            c_tras += tras
            c_ins += ins
            c_alm += alm
            if done:
                break
            estado = nuevo_estado
        return costo, c_tras,c_ins,c_alm

    def politica_optima(self, state, acciones):
        """
        Selecciona la acción óptima según los valores Q actuales de la red.
        
        Para cada acción válida:
        1. Extrae características del par (estado, acción)
        2. Evalúa la red neuronal para obtener valor Q estimado
        3. Selecciona la acción con menor valor Q (problema de minimización de costo)
        
        Args:
            state: Estado actual
            acciones: Lista de acciones válidas desde el estado
            
        Returns:
            dict: Acción con el menor valor Q estimado
        """
        self.net.eval()  # Modo evaluación (sin gradientes)
        
        # Optimización: Evaluar todas las acciones simultáneamente en un solo batch
        with torch.no_grad():  # Desactivar cálculo de gradientes para eficiencia
            # Generamos las features para todas las acciones
            features_list = [self._obtener_features(state, action) for action in acciones]
            x_batch = torch.tensor(np.array(features_list), dtype=torch.float32)
            
            # Un solo forward pass para todas las acciones
            q_values = self.net(x_batch).squeeze(-1).numpy()
            
            c_st_at_list = [self.proceso.determinar_c_st_at(state, action) for action in acciones]
            c_st_at_array = np.array(c_st_at_list, dtype=np.float32)
            
            total_q_values = q_values + c_st_at_array
            
        # Encontrar el índice con el menor valor Q
        best_idx = np.argmin(total_q_values)
        return acciones[best_idx]

    def ejecutar_episodio_epsilon_greedy(self, epsilon=None):
        """
        Ejecuta un episodio completo del problema.
        
        Un episodio consiste en una secuencia completa de decisiones desde un estado
        inicial hasta un estado terminal. Durante el episodio:
        
        1. Se parte del estado inicial definido por el proceso
        2. En cada paso se selecciona una acción usando epsilon-greedy
        3. Se observa la transición: nuevo estado y recompensa
        4. Se almacena la información del paso para posterior entrenamiento
        5. Se continúa hasta alcanzar un estado terminal
        
        Args:
            epsilon (float, opcional): Valor específico de epsilon para este episodio.
                                     Si no se proporciona, usa el epsilon de la instancia.
            
        Returns:
            tuple: (trayectoria, costo_total) donde:
                - trayectoria (list): Lista de diccionarios con información de cada paso:
                  {estado, accion, features, recompensa, nuevo_estado, done}
                - costo_total (float): Suma total de recompensas del episodio
                - costo_traslado (float): Suma total de traslado
        """
        # Usar epsilon específico o el de la instancia
        if epsilon is None:
            epsilon = self.epsilon

        # Inicializar variables del episodio
        proceso = self.proceso
        es_terminal = self.es_terminal
        estado = proceso.determinar_estado_inicial()  # Estado inicial
        trayectoria = []  # Almacenar secuencia de pasos
        costo_total = 0.0  # Acumular costo del episodio
        costo_total_traslado= 0.0
        costo_total_ins = 0.0
        costo_total_alm = 0.0


        # Ejecutar episodio hasta llegar a estado terminal
        while True:
            # Seleccionar acción según política epsilon-greedy
            accion = self.tomar_accion_epsilon_greedy(estado)
            
            # Extraer características del par (estado, acción) para entrenamiento
            features = self._obtener_features(estado, accion)
            c_st_at = proceso.determinar_c_st_at(estado, accion)
            
            # Ejecutar transición en el entorno
            nuevo_estado, recompensa, traslado, ins,alm = proceso.transicion(estado, accion)
            
            # Verificar si el nuevo estado es terminal
            done = es_terminal(nuevo_estado)
            
            # Almacenar información del paso
            trayectoria.append({
                'estado': estado,
                'accion': accion,
                'features': features,
                'recompensa': recompensa,
                'c_st_at': c_st_at,
                'nuevo_estado': nuevo_estado,
                'done': done
            })
            
            # Acumular costo y actualizar estado
            costo_total += recompensa
            costo_total_traslado += traslado
            costo_total_ins += ins
            costo_total_alm += alm
            
            # Terminar episodio si se alcanzó estado terminal
            if done:
                break
                
            estado = nuevo_estado

        return trayectoria, costo_total,costo_total_traslado,costo_total_ins,costo_total_alm


    def mejor_red_inicial(self):
        """
        Establece un baseline inicial evaluando la red sin entrenar.
        
        Este método:
        1. Ejecuta 10 episodios deterministas (epsilon=0) con la red inicial
        2. Calcula el costo promedio como baseline
        3. Guarda una copia de la red inicial como "mejor red"
        
        Esto proporciona un punto de referencia para medir mejoras durante el entrenamiento.
        """
        costos = []
        costos_traslado = []
        costos_insatisfecha = []
        costos_almacenamiento = []
        # Ejecutar múltiples episodios para obtener estimación estable
        for _ in range(20):
            costo_episodio, costo_traslado, costo_insatisfecha, costo_almacenamiento = self.ejecutar_episodio()  # Evaluación determinista
            costos.append(costo_episodio)
            costos_traslado.append(costo_traslado)
            costos_insatisfecha.append(costo_insatisfecha)
            costos_almacenamiento.append(costo_almacenamiento)
        self.mejor_fo = np.array(costos).mean()  # Promedio como función objetivo baseline
        self.traslados_mejor_betas = np.array(costos_traslado).mean()
        self.insatisfechas_mejor_betas = np.array(costos_insatisfecha).mean()
        self.almacenamientos_mejor_betas = np.array(costos_almacenamiento).mean()
        self.mejor_red = copy.deepcopy(self.net)  # Copiar red inicial


    def actualizar_mejor_red(self):
        """
        Evalúa la red actual y actualiza la mejor red si se encuentra una mejora.
        
        Este método:
        1. Ejecuta 10 episodios deterministas para evaluar rendimiento actual
        2. Compara el promedio con la mejor función objetivo conocida
        3. Si hay mejora, actualiza la mejor red y función objetivo
        4. Registra el progreso para seguimiento
        
        Se llama periódicamente durante el entrenamiento para mantener la mejor versión.
        """
        costos = []
        costos_traslado = []
        costos_insatisfecha = []
        costos_almacenamiento = []
        # Ejecutar múltiples episodios para obtener estimación estable
        for _ in range(20):
            costo_episodio, costo_traslado, costo_insatisfecha, costo_almacenamiento = self.ejecutar_episodio()  # Evaluación determinista
            costos.append(costo_episodio)
            costos_traslado.append(costo_traslado)
            costos_insatisfecha.append(costo_insatisfecha)
            costos_almacenamiento.append(costo_almacenamiento)
        promedio = np.array(costos).mean()  # Promedio como función objetivo
        prom_tras = np.array(costos_traslado).mean()
        prom_ins = np.array(costos_insatisfecha).mean()
        prom_alm = np.array(costos_almacenamiento).mean()

        # Actualizar mejor red si hay mejora (menor costo)
        if promedio < self.mejor_fo:
            self.mejor_fo = promedio
            self.mejor_red = copy.deepcopy(self.net)
            self.traslados_mejor_betas = prom_tras
            self.insatisfechas_mejor_betas = prom_ins
            self.almacenamientos_mejor_betas = prom_alm
        
        # Registrar progreso para análisis posterior
        self.registro_fo.append(promedio)
        self.registro_mejores_fo.append(self.mejor_fo)

    def entrenar_modelo(self):
        """
        Entrenamiento principal del algoritmo Deep Monte Carlo.
        
        ALGORITMO:
        =========
        Para cada episodio:
            1. Ejecutar episodio completo con política epsilon-greedy
            2. Calcular retornos G_t para cada paso (desde el final hacia atrás)
            3. Entrenar red neuronal: features -> retorno G_t
            4. Evaluar progreso periódicamente
        
        DETALLES TÉCNICOS:
        ==================
        - Monte Carlo: usa episodios completos (no bootstrapping como TD)
        - Retornos G_t: suma de recompensas futuras desde el paso t
        - Entrenamiento inmediato: actualiza red después de cada paso
        - Evaluación cada 100 episodios para monitoreo
        
        FLUJO DEL ENTRENAMIENTO:
        ========================
        1. Recopilar experiencia (ejecutar episodio)
        2. Calcular targets (retornos G)
        3. Entrenar red neuronal (supervised learning)
        4. Repetir hasta convergencia o máximo de episodios
        """
        inicio = time.time()  # Cronometrar entrenamiento
        # LOOP PRINCIPAL DE ENTRENAMIENTO
        for episodio in range(self.episodios):
            # PASO 1: GENERAR EXPERIENCIA
            # Ejecutar episodio completo con política epsilon-greedy
            trayectoria, costo_total, costo_traslado, costo_insatisfecha, costo_almacenamiento = self.ejecutar_episodio_epsilon_greedy(epsilon=self.epsilon)
            # PASO 2: CALCULAR TARGETS (RETORNOS MONTE CARLO)
            G = 0.0  # Retorno acumulado (suma de recompensas futuras)
            features_batch = []
            targets_batch = []
            
            # Recorrer trayectoria en orden inverso para calcular retornos correctamente
            # G_t = R_{t+1} + G_{t+1}  (sin descuento en esta implementación)
            for paso in reversed(trayectoria):
                r = paso['recompensa']  # Recompensa inmediata en este paso
                G = r + G  # Retorno: recompensa actual + retornos futuros
                c_st_at = paso['c_st_at']
                
                features_batch.append(paso['features'])
                targets_batch.append([G - c_st_at])
                
            # PASO 3: ENTRENAMIENTO DE LA RED NEURONAL (Batch Completo)
            # Entrenar toda la trayectoria de golpe acelera PyTorch inmensamente
            x_tensor = torch.tensor(np.array(features_batch), dtype=torch.float32)
            y_tensor = torch.tensor(targets_batch, dtype=torch.float32)

            self.net.train()
            pred = self.net(x_tensor)
            loss = self.loss_fn(pred, y_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # PASO 4: EVALUACIÓN PERIÓDICA
            if episodio % 100 == 0:
                self.actualizar_mejor_red()  # Evaluar y posiblemente guardar mejor red
                print(f"Episodio: {episodio:4d} | FO actual: {self.registro_fo[-1]:7.2f} | "
                      f"Mejor FO: {self.mejor_fo:7.2f} | Epsilon: {self.epsilon:.3f}")

        # FINALIZACIÓN DEL ENTRENAMIENTO
        fin = time.time()
        self.tiempo_entrenamiento = round(fin - inicio)
        
        print("-" * 50)
        print(f"Entrenamiento completado en {self.tiempo_entrenamiento} segundos")
        print(f"FO inicial: {self.registro_mejores_fo[0]:.2f}")
        print(f"Mejor FO encontrada: {self.mejor_fo:.2f}")
        mejora = ((self.registro_mejores_fo[0] - self.mejor_fo) / self.registro_mejores_fo[0]) * 100
        print(f"Mejora: {mejora:.1f}%")
        
        # Restaurar la mejor red encontrada
        self.net = self.mejor_red
