import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .modelos_avanzados import Modelos_de_aproximacion
import time
from src.Estado import Estado
import copy
from collections import deque
import random
from src.Proceso import Proceso

class QNetwork(nn.Module):
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
        return self.model(x)



class QLNN(Modelos_de_aproximacion):
    def __init__(self, instancia, proceso, episodios, epsilon, learning_rate, gamma, batch_size, buffer_size, estructura):
        super().__init__(instancia, proceso, episodios, epsilon, learning_rate)    
        self.gamma = gamma # float que representa el factor de descuento del futuro
        self._crear_betas() # inicializa el self.betas (np.array)
        self.input_dim = len(self.betas) # da el número de features de entrada que recibirá por muestra (int)
        self.net = QNetwork(self.input_dim, hidden_layers=estructura)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate) # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.batch_size = batch_size
        self.loss_fn = nn.MSELoss()
        self.buffer = deque(maxlen=buffer_size)  # replay buffer con límite de experiencia
        self.start_training_after = 10  # para evitar entrenar con buffer vacío
        self.mejor_fo = None
        self.mejor_red = None
        self.registro_fo = [] # registra la fo obtenida por la red neuronal cada 100 episodios
        self.registro_mejores_fo = [] # registra la mejor fo obtenida por la red neuronal cada 100 episodios
        self.target_net = copy.deepcopy(self.net)
        self.pasos_totales = 0  # Contador de pasos globales
        self.target_net.eval()
        self.mejor_red_inicial()


    def tomar_accion_epsilon_greedy(self, estado):
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
            accion = self.politica_optima(estado, acciones)
        return accion

    def politica_optima(self, state, acciones):
        '''
        Descripción:
        *   Método que devuelve la mejor acción para un estado utilizando la red neuronal Q(s,a)

        Parámetros:
        *   state: Estado actual
        *   acciones: Lista de acciones posibles (lista de diccionarios)

        Return:
        *   action: Mejor acción (la que tiene el menor Q)
        '''
        self.net.eval()  # Ponemos la red en modo evaluación
        
        # Optimización: Evaluar todas las acciones en un solo batch
        with torch.no_grad():
            features_list = [self._obtener_features(state, action) for action in acciones]
            x_batch = torch.tensor(np.array(features_list), dtype=torch.float32)
            
            # Pasamos todo el batch por la red de una sola vez
            q_values = self.net(x_batch).squeeze(-1).numpy()
            
            c_st_at_list = [self.proceso.determinar_c_st_at(state, action) for action in acciones]
            c_st_at_array = np.array(c_st_at_list, dtype=np.float32)
            
            total_q_values = q_values + c_st_at_array
            
        # Encontrar el índice con el menor valor Q
        best_idx = np.argmin(total_q_values)
        return acciones[best_idx]

    def politica_optima_target(self, state, acciones):
        """
        Determina la acción óptima para un estado dado utilizando la red neuronal "target".

        Esta política es "codiciosa" (greedy) porque selecciona la acción que la red
        estima como la mejor en el momento actual, basándose en un criterio de minimización.

        Parámetros:
        ----------
        state: Dict[str, Any]
            Un diccionario u otro objeto que representa el estado actual del entorno.

        acciones: List[Dict[str, Any]]
            Una lista de acciones posibles desde el estado actual. Se asume que cada
            acción es un diccionario.

        Retorno:
        -------
        Dict[str, Any]
            La acción seleccionada del listado `acciones` que minimiza el valor Q predicho.
        """
        self.target_net.eval()  # Ponemos la red en modo evaluación
        
        with torch.no_grad():
            features_list = [self._obtener_features(state, action) for action in acciones]
            x_batch = torch.tensor(np.array(features_list), dtype=torch.float32)
            
            q_values = self.target_net(x_batch).squeeze(-1).numpy()
            
            c_st_at_list = [self.proceso.determinar_c_st_at(state, action) for action in acciones]
            c_st_at_array = np.array(c_st_at_list, dtype=np.float32)
            
            total_q_values = q_values + c_st_at_array
            
        best_idx = np.argmin(total_q_values)
        return acciones[best_idx]

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
        '''Entrenamiento Q-learning con Replay Buffer y NN'''
        proceso = self.proceso
        es_terminal = self.es_terminal
        inicio = time.time()
        for episodio in range(self.episodios):
            estado = proceso.determinar_estado_inicial()
            # El bucle while ahora termina si el estado es terminal desde el inicio o después de una transición.
            while not es_terminal(estado):
                accion = self.tomar_accion_epsilon_greedy(estado)
                features = self._obtener_features(estado, accion)
                c_st_at = proceso.determinar_c_st_at(estado, accion)
                nuevo_estado, recompensa, *_ = proceso.transicion(estado, accion)
                done = es_terminal(nuevo_estado)

                # Almacenamos la tupla completa en el buffer.
                # Nota: 'accion' es el objeto acción, no un índice.
                self.buffer.append((estado, accion, features, recompensa, nuevo_estado, done, c_st_at))
                
                # El estado actual se convierte en el nuevo estado para el siguiente paso.
                estado = nuevo_estado

                # Solo entrenamos si el buffer tiene suficientes muestras.
                if len(self.buffer) >= self.start_training_after:
                    self.actualizar_desde_buffer()

            # Las siguientes comprobaciones se realizan al final de cada episodio.
            if episodio % 100 == 0:
                self.actualizar_mejor_red()
                print(f"Episodio: {episodio} ok, FO actual: {self.mejor_fo}")

            if episodio % 20 == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        fin = time.time()
        self.tiempo_entrenamiento = round(fin - inicio)

    def actualizar_desde_buffer(self):
        """
        Actualiza los pesos de la red Q (self.net) utilizando una muestra (batch)
        del buffer de repetición. Mantiene la estructura de bucles pero corrige
        la lógica para ser funcional.
        """
        # No hacer nada si el buffer es muy pequeño.
        if len(self.buffer) < self.batch_size:
            return

        # Poner la red principal en modo de entrenamiento.
        self.net.train()
        # Muestrear un batch aleatorio del buffer.
        batch = random.sample(self.buffer, self.batch_size)

        # --- 1. Descomposición y Preparación de Datos ---
        # Descomponemos el batch en listas separadas. Guardamos las 'features'
        # que ya fueron precalculadas para los pares (estado, acción) originales.
        states, actions, features, rewards, next_states, dones, c_st_ats = zip(*batch)

        # Convertimos las features de los estados originales y las recompensas a tensores.
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        c_st_at_tensor = torch.tensor(c_st_ats, dtype=torch.float32)

        # Convertimos los flags de terminación a un tensor booleano. Es más limpio para máscaras.
        dones_tensor = torch.tensor(dones, dtype=torch.bool)


        # --- 2. Cálculo de los Targets (Valor Futuro) ---
        # Identificar los estados que no son terminales
        non_final_mask = ~dones_tensor
        non_final_next_states = [s for i, s in enumerate(next_states) if non_final_mask[i]]

        # Inicializamos el valor Q del siguiente estado. Será cero para estados terminales.
        next_q_values = torch.zeros(self.batch_size, dtype=torch.float32)

        # Solo si hay estados no terminales, calculamos su valor
        if non_final_next_states:
            # --- Lógica Clave de Double DQN (Vectorizada) ---
            # Este bucle es para la SELECCIÓN de la acción, que es difícil de vectorizar
            # porque el número de acciones varía por estado.
            features_de_mejores_acciones = []
            c_st_at_de_mejores_acciones = []
            with torch.no_grad():
                for next_state in non_final_next_states:
                    acciones_siguientes = self._obtener_acciones(next_state)
                    if not acciones_siguientes:
                        # Si no hay acciones, el feature es un vector de ceros.
                        features_de_mejores_acciones.append(np.zeros(self.input_dim, dtype=np.float32))
                        c_st_at_de_mejores_acciones.append(0.0)
                        continue
                    
                    # 1. SELECCIÓN: La red principal (self.net) elige la mejor acción.
                    mejor_accion = self.politica_optima(next_state, acciones_siguientes)
                    
                    # Recolectamos los features de la mejor acción.
                    features_de_mejores_acciones.append(self._obtener_features(next_state, mejor_accion))
                    c_st_at_de_mejores_acciones.append(self.proceso.determinar_c_st_at(next_state, mejor_accion))

            # 2. EVALUACIÓN (Vectorizada): La red objetivo (target_net) evalúa el valor
            # de TODAS las mejores acciones en un solo batch.
            if features_de_mejores_acciones:
                features_batch_next = torch.tensor(np.array(features_de_mejores_acciones), dtype=torch.float32)
                c_st_at_next_tensor = torch.tensor(np.array(c_st_at_de_mejores_acciones), dtype=torch.float32)
                with torch.no_grad():
                    # Un solo forward pass para todas las evaluaciones.
                    q_values_evaluados = self.target_net(features_batch_next).squeeze(-1)
                    total_q_values_evaluados = q_values_evaluados + c_st_at_next_tensor

                # Asignamos los valores Q calculados solo a las posiciones de los estados no terminales.
                next_q_values[non_final_mask] = total_q_values_evaluados

        # Calculamos el target final con la ecuación de Bellman.
        # Para estados terminales, next_q_values es 0, por lo que target = recompensa.
        targets_tensor = rewards_tensor + self.gamma * next_q_values - c_st_at_tensor

        # --- 3. Actualización de la Red ---
        # Obtenemos la predicción de Q(s,a) de la red principal para las features originales.
        # Dado que tu red devuelve un solo valor, la forma será [batch_size, 1].
        self.net.train() # Aseguramos que vuelva a modo entrenamiento tras politica_optima
        current_q_values = self.net(features_tensor)

        # Calculamos la pérdida. Las formas deben coincidir.
        # current_q_values.shape -> [batch_size, 1]
        # targets_tensor.unsqueeze(1).shape -> [batch_size, 1]
        loss = self.loss_fn(current_q_values, targets_tensor.unsqueeze(1))

        # Realizamos la retropropagación para actualizar los pesos de self.net.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()