class QlearningAprox(PoliticasAprox):
    ''' 
    Descripción:
        * Clase asociada al proceso de Aproximación de Q learning con función lineal
    Parámetros:
        * racetrack: Objeto que contiene todos los elementos de la simulación
        * episodes: Número de episodios a realizar
        * gamma: Factor de descuento
        * alpha: Factor de aprendizaje
        * epsilon: Factor de epsilon greedy
        * learning_rate: Factor de aprendizaje
        * betas: Vector de betas
    '''
    def __init__(self, racetrack, episodes, gamma, epsilon, learning_rate ,betas):
        super().__init__(racetrack, episodes, gamma, epsilon, learning_rate,betas)
    
    def run(self):
        '''
        Descripción:
            * Método que ejecuta el algoritmo de Q-Learning para aprender la política óptima.

        Parámetros:
            * None

        Return:
            * None
        '''
        for episode in range(1, self.episodes + 1):
            self.correr_episodio()
        
    def correr_episodio(self):
        '''
        Descripción:
            * Método que corre un episodio usando Q-Learning.

        Parámetros:
            * None

        Return:
            * None
        '''
        
        self.racetrack.reset()
        state = self.racetrack.get_state()
        done = False

        while not done:
            action, features = self.epsilon_greedy(state) # se obtiene la mejor acción pero no las features del estado accion, pq no se usan
            reward = self.step(action)  # se obtiene la recopensa
            next_state = self.racetrack.get_state() # Se obtiene el siguiente estado
            next_action = self.politica_optima(next_state) # Se obtiene la mejor acción en el estado siguiente
            next_features = self.obtener_features(next_state, next_action)  # se obtienen las features del estado-accion siguiente
            next_qsa = np.dot(next_features, self.betas) # Se aproxima el Q valor siguiente
            G = reward + self.gamma * next_qsa # Se obtiene la recompensa considerando el futuro
            self.SGD(features, G) # Se entrenan los betas
            state = next_state
            done = self.finished(state)