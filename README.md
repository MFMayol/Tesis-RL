# Proyecto SDMMIRP

Este proyecto aborda un problema dinámico de **SDMM-Inventory Routing Problem**. Su objetivo es tomar decisiones secuenciales en el tiempo sobre qué clientes visitar y cuánto producto entregar para minimizar una función de costos compuesta por:
1. **Costos de Traslado**: Distancia recorrida por los vehículos.
2. **Costos de Almacenamiento**: Mantención de inventario en los clientes.
3. **Costos de Demanda Insatisfecha**: Penalización por quiebres de stock.

Para lograr esto, el proyecto emplea una variedad de enfoques, desde heurísticas miopes y algoritmos predictivos (Rollout), hasta técnicas de Aprendizaje por Refuerzo (Monte Carlo, Características de Fourier, Redes Neuronales).

---

## 📂 Estructura del Proyecto

```text

├── src/                # Código fuente principal
│   ├── Proceso.py
│   ├── Instancia.py
│   ├── Estado.py
│   ├── FuncionesAuxiliares.py
│   └── politicas/      # Algoritmos de decisión
│       ├── base.py
│       ├── simples.py
│       ├── rollout.py
│       ├── modelos_avanzados.py
│       ├── MCNN.py
│       ├── QLNN.py
│       └── MCFAlmacenamiento.py
└── Resultados_*/       # (Generado automáticamente) Resultados en Excel por modelo
└── Graficos_*/         # (Generado automáticamente) Gráficos de entrenamiento
```

---

## 🧠 Arquitectura del Código Fuente (`src/`)

El diseño del código sigue un esquema de Simulación de Procesos de Decisión de Markov (MDP), separando el entorno (Instancia, Estado, Proceso) del agente decisor (Políticas).

### 1. Entorno y Transiciones

*   **`Instancia.py`** *(Clase `Instancia`)*: 
    Carga y almacena los datos estáticos del problema desde archivos XML. Contiene la información de vehículos (capacidad, velocidad), clientes (posiciones, tasas de demanda, costos) y productos (pesos, volúmenes).

*   **`Estado.py`** *(Clase `Estado`)*: 
    Representa una "foto" del sistema en un tiempo $t$. Almacena los inventarios actuales de vehículos y clientes, las posiciones de los vehículos en el mapa cartesiano, y la planificación (hacia dónde se dirige cada vehículo y cuánto entregará).

*   **`Proceso.py`** *(Clase `Proceso`)*: 
    **Es el motor del entorno.** 
    Contiene la lógica de simulación física y temporal. Su método principal es `transicion(estado, accion)`, el cual:
    1. Actualiza posiciones de vehículos interpolando su velocidad en el tiempo.
    2. Transfiere productos del vehículo al cliente al llegar a destino.
    3. Simula las demandas estocásticas en cada cliente.
    4. Calcula las recompensas paso a paso (Costos de traslado + almacenamiento + escasez).
    5. Detiene la simulación temporal cuando se necesita que el agente tome una nueva decisión (ej. un vehículo se queda sin ruta, el inventario cae bajo un umbral o transcurre un horizonte fijo).

### 2. Políticas de referencia (`src/politicas/`)

Las políticas son responsables de tomar un objeto `Estado` y devolver una acción (`Dict`).

*   **`base.py`** *(Clase `Politica`)*:
    Clase abstracta base. Contiene el bucle principal de ejecución de episodios (`run()`) y la lógica fundamental para construir una asignación de entrega (`planificar_entrega`), repartiendo el inventario proporcionalmente a las demandas.

*   **`simples.py`** *(Clases `PoliticasSimples`, `PoliticaSimple`, `PoliticaSimpleClusterisada`)*:
    Políticas heurísticas y "miopes". 
    - Buscan el primer vehículo sin asignar.
    - Si el vehículo tiene inventario bajo el umbral (ej. < 20%), lo envían de regreso al Depot.
    - Si tiene inventario, calculan qué clientes están más críticos (evaluando el ratio de inventario sobre demanda promedio).
    - La versión **Clusterizada** limita la búsqueda de clientes críticos solo a aquellos que pertenecen a la misma zona geográfica del vehículo, previamente calculada con K-Means.

*   **`rollout.py`** *(Clases `RollOutSimple`, `RollOutCluster`)*:
    Algoritmos de mejora de políticas (Lookahead). Evalúan todas las acciones "factibles" en el estado actual, realizando $N$ simulaciones completas (trayectorias) hacia el futuro utilizando una política simple (heurística) como política base. Luego, eligen la acción que entregó el costo promedio esperado más bajo.

### 3. Modelos Avanzados de Aprendizaje por Refuerzo

Esta sección cubre los algoritmos más complejos que utilizan aproximación de funciones para resolver el problema.

*   **`modelos_avanzados.py`** *(Clases `MonteCarlo`, `MonteCarlo_Fourier`)*:
    *   **`MonteCarlo` (MC)**: Implementa el algoritmo clásico *On-Policy Monte Carlo Control*. Aprende el valor de los pares estado-acción promediando los retornos (costos totales) obtenidos al final de episodios completos. Utiliza una política $\epsilon$-greedy para el balance entre exploración y explotación, y almacena la función de valor en una tabla (diccionario).
    *   **`MonteCarlo_Fourier` (MCF)**: Es una extensión que utiliza **aproximación de funciones** con una combinación lineal de **bases de Fourier**. Esto le permite encontrar relaciones No Lineales entre los features. El algoritmo aprende los pesos (coeficientes `betas`) de estas funciones base para aproximar la función de valor.

*   **`MCNN.py`** *(Clase `MCNN`)*:
    *   Implementa un algoritmo de **Monte Carlo Profundo con Redes Neuronales**. Al igual que el método Monte Carlo clásico, espera a que un episodio termine para actualizar su política.
    *   Para cada par estado-acción $(s_t, a_t)$ visitado, calcula el **retorno completo** $G_t$ (la suma de todas las recompensas futuras hasta el final del episodio).
    *   Utiliza una red neuronal que aprende a mapear las características de un estado-acción a su retorno $G_t$ correspondiente. A diferencia de Q-Learning, no utiliza bootstrapping (no estima el valor de un estado basándose en el valor de estados sucesores), lo que puede reducir el sesgo a costa de una mayor varianza.

*   **`QLNN.py`** *(Clase `QLNN`)*:
    *   Implementa el algoritmo **Deep Q-Learning**, un método *off-policy* de Diferencia Temporal (TD).
    *   Utiliza una **Red Neuronal Profunda** para aproximar la función de valor-acción óptima, $Q^*(s, a)$.
    *   Incorpora dos técnicas clave para estabilizar el entrenamiento:
        1.  **Experience Replay**: Almacena las transiciones $(s, a, r, s')$ en un búfer de memoria y entrena la red con mini-lotes de muestras aleatorias. Esto rompe la correlación entre muestras consecutivas.
        2.  **Target Network**: Usa una segunda red (red objetivo) para calcular los valores Q de los estados siguientes. Esta red se actualiza de forma más lenta, lo que proporciona un objetivo de entrenamiento más estable.
    *   La red se entrena para minimizar el error entre el valor Q predicho y el "objetivo Q" calculado con la ecuación de Bellman.

*   **`MCFAlmacenamiento.py`** *(Clase `MCFourierH`)*:
    *   Corresponde a una variante especializada del modelo `MonteCarlo_Fourier`. La `H` hace referencia a que es el que se usa para las instancias de almacenamiento. Este modelo extiende el espacio de acciones que hay.

### 4. Herramientas Auxiliares

*   **`FuncionesAuxiliares.py`**:
    Un conjunto de métodos matemáticos altamente optimizados mediante NumPy:
    - `distancia_euclidiana`: Cálculo de distancias en el espacio cartesiano.
    - `kmeans_clustering_sklearn`: Segmentación geográfica de clientes en sub-zonas (clústeres).
    - `fourier_transform_features_vectorized`: Transformaciones de bases no lineales (seno/coseno) utilizadas por los algoritmos de aproximación de valor (MCF).

---

## ⚙️ Uso y Ejecución Local

Para correr cualquiera de los experimentos localmente, asegúrate de tener instaladas las dependencias matemáticas estándar (`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `openpyxl`).

Ejecuta el script deseado pasándole la ruta al archivo de la instancia como primer argumento de la línea de comandos:

```bash
# Ejecutar Q-Learning con Redes Neuronales
python RunNN.py "rutas/hacia/tus/instancias/instancia_C1M1P1_01.xml"

# Ejecutar Política de Rollout
python RunROC.py "rutas/hacia/tus/instancias/instancia_C1M1P1_01.xml"
```

**Resultados generados:**
Tras la ejecución, el script creará automáticamente dos carpetas (`Resultados_{Modelo}` y `Graficos_{Modelo}`). Dentro encontrarás un archivo Excel consolidando métricas (Tiempos, Costos y % de cada penalización) y un gráfico de convergencia si aplica.
