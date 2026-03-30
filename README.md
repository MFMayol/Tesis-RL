# SDMMIRP

Este proyecto aborda un problema dinámico de **Enrutamiento de Vehículos y Control de Inventarios (IRP - Inventory Routing Problem)**. Su objetivo es tomar decisiones secuenciales en el tiempo sobre qué clientes visitar y cuánto producto entregar para minimizar una función de costos compuesta por:
1. **Costos de Traslado**: Distancia recorrida por los vehículos.
2. **Costos de Almacenamiento**: Mantención de inventario en los clientes.
3. **Costos de Demanda Insatisfecha**: Penalización por quiebres de stock.

Para lograr esto, el proyecto emplea una variedad de enfoques, desde heurísticas miopes y algoritmos predictivos (Rollout), hasta técnicas de Aprendizaje por Refuerzo (Monte Carlo, Características de Fourier, Redes Neuronales).

---

## 📂 Estructura del Proyecto

```text
AConC/
├── RunMC.py            # Ejecución del agente Monte Carlo tradicional
├── RunMCF.py           # Ejecución del agente Monte Carlo con Expansión de Fourier
├── RunMCFH.py          # Ejecución del agente Monte Carlo Fourier enfocado en Almacenamiento
├── RunNN.py            # Ejecución del agente Q-Learning con Redes Neuronales Profundas
├── RunROC.py           # Ejecución de la política de Rollout con Clústeres (K-Means)
├── run_clusterh.slurm  # Script para ejecución en paralelo en clústeres HPC (SLURM)
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
│       ├── QLNN.py
│       └── MCFAlmacenamiento.py
└── Resultados_*/       # (Generado automáticamente) Resultados en Excel por modelo
└── Graficos_*/         # (Generado automáticamente) Gráficos de entrenamiento
```

---

## 🚀 Scripts de Ejecución Principal

Los archivos `Run*.py` son los puntos de entrada para ejecutar simulaciones y entrenamientos en instancias específicas. Todos reciben la ruta de un archivo XML (la instancia) como argumento por consola.

- **`RunMC.py`**: Ejecuta el entrenamiento de una política mediante el método de **Monte Carlo clásico** (`MonteCarlo`), guardando los costos históricos y el registro de convergencia en un archivo `.xlsx` y un gráfico `.png`.
- **`RunMCF.py`**: Entrena un modelo **Monte Carlo con Características de Fourier** (`MonteCarlo_Fourier`), un método de aproximación de valor lineal que maneja espacios de estado continuos.
- **`RunMCFH.py`**: Similar a MCF, pero utiliza una variante híbrida o con enfoque ajustado en almacenamiento (`MCFourierH`).
- **`RunNN.py`**: Ejecuta **Deep Q-Learning (QLNN)** utilizando redes neuronales para aproximar la función de valor $Q(s, a)$.
- **`RunROC.py`**: Ejecuta simulaciones utilizando la política **RollOutCluster** (`RollOutCluster`), la cual no requiere entrenamiento previo, sino que toma decisiones en línea (Lookahead) basadas en múltiples trayectorias de simulación hacia adelante.

### Ejecución en Clúster HPC
- **`run_clusterh.slurm`**: Script configurado para lanzar trabajos paralelos (Job Arrays) en un entorno SLURM. Útil para correr decenas de instancias XML simultáneamente dividiendo la carga en nodos computacionales.

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

### 2. Políticas de Decisión (`src/politicas/`)

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

### 3. Herramientas Auxiliares

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
