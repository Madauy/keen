# keen-probe-latinoamerica

Proyecto de entrenamiento de una sonda (probe) MLP para predecir scores basándose en representaciones vectoriales normalizadas, basado en el código original de [keen_estimating_knowledge_in_llms](https://github.com/dhgottesman/keen_estimating_knowledge_in_llms).

## Estructura del Repositorio

```plaintext
keen-probe-latinoamerica/
├── data/                        # Directorio de datos
│   ├── hidden_states/           # .npz con representaciones y row_indices
│   ├── score.json               # Scores por entidad
│   └── hidden_states_hs_*.pkl   # Salida de utils.py (DataFrame pickled)
├── hyperparams.py               # Hiperparámetros del modelo
├── mlp_regressor.py             # MLPRegressor de código fuente KEEN debuggeado
├── utils.py                     # Genera los .pkl con hidden_states y metadatos
├── main.py                      # Entrena la sonda a partir de los .pkl
└── probe.ipynb                  # Notebook principal de experimentos
```

## Archivos Principales

### `mlp_regressor.py`

MLPRegressor de código fuente KEEN debuggeado, que incluye:

- Arquitectura de red neuronal simple (una capa lineal + sigmoid)
- Función de pérdida MSE
- Entrenamiento con validación
- Métricas de evaluación (Pearson correlation)
- Guardado de mejores pesos durante el entrenamiento

### `utils.py`

- Combina `data/score.json` y un archivo `.npz` de `data/hidden_states/` (llaves `PATHS`: `en`, `es`, `paper`).
- Produce un DataFrame de pandas pickled (`data/hidden_states_hs_<key>.pkl`) con columnas:
  - `subject`, `country`, `accuracy`, `total_examples`
  - `hidden_states` (`numpy.ndarray` de la representación normalizada)
- Uso: `python utils.py <en|es|paper>` y luego `pd.read_pickle("data/hidden_states_hs_<key>.pkl")`.

## Datos

### Formato de Datos

- **`hidden_states_hs_<key>.npz`**: Contiene `representations_normalized` y `row_indices` (índices para cruzar con `score.json`).
- **`score.json`**: Lista de objetos con llaves `entidad`, `pais`, `score`, `vector_index`, `total_preguntas`.
- **`hidden_states_hs_<key>.pkl`**: DataFrame pickled con las columnas anteriores más `hidden_states` (`numpy.ndarray`).

## Generación y entrenamiento rápidos

1. Generar el DataFrame pickled:
   - `python utils.py paper` (o `en`, `es`).
2. Entrenar la sonda con los datos pickled:
   - `python main.py --prompt paper --learning_rate 0.01 --max_iter 100 --batch_size 32`
   - `--country` filtra por país (opcional).

## Características Técnicas

- Entrenamiento determinístico (seed fijo en 42)
- Uso de GPU mediante CUDA
- Validación con Pearson correlation
- Guardado automático de mejores pesos durante entrenamiento
- Sin logging de WandB (removido del código original)

## Requisitos

- PyTorch
- NumPy
- pandas
- scikit-learn
- scipy

### Versiones probadas

- Python >= 3.12
- torch >= 2.9.1
- numpy >= 2.3.5
- pandas >= 2.3.3
- scikit-learn >= 1.7.2
- scipy >= 1.16.3
- wandb >= 0.23.1
