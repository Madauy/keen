# Extracción de Hidden States KEEN

Sistema para extraer representaciones internas de Falcon3-3B-Instruct usando la metodología KEEN (Knowledge Estimation in Neural Networks), permitiendo medir el conocimiento del modelo sobre entidades sin generar tokens.

## Instalación

```bash
pip install -r requirements.txt
```

**Dependencias principales:**

- `transformers>=4.30.0`
- `torch>=2.0.0`
- `pandas>=2.0.0`
- `scikit-learn`

## Uso

### Extracción básica

```bash
# Procesar todas las entidades con Hidden States
python main.py --data ./questions.csv --method hs

# Usar Vocabulary Projection Top-k
python main.py --data ./questions.csv --method vp-k --k 50

# Procesar en lotes de 100
python main.py --data ./questions.csv --limit 100
```

### Filtros

```bash
# Filtrar por país (campo obtenido_de)
python main.py --data ./questions.csv --country chile,usa

# Cambiar idioma del prompt
python main.py --data ./questions.csv --language en
```

### Resumir ejecución interrumpida

```bash
# Simplemente ejecutar el mismo comando - detecta progreso automáticamente
python main.py --data ./questions.csv --method hs
```

### Normalización

```bash
# Normalizar después de extracción completa
python main.py --data ./questions.csv --normalize-only
```

## Argumentos CLI

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--data` | Ruta al CSV con entidades (requerido) | - |
| `--method` | Método: `hs`, `vp`, `vp-k` | `hs` |
| `--k` | Valor k para VP-k | `50` |
| `--vpk-mode` | Modo VP-k: `positive`, `bidirectional`, `abs` | `bidirectional` |
| `--language` | Idioma del prompt: `es`, `en` | `es` |
| `--limit` | Máximo de filas por ejecución | Todas |
| `--country` | Filtro por países (separados por coma) | Todos |
| `--normalize-only` | Solo aplicar normalización global | - |

## Formato del CSV

El archivo CSV debe contener las siguientes columnas:

```plaintext
entidad,relacion,objetos,pregunta,respuestas,obtenido_de,respuestas_aliases
```

Ejemplo:

```csv
Balón de Oro,deporte,['futbol'],¿A qué deporte pertenece?,['futbol'],['argentina'],"[['futbol', 'balompie']]"
```

## Métodos de Extracción

| Método | Dimensión | Descripción |
|--------|-----------|-------------|
| `hs` | 3,072 | Hidden States directos de capas 15-17 |
| `vp` | 131,072 | Proyección al vocabulario completo |
| `vp-k` | 3×k | Top-k tokens más relevantes (interpretable) |

## Arquitectura

```plaintext
proyecto/
├── main.py              # CLI principal
├── extractor.py         # FalconHiddenStatesExtractor (métodos KEEN)
├── state_manager.py     # Gestión de estado resumible
├── config.py            # Configuración centralizada
└── states/              # Archivos de estado (JSON + NPZ)
```

### Flujo de datos

1. **Entrada**: CSV con entidades
2. **Prompt**: "Dime todo lo que sabes de {entidad}" (forward pass sin generación)
3. **Extracción**: Hidden states del token de la entidad en capas 15-17
4. **Buffer**: Guarda cada 10 filas al NPZ
5. **Normalización**: MinMax global al completar todas las entidades

### Archivos de estado

- `states/state_{hash}_{method}.json`: Metadatos y tracking de filas
- `states/state_{hash}_{method}.npz`: Vectores raw y normalizados

## Metodología KEEN

Este proyecto implementa la metodología descrita en:

> Gottesman & Geva (2024). "Estimating knowledge in large language models without generating a single token." EMNLP 2024.

**Principio**: Las representaciones internas del modelo en el token de una entidad contienen información sobre cuánto "sabe" el modelo de esa entidad, sin necesidad de generar texto.

## Lint

```bash
ruff check .
ruff format .
```
