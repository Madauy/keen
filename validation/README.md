# Validation

Sistema de validación de LLMs reanudable que evalúa modelos de lenguaje en tareas de preguntas y respuestas. Procesa preguntas desde archivos CSV, genera respuestas usando modelos de HuggingFace, y evalúa la correctitud usando similitud semántica.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Uso básico
python main.py --data /ruta/a/datos.csv

# Con opciones
python main.py --data ./preguntas.csv --model "Qwen/Qwen3-4B-Instruct" --language es --limit 100 --country chile,usa
```

### Argumentos CLI

- `--data`: Ruta al archivo CSV de entrada (requerido)
- `--model`: Modelo de HuggingFace a usar (por defecto: `tiiuae/Falcon3-3B-Instruct`)
- `--language`: Idioma para los prompts - `es` (español) o `en` (inglés)
- `--limit`: Número máximo de preguntas a procesar
- `--country`: Filtrar por país (lista separada por comas)

## Arquitectura

El sistema tiene tres componentes principales:

- **main.py**: Punto de entrada CLI. Parsea argumentos, orquesta el loop de validación y maneja interrupciones de forma elegante
- **evaluator.py**: Contiene `Evaluator` (inferencia LLM) y `SimilarityEvaluator` (comparación de respuestas usando sentence-transformers)
- **state_manager.py**: Maneja la ejecución reanudable mediante archivos de estado JSON. Los archivos de estado se nombran `state_{hash}.json` basándose en el hash de la ruta del CSV

### Flujo de Datos

1. StateManager carga/crea el estado desde el CSV (columnas: entidad, relacion, objetos, pregunta, respuestas, obtenido_de, respuestas_aliases)
2. Evaluator genera respuestas usando formato de chat template con ejemplos few-shot
3. SimilarityEvaluator compara respuestas usando coincidencia exacta primero, luego similitud semántica (umbral: 0.80). Los años requieren solo coincidencia exacta.
4. Los resultados se guardan inmediatamente después de cada pregunta (escrituras JSON atómicas)

### Configuración

Configuración por defecto en `config.py`:

- Modelo por defecto: `tiiuae/Falcon3-3B-Instruct`
- Modelo de similitud: `sentence-transformers/distiluse-base-multilingual-cased-v2`
- Idiomas soportados: `es` (español), `en` (inglés)

## Desarrollo

```bash
# Lint del código
ruff check .
ruff format .
```

Estilo de código: ruff con indentación de 2 espacios y líneas de máximo 100 caracteres.
