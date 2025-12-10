# KEEN - Knowledge Estimation in Neural Networks

Implementación de KEEN para medir el conocimiento de LLMs sobre entidades sin generar tokens. Este proyecto evalúa si los modelos de lenguaje tienen más conocimiento sobre entidades de Estados Unidos versus entidades de Latinoamérica.

## Descripción

KEEN (Knowledge Estimation in Neural Networks) es una metodología que permite estimar cuánto "sabe" un modelo de lenguaje sobre una entidad específica analizando sus representaciones internas, sin necesidad de generar texto.

**Hipótesis de investigación**: Los LLMs tienen un sesgo de conocimiento hacia entidades estadounidenses en comparación con entidades latinoamericanas.

**Modelo base**: Falcon3-3B-Instruct

## Pipeline

El proyecto consta de tres etapas secuenciales:

```plaintext
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  VALIDATION │  →   │  EXTRACTOR  │  →   │    PROBE    │
│             │      │             │      │             │
│  Q&A eval   │      │   Hidden    │      │    MLP      │
│  → scores   │      │   States    │      │  training   │
└─────────────┘      └─────────────┘      └─────────────┘
     CSV              score.json              Pearson
   entrada           + vectores           correlation
```

1. **Validation**: Evalúa el LLM en tareas de Q&A para obtener accuracy por entidad
2. **Extractor**: Extrae hidden states del token de la entidad (capas 15-17)
3. **Probe**: Entrena un regresor MLP para predecir accuracy desde los hidden states

## Módulos

### Validation

Evalúa modelos de lenguaje en preguntas y respuestas usando similitud semántica. Genera scores de accuracy por entidad.

[Ver documentación completa](./validation/README.md)

### Extractor

Extrae representaciones internas de Falcon3-3B-Instruct usando la metodología KEEN. Soporta tres métodos: Hidden States, Vocabulary Projection y Top-k VP.

[Ver documentación completa](./extractor/README.md)

### Probe

Entrena una sonda MLP para predecir scores de accuracy a partir de los hidden states extraídos. Usa correlación de Pearson como métrica.

[Ver documentación completa](./probe/README.md)

## Quick Start

```bash
# 1. Validation - Evaluar Q&A
cd validation
pip install -r requirements.txt
python main.py --data ./QA_final.csv

# 2. Extractor - Extraer hidden states
cd ../extractor
pip install -r requirements.txt
python main.py --data ./QA_final.csv --method hs

# 3. Probe - Entrenar sonda
cd ../probe
pip install -r requirements.txt
python utils.py paper
python main.py --prompt paper --learning_rate 0.01 --max_iter 100 --batch_size 32
```

## Formato de Datos

El CSV de entrada debe contener las siguientes columnas:

```plaintext
entidad,relacion,objetos,pregunta,respuestas,obtenido_de,respuestas_aliases
```

El campo `obtenido_de` contiene códigos de país usados para filtrar (ej: `chile`, `usa`, `argentina`).

## Desarrollo

```bash
# Lint (en cualquier módulo)
ruff check .
ruff format .
```

## Referencias

> Gottesman & Geva (2024). "Estimating knowledge in large language models without generating a single token." EMNLP 2024.
