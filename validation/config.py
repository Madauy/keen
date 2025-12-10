# Model configuration
DEFAULT_MODEL = "tiiuae/Falcon3-3B-Instruct"
MAX_NEW_TOKENS = 256

# Similarity evaluation configuration
SIMILARITY_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"
SIMILARITY_THRESHOLD = 0.80

# Required CSV columns
REQUIRED_COLUMNS = [
  "entidad",
  "relacion",
  "objetos",
  "pregunta",
  "respuestas",
  "obtenido_de",
  "respuestas_aliases",
]

# Language-specific configurations
LANGUAGE_CONFIGS = {
  "es": {
    "system_message": (
      "Eres un asistente experto que responde preguntas de forma concisa y precisa. "
      "Responde ÚNICAMENTE con la respuesta, sin explicaciones adicionales ni texto extra. "
      "Si la pregunta tiene múltiples respuestas posibles, elige la más común o conocida."
    ),
    "few_shot_examples": [
      {
        "pregunta": "Contesta la siguiente pregunta: ¿En qué año comenzó la Primera Guerra Mundial?",
        "respuesta": "1914",
      },
      {
        "pregunta": "Contesta la siguiente pregunta: ¿Cuál es la capital de Chile?",
        "respuesta": "Santiago",
      },
      {
        "pregunta": "Contesta la siguiente pregunta: ¿A qué deporte pertenece el fútbol?",
        "respuesta": "deporte de equipo",
      },
      {
        "pregunta": "Contesta la siguiente pregunta: ¿Quién escribió Don Quijote de la Mancha?",
        "respuesta": "Miguel de Cervantes",
      },
      {
        "pregunta": "Contesta la siguiente pregunta: ¿Cuál es el río más largo de Sudamérica?",
        "respuesta": "Amazonas",
      },
    ],
  },
  "en": {
    "system_message": (
      "You are an expert assistant that answers questions concisely and accurately. "
      "Respond ONLY with the answer, without additional explanations or extra text. "
      "If the question has multiple possible answers, choose the most common or well-known one."
    ),
    "few_shot_examples": [
      {
        "pregunta": "Answer the following question: In what year did World War I begin?",
        "respuesta": "1914",
      },
      {
        "pregunta": "Answer the following question: What is the capital of the United States?",
        "respuesta": "Washington D.C.",
      },
      {
        "pregunta": "Answer the following question: What sport does basketball belong to?",
        "respuesta": "team sport",
      },
      {
        "pregunta": "Answer the following question: Who wrote Romeo and Juliet?",
        "respuesta": "William Shakespeare",
      },
      {
        "pregunta": "Answer the following question: What is the longest river in Africa?",
        "respuesta": "Nile",
      },
    ],
  },
}


def get_language_config(language: str) -> dict:
  """Get the configuration for a specific language.

  Args:
      language: Language code ('es' or 'en')

  Returns:
      Dictionary with system_message and few_shot_examples

  Raises:
      ValueError: If the language is not supported
  """
  if language not in LANGUAGE_CONFIGS:
    raise ValueError(
      f"Unsupported language: {language}. Supported languages: {list(LANGUAGE_CONFIGS.keys())}"
    )
  return LANGUAGE_CONFIGS[language]
