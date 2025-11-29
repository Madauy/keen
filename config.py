# Modelo por defecto
DEFAULT_MODEL = "tiiuae/Falcon3-3B-Instruct"

# Columnas requeridas del CSV
REQUIRED_COLUMNS = [
  "entidad",
  "relacion",
  "objetos",
  "pregunta",
  "respuestas",
  "obtenido_de",
  "respuestas_aliases",
]

# Métodos de extracción disponibles
EXTRACTION_METHODS = ["hs", "vp", "vp-k"]

# Configuración por defecto para VP-k
DEFAULT_K = 50
DEFAULT_VPK_MODE = "bidirectional"

# Configuración de idiomas con templates de prompt
LANGUAGE_CONFIGS = {
  "es": {
    "prompt_template": "Dime todo lo que sabes de {entidad}",
  },
  "en": {
    "prompt_template": "Tell me everything you know about {entidad}",
  },
}

# Directorio para archivos de estado
STATE_DIR = "states"

# Tamaño del buffer para guardar vectores (cada N filas)
# Note: Use smaller values for vp method (large 131K vectors) vs hs method (small 3K vectors)
# Recommended: hs=1000, vp=50, vp-k=500
BUFFER_SIZE = 50

def get_language_config(language: str) -> dict:
  """Obtiene la configuración para un idioma específico.

  Args:
      language: Código de idioma ('es' o 'en')

  Returns:
      Diccionario con prompt_template

  Raises:
      ValueError: Si el idioma no está soportado
  """
  if language not in LANGUAGE_CONFIGS:
    raise ValueError(
      f"Idioma no soportado: {language}. Idiomas disponibles: {list(LANGUAGE_CONFIGS.keys())}"
    )
  return LANGUAGE_CONFIGS[language]
