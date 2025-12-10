import argparse
import logging
import os
import signal
import sys

from config import (
  DEFAULT_K,
  DEFAULT_VPK_MODE,
  EXTRACTION_METHODS,
  LANGUAGE_CONFIGS,
  get_language_config,
)
from extractor import FalconHiddenStatesExtractor
from state_manager import ExtractionStateManager

# Configurar logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Variable global para el state_manager (para manejo de señales)
_state_manager: ExtractionStateManager | None = None


def signal_handler(sig, frame):
  """Manejador de señales para guardar progreso antes de salir."""
  logger.info("\nInterrumpido por el usuario. Guardando progreso...")
  if _state_manager is not None:
    _state_manager.finalize()
    print_summary(_state_manager)
  sys.exit(0)


def parse_args() -> argparse.Namespace:
  """Parsea argumentos de línea de comandos.

  Returns:
      Namespace con los argumentos parseados
  """
  parser = argparse.ArgumentParser(
    description="Extracción de Hidden States KEEN - Sistema resumible para extraer representaciones internas de LLMs",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Ejemplos:
    python main.py --data ./questions.csv
    python main.py --data ./questions.csv --method vp-k --k 50
    python main.py --data ./questions.csv --language en
    python main.py --data ./questions.csv --limit 100
    python main.py --data ./questions.csv --country chile,usa
    python main.py --data ./questions.csv --normalize-only

El programa guarda progreso automáticamente después de cada lote.
Si se interrumpe, simplemente ejecute de nuevo con los mismos argumentos para continuar.
        """,
  )

  parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Ruta al archivo CSV con las entidades",
  )

  parser.add_argument(
    "--method",
    type=str,
    default="hs",
    choices=EXTRACTION_METHODS,
    help="Método de extracción: hs (Hidden States), vp (Vocabulary Projection), vp-k (Top-k VP). Default: hs",
  )

  parser.add_argument(
    "--k",
    type=int,
    default=DEFAULT_K,
    help=f"Valor k para el método VP-k. Default: {DEFAULT_K}",
  )

  parser.add_argument(
    "--vpk-mode",
    type=str,
    default=DEFAULT_VPK_MODE,
    choices=["positive", "bidirectional", "abs"],
    help=f"Modo de selección para VP-k. Default: {DEFAULT_VPK_MODE}",
  )

  parser.add_argument(
    "--language",
    type=str,
    default="es",
    choices=list(LANGUAGE_CONFIGS.keys()),
    help="Idioma del prompt de consulta. Default: es",
  )

  parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Máximo de filas a procesar en esta ejecución. Default: todas las pendientes",
  )

  parser.add_argument(
    "--country",
    type=str,
    default=None,
    help="Filtro por países separados por coma (ej: 'chile,usa'). Filtra por campo obtenido_de.",
  )

  parser.add_argument(
    "--normalize-only",
    action="store_true",
    help="Solo ejecutar normalización global (requiere extracción completa)",
  )

  return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
  """Valida los argumentos de línea de comandos.

  Args:
      args: Argumentos parseados

  Raises:
      SystemExit: Si la validación falla
  """
  if not os.path.exists(args.data):
    logger.error(f"Archivo no encontrado: {args.data}")
    sys.exit(1)

  if not args.data.endswith(".csv"):
    logger.warning("El archivo no tiene extensión .csv")

  if args.method == "vp-k" and args.k <= 0:
    logger.error("El valor de k debe ser positivo")
    sys.exit(1)


def parse_countries(country_arg: str | None) -> list[str] | None:
  """Parsea el argumento de países.

  Args:
      country_arg: String con países separados por coma

  Returns:
      Lista de países o None
  """
  if not country_arg:
    return None
  return [c.strip().lower() for c in country_arg.split(",") if c.strip()]


def print_summary(state_manager: ExtractionStateManager) -> None:
  """Imprime resumen del estado.

  Args:
      state_manager: Gestor de estado con los resultados
  """
  summary = state_manager.get_summary()

  print("\n" + "=" * 60)
  print("RESUMEN DE EXTRACCIÓN")
  print("=" * 60)
  print(f"Total filas:     {summary['total_rows']}")
  print(f"Procesadas:      {summary['processed']}")
  print(f"Pendientes:      {summary['total_rows'] - summary['processed']}")
  print(f"Normalizado:     {'Sí' if summary['normalized'] else 'No'}")
  print("=" * 60)
  print(f"Archivo estado:  {state_manager.state_file_path}")
  print(f"Archivo NPZ:     {state_manager.npz_file_path}")
  print("=" * 60 + "\n")


def main() -> None:
  """Punto de entrada principal."""
  global _state_manager

  args = parse_args()
  validate_args(args)

  # Parsear filtro de países
  countries = parse_countries(args.country)

  logger.info("Iniciando Extracción de Hidden States KEEN")
  logger.info(f"Archivo datos: {args.data}")
  logger.info(f"Método: {args.method}")
  if args.method == "vp-k":
    logger.info(f"k: {args.k}, modo: {args.vpk_mode}")
  logger.info(f"Idioma: {args.language}")
  logger.info(f"Límite: {args.limit if args.limit else 'Ninguno (todas las pendientes)'}")
  logger.info(f"Filtro países: {countries if countries else 'Ninguno (todos)'}")

  # Inicializar state manager
  state_manager = ExtractionStateManager(
    data_path=args.data,
    method=args.method,
    k=args.k,
    vpk_mode=args.vpk_mode,
    language=args.language,
  )
  _state_manager = state_manager

  # Registrar manejador de señales
  signal.signal(signal.SIGINT, signal_handler)

  state_manager.load_or_create()

  # Modo solo normalización
  if args.normalize_only:
    if not state_manager.is_complete():
      logger.error("No se puede normalizar: extracción incompleta")
      print_summary(state_manager)
      sys.exit(1)

    if state_manager.is_normalized():
      logger.info("Las representaciones ya están normalizadas")
      print_summary(state_manager)
      return

    logger.info("Cargando modelo para normalización...")
    extractor = FalconHiddenStatesExtractor()
    state_manager.apply_global_normalization(extractor)
    print_summary(state_manager)
    return

  # Obtener filas pendientes
  pending_rows = state_manager.get_pending_rows(limit=args.limit, countries=countries)
  total_pending = len(pending_rows)
  all_pending = len(state_manager.get_pending_rows(countries=countries))

  if total_pending == 0:
    if state_manager.is_complete():
      logger.info("Extracción completa.")
      if not state_manager.is_normalized():
        logger.info("Aplicando normalización global...")
        extractor = FalconHiddenStatesExtractor()
        state_manager.apply_global_normalization(extractor)
    else:
      logger.info("No hay filas pendientes con los filtros aplicados")
    print_summary(state_manager)
    return

  if args.limit and all_pending > args.limit:
    logger.info(
      f"Encontradas {all_pending} filas pendientes, procesando {total_pending} (límite: {args.limit})"
    )
  else:
    logger.info(f"Encontradas {total_pending} filas pendientes para procesar")

  # Inicializar extractor
  logger.info("Cargando modelo Falcon3-3B-Instruct...")
  extractor = FalconHiddenStatesExtractor()

  # Obtener template de prompt
  language_config = get_language_config(args.language)
  prompt_template = language_config["prompt_template"]
  logger.info(f"Template de prompt: '{prompt_template}'")

  try:
    # Procesar cada fila pendiente
    for i, row in enumerate(pending_rows, 1):
      entity = row["entidad"]
      prompt = prompt_template.format(entidad=entity)

      logger.info(f"[{i}/{total_pending}] Procesando: {entity}")

      # Extraer representación (sin normalizar)
      if args.method == "hs":
        representation = extractor.method_hs(entity, context=prompt, normalize=False)
        tokens = None
      elif args.method == "vp":
        representation = extractor.method_vp(entity, context=prompt, normalize=False)
        tokens = None
      else:  # vp-k
        representation, tokens = extractor.method_vp_k(
          entity,
          k=args.k,
          context=prompt,
          mode=args.vpk_mode,
          normalize=False,
        )

      # Guardar resultado
      state_manager.save_row_result(
        index=row["index"],
        representation=representation,
        tokens=tokens,
      )

      # Log de progreso
      summary = state_manager.get_summary()
      logger.info(
        f"Progreso: {summary['processed']}/{summary['total_rows']} "
        f"({summary['processed'] / summary['total_rows'] * 100:.1f}%)"
      )

  except KeyboardInterrupt:
    logger.info("\nInterrumpido por el usuario. Guardando progreso...")
    state_manager.finalize()
    print_summary(state_manager)
    sys.exit(0)

  except Exception as e:
    logger.error(f"Error durante el procesamiento: {e}")
    state_manager.finalize()
    print_summary(state_manager)
    raise

  # Finalizar (flush buffer pendiente)
  state_manager.finalize()

  # Auto-normalizar si se completó
  if state_manager.is_complete() and not state_manager.is_normalized():
    logger.info("Extracción completa. Aplicando normalización global...")
    state_manager.apply_global_normalization(extractor)

  print_summary(state_manager)
  logger.info("Extracción completada exitosamente!")


if __name__ == "__main__":
  main()
