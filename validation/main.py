import argparse
import logging
import os
import sys

from config import DEFAULT_MODEL, LANGUAGE_CONFIGS
from evaluator import Evaluator
from state_manager import StateManager

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
  """Parse command line arguments.

  Returns:
      Parsed arguments namespace
  """
  parser = argparse.ArgumentParser(
    description="Validation - Simplified LLM validation system",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    python main.py --data ./questions.csv
    python main.py --data ./questions.csv --model "Qwen/Qwen3-4B-Instruct"
    python main.py --data ./questions.csv --language en
    python main.py --data ./questions.csv --limit 100
    python main.py --data ./questions.csv --country chile,usa

The program automatically saves progress after each question.
If interrupted, simply run again with the same arguments to resume.
Use --limit to process a fixed number of rows per execution.
Use --country to filter rows by the obtenido_de field.
        """,
  )

  parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Path to the CSV file with questions",
  )

  parser.add_argument(
    "--model",
    type=str,
    default=DEFAULT_MODEL,
    help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
  )

  parser.add_argument(
    "--language",
    type=str,
    default="es",
    choices=list(LANGUAGE_CONFIGS.keys()),
    help="Language for prompts (default: es)",
  )

  parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of rows to process in this run (default: all pending)",
  )

  parser.add_argument(
    "--country",
    type=str,
    default=None,
    help="Comma-separated country filter (e.g., 'chile,usa'). Filters rows by obtenido_de field.",
  )

  return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
  """Validate command line arguments.

  Args:
      args: Parsed arguments

  Raises:
      SystemExit: If validation fails
  """
  if not os.path.exists(args.data):
    logger.error(f"Data file not found: {args.data}")
    sys.exit(1)

  if not args.data.endswith(".csv"):
    logger.warning("Data file does not have .csv extension")


def print_summary(state_manager: StateManager) -> None:
  """Print the final summary.

  Args:
      state_manager: The state manager with results
  """
  summary = state_manager.get_summary()
  accuracy = state_manager.get_accuracy()

  print("\n" + "=" * 60)
  print("VALIDATION SUMMARY")
  print("=" * 60)
  print(f"Total rows:      {summary['total_rows']}")
  print(f"Processed:       {summary['processed']}")
  print(f"Correct:         {summary['correct']}")
  print(f"Incorrect:       {summary['incorrect']}")
  print(f"Accuracy:        {accuracy:.2%}")
  print("=" * 60)
  print(f"State file:      {state_manager.state_file_path}")
  print("=" * 60 + "\n")


def main() -> None:
  """Main entry point."""
  args = parse_args()
  validate_args(args)

  # Parse country filter
  countries = None
  if args.country:
    countries = [c.strip().lower() for c in args.country.split(",") if c.strip()]

  logger.info("Starting Validation")
  logger.info(f"Data file: {args.data}")
  logger.info(f"Model: {args.model}")
  logger.info(f"Language: {args.language}")
  logger.info(f"Limit: {args.limit if args.limit else 'None (all pending)'}")
  logger.info(f"Country filter: {countries if countries else 'None (all countries)'}")

  # Initialize state manager
  state_manager = StateManager(args.data, args.model, args.language)
  state_manager.load_or_create()

  # Get pending rows (with optional country filter and limit)
  pending_rows = state_manager.get_pending_rows(limit=args.limit, countries=countries)
  total_pending = len(pending_rows)
  all_pending = len(state_manager.get_pending_rows(countries=countries))

  if total_pending == 0:
    logger.info("All rows have been processed!")
    print_summary(state_manager)
    return

  if args.limit and all_pending > args.limit:
    logger.info(
      f"Found {all_pending} pending rows, processing {total_pending} (limit: {args.limit})"
    )
  else:
    logger.info(f"Found {total_pending} pending rows to process")

  # Initialize evaluator
  evaluator = Evaluator(args.model, args.language)

  try:
    # Load model
    evaluator.load_model()

    # Process each pending row
    for i, row in enumerate(pending_rows, 1):
      index = row["index"]
      question = row["pregunta"]

      logger.info(f"Processing row {i}/{total_pending} (index {index})")

      # Generate response
      response = evaluator.generate_response(question)
      logger.info(f"LLM response: {response}")

      # Evaluate answer
      is_correct, similarity_score, matched_answer = evaluator.evaluate_answer(
        response,
        row["respuestas"],
        row["respuestas_aliases"],
      )

      # Log result
      status = "CORRECT" if is_correct else "INCORRECT"
      logger.info(f"Result: {status} (similarity: {similarity_score})")

      # Save result immediately
      state_manager.save_row_result(
        index=index,
        llm_response=response,
        is_correct=is_correct,
        similarity_score=similarity_score,
        matched_answer=matched_answer,
      )

      # Log progress
      current_accuracy = state_manager.get_accuracy()
      logger.info(f"Progress: {i}/{total_pending} (Accuracy: {current_accuracy:.2%})")

  except KeyboardInterrupt:
    logger.info("\nInterrupted by user. Progress has been saved.")
    print_summary(state_manager)
    sys.exit(0)

  except Exception as e:
    logger.error(f"Error during processing: {e}")
    print_summary(state_manager)
    raise

  finally:
    # Unload model
    evaluator.unload_model()

  # Print final summary
  print_summary(state_manager)
  logger.info("Validation completed successfully!")


if __name__ == "__main__":
  main()
