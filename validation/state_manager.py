import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import REQUIRED_COLUMNS


class StateManager:
  """Manages the state file for resumable validation execution."""

  def __init__(self, data_path: str, model: str, language: str):
    """Initialize the state manager.

    Args:
        data_path: Path to the CSV data file
        model: Name of the LLM model being used
        language: Language code for prompts
    """
    self.data_path = os.path.abspath(data_path)
    self.model = model
    self.language = language
    self.state_file_path = self._get_state_file_path()
    self.state: dict[str, Any] = {}

  def _get_state_file_path(self) -> str:
    """Generate the state file path based on the data file path hash.

    Returns:
        Path to the state file
    """
    path_hash = hashlib.md5(self.data_path.encode()).hexdigest()[:12]
    state_dir = Path(__file__).parent
    return str(state_dir / f"state_{path_hash}.json")

  def load_or_create(self) -> dict[str, Any]:
    """Load existing state file or create a new one from CSV.

    Returns:
        The current state dictionary
    """
    if os.path.exists(self.state_file_path):
      self._load_state()
      self._sync_with_csv()
    else:
      self._create_new_state()

    return self.state

  def _load_state(self) -> None:
    """Load state from existing JSON file."""
    with open(self.state_file_path, "r", encoding="utf-8") as f:
      self.state = json.load(f)

    # Verify metadata matches current execution
    metadata = self.state.get("metadata", {})
    if metadata.get("data_path") != self.data_path:
      raise ValueError(
        f"State file data_path mismatch. "
        f"Expected: {self.data_path}, Found: {metadata.get('data_path')}"
      )

  def _create_new_state(self) -> None:
    """Create a new state from the CSV file."""
    df = self._load_csv()

    now = datetime.now().isoformat()
    self.state = {
      "metadata": {
        "data_path": self.data_path,
        "model": self.model,
        "language": self.language,
        "created_at": now,
        "last_updated": now,
      },
      "summary": {
        "total_rows": len(df),
        "processed": 0,
        "correct": 0,
        "incorrect": 0,
      },
      "rows": [],
    }

    # Add each row from CSV
    for i, (idx, row) in enumerate(df.iterrows()):
      self.state["rows"].append(self._row_to_state_entry(i, row))

    self._save_state()

  def _load_csv(self) -> pd.DataFrame:
    """Load and validate the CSV file.

    Returns:
        DataFrame with the CSV data

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(self.data_path):
      raise FileNotFoundError(f"Data file not found: {self.data_path}")

    df = pd.read_csv(self.data_path, encoding="utf-8")

    # Validate required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
      raise ValueError(f"Missing required columns: {missing_columns}")

    return df

  def _row_to_state_entry(self, index: int, row: pd.Series) -> dict[str, Any]:
    """Convert a CSV row to a state entry.

    Args:
        index: Row index
        row: Pandas Series representing the row

    Returns:
        State entry dictionary
    """
    return {
      "index": index,
      "entidad": str(row["entidad"]),
      "relacion": str(row["relacion"]),
      "objetos": self._parse_list_field(row["objetos"]),
      "pregunta": str(row["pregunta"]),
      "respuestas": self._parse_list_field(row["respuestas"]),
      "obtenido_de": self._parse_list_field(row["obtenido_de"]),
      "respuestas_aliases": self._parse_aliases_field(row["respuestas_aliases"]),
      "llm_response": None,
      "is_correct": None,
      "similarity_score": None,
      "matched_answer": None,
      "processed_at": None,
    }

  def _parse_list_field(self, value: Any) -> list[str]:
    """Parse a list field from CSV (stored as string representation).

    Args:
        value: The field value (e.g., "['atletismo']")

    Returns:
        Parsed list of strings
    """
    if pd.isna(value):
      return []
    if isinstance(value, list):
      return [str(item) for item in value]

    value_str = str(value).strip()
    if not value_str or value_str == "[]":
      return []

    try:
      import ast

      parsed = ast.literal_eval(value_str)
      if isinstance(parsed, list):
        return [str(item) for item in parsed]
      return [str(parsed)]
    except (ValueError, SyntaxError):
      return [value_str]

  def _parse_aliases_field(self, value: Any) -> list[list[str]]:
    """Parse the respuestas_aliases field (list of lists).

    Args:
        value: The field value (e.g., "[['atletismo', 'pista y campo']]")

    Returns:
        Parsed list of lists of strings
    """
    if pd.isna(value):
      return []
    if isinstance(value, list):
      return [[str(item) for item in sublist] for sublist in value if isinstance(sublist, list)]

    value_str = str(value).strip()
    if not value_str or value_str == "[]":
      return []

    try:
      import ast

      parsed = ast.literal_eval(value_str)
      if isinstance(parsed, list):
        result = []
        for item in parsed:
          if isinstance(item, list):
            result.append([str(x) for x in item])
          else:
            result.append([str(item)])
        return result
      return [[str(parsed)]]
    except (ValueError, SyntaxError):
      return [[value_str]]

  def _sync_with_csv(self) -> None:
    """Sync the state with the current CSV file (in case rows were added)."""
    df = self._load_csv()

    current_row_count = len(self.state["rows"])
    csv_row_count = len(df)

    if csv_row_count > current_row_count:
      # Add new rows from CSV
      for idx in range(current_row_count, csv_row_count):
        self.state["rows"].append(self._row_to_state_entry(idx, df.iloc[idx]))

      self.state["summary"]["total_rows"] = csv_row_count
      self._save_state()

  def get_pending_rows(
    self,
    limit: Optional[int] = None,
    countries: Optional[list[str]] = None,
  ) -> list[dict[str, Any]]:
    """Get rows that haven't been processed yet.

    Args:
        limit: Maximum number of rows to return (None for all)
        countries: List of country names to filter by (None for all).
                   Matches against obtenido_de field (case-insensitive).

    Returns:
        List of state entries without llm_response
    """
    pending = [row for row in self.state["rows"] if row["llm_response"] is None]

    # Apply country filter
    if countries:
      pending = [
        row
        for row in pending
        if any(country in [x.lower() for x in row["obtenido_de"]] for country in countries)
      ]

    # Apply limit
    if limit is not None:
      return pending[:limit]
    return pending

  def save_row_result(
    self,
    index: int,
    llm_response: str,
    is_correct: bool,
    similarity_score: Optional[float],
    matched_answer: Optional[str],
  ) -> None:
    """Save the result for a specific row and update summary.

    Args:
        index: Row index
        llm_response: The response from the LLM
        is_correct: Whether the response was correct
        similarity_score: Similarity score (if applicable)
        matched_answer: The answer that was matched (if any)
    """
    row = self.state["rows"][index]
    row["llm_response"] = llm_response
    row["is_correct"] = is_correct
    row["similarity_score"] = similarity_score
    row["matched_answer"] = matched_answer
    row["processed_at"] = datetime.now().isoformat()

    # Update summary
    self.state["summary"]["processed"] += 1
    if is_correct:
      self.state["summary"]["correct"] += 1
    else:
      self.state["summary"]["incorrect"] += 1

    self.state["metadata"]["last_updated"] = datetime.now().isoformat()

    self._save_state()

  def _save_state(self) -> None:
    """Save the current state to the JSON file."""
    # Write to temp file first, then rename (atomic operation)
    temp_path = self.state_file_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
      json.dump(self.state, f, ensure_ascii=False, indent=2)

    os.replace(temp_path, self.state_file_path)

  def get_summary(self) -> dict[str, Any]:
    """Get the current summary statistics.

    Returns:
        Summary dictionary with total_rows, processed, correct, incorrect
    """
    return self.state["summary"].copy()

  def get_accuracy(self) -> float:
    """Calculate the current accuracy.

    Returns:
        Accuracy as a float between 0 and 1
    """
    processed = self.state["summary"]["processed"]
    if processed == 0:
      return 0.0
    return self.state["summary"]["correct"] / processed
