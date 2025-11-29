import ast
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import REQUIRED_COLUMNS, STATE_DIR, BUFFER_SIZE


class ExtractionStateManager:
  """Gestiona el estado de extracción para ejecuciones resumibles."""

  def __init__(
    self,
    data_path: str,
    method: str,
    k: int = 50,
    vpk_mode: str = "bidirectional",
    language: str = "es",
  ):
    """Inicializa el gestor de estado.

    Args:
        data_path: Ruta al archivo CSV con entidades
        method: Método de extracción ('hs', 'vp', 'vp-k')
        k: Valor k para VP-k
        vpk_mode: Modo de VP-k ('positive', 'bidirectional', 'abs')
        language: Código de idioma ('es' o 'en')
    """
    self.data_path = os.path.abspath(data_path)
    self.method = method
    self.k = k
    self.vpk_mode = vpk_mode
    self.language = language

    # Generar rutas de archivos de estado
    self.state_dir = Path(data_path).parent / STATE_DIR
    self.state_id = self._generate_state_id()
    self.state_file_path = str(self.state_dir / f"{self.state_id}.json")
    self.npz_file_path = str(self.state_dir / f"{self.state_id}.npz")

    # Estado y buffer
    self.state: dict[str, Any] = {}
    self.buffer: list[tuple[int, np.ndarray, Optional[list]]] = []
    self.buffer_size = BUFFER_SIZE

  def _generate_state_id(self) -> str:
    """Genera un ID único basado en los parámetros de ejecución.

    Returns:
        ID único para los archivos de estado
    """
    # Hash del path absoluto del CSV
    path_hash = hashlib.md5(self.data_path.encode()).hexdigest()[:12]

    # Construir nombre según método
    if self.method in ("vp-k", "vpk"):
      return f"state_{path_hash}_vpk_k{self.k}"
    return f"state_{path_hash}_{self.method}"

  def load_or_create(self) -> dict[str, Any]:
    """Carga estado existente o crea uno nuevo desde CSV.

    Returns:
        El diccionario de estado actual
    """
    # Asegurar que existe el directorio de estados
    self.state_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(self.state_file_path):
      self._load_state()
      self._sync_with_csv()
    else:
      self._create_new_state()

    return self.state

  def _load_state(self) -> None:
    """Carga estado desde archivo JSON existente."""
    with open(self.state_file_path, "r", encoding="utf-8") as f:
      self.state = json.load(f)

    # Verificar que los metadatos coinciden
    metadata = self.state.get("metadata", {})
    if metadata.get("data_path") != self.data_path:
      raise ValueError(
        f"El archivo de estado corresponde a otro CSV. "
        f"Esperado: {self.data_path}, Encontrado: {metadata.get('data_path')}"
      )
    if metadata.get("method") != self.method:
      raise ValueError(
        f"El archivo de estado corresponde a otro método. "
        f"Esperado: {self.method}, Encontrado: {metadata.get('method')}"
      )

  def _create_new_state(self) -> None:
    """Crea un nuevo estado desde el archivo CSV."""
    df = self._load_csv()

    now = datetime.now().isoformat()
    self.state = {
      "metadata": {
        "data_path": self.data_path,
        "data_hash": hashlib.md5(self.data_path.encode()).hexdigest(),
        "method": self.method,
        "k": self.k,
        "vpk_mode": self.vpk_mode,
        "language": self.language,
        "created_at": now,
        "last_updated": now,
        "normalized_at": None,
      },
      "summary": {
        "total_rows": len(df),
        "processed": 0,
        "normalized": False,
      },
      "rows": [],
      "npz_path": self.npz_file_path,
    }

    # Agregar cada fila del CSV
    for i, (_, row) in enumerate(df.iterrows()):
      self.state["rows"].append(self._row_to_state_entry(i, row))

    self._save_state()

  def _load_csv(self) -> pd.DataFrame:
    """Carga y valida el archivo CSV.

    Returns:
        DataFrame con los datos del CSV

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si faltan columnas requeridas
    """
    if not os.path.exists(self.data_path):
      raise FileNotFoundError(f"Archivo no encontrado: {self.data_path}")

    df = pd.read_csv(self.data_path, encoding="utf-8")

    # Validar columnas requeridas
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
      raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    return df

  def _row_to_state_entry(self, index: int, row: pd.Series) -> dict[str, Any]:
    """Convierte una fila del CSV a entrada de estado.

    Args:
        index: Índice de la fila
        row: Serie de pandas con los datos

    Returns:
        Diccionario de entrada de estado
    """
    return {
      "index": index,
      "entidad": str(row["entidad"]),
      "relacion": str(row["relacion"]),
      "obtenido_de": self._parse_list_field(row["obtenido_de"]),
      "processed": False,
      "processed_at": None,
      "vector_index": None,
    }

  def _parse_list_field(self, value: Any) -> list[str]:
    """Parsea un campo de lista desde CSV (almacenado como string).

    Args:
        value: Valor del campo (ej: "['chile', 'usa']")

    Returns:
        Lista de strings parseada
    """
    if pd.isna(value):
      return []
    if isinstance(value, list):
      return [str(item) for item in value]

    value_str = str(value).strip()
    if not value_str or value_str == "[]":
      return []

    try:
      parsed = ast.literal_eval(value_str)
      if isinstance(parsed, list):
        return [str(item) for item in parsed]
      return [str(parsed)]
    except (ValueError, SyntaxError):
      return [value_str]

  def _sync_with_csv(self) -> None:
    """Sincroniza el estado con el CSV actual (si se agregaron filas)."""
    df = self._load_csv()

    current_row_count = len(self.state["rows"])
    csv_row_count = len(df)

    if csv_row_count > current_row_count:
      # Agregar nuevas filas
      for idx in range(current_row_count, csv_row_count):
        self.state["rows"].append(self._row_to_state_entry(idx, df.iloc[idx]))

      self.state["summary"]["total_rows"] = csv_row_count
      self._save_state()

  def get_pending_rows(
    self,
    limit: Optional[int] = None,
    countries: Optional[list[str]] = None,
  ) -> list[dict[str, Any]]:
    """Obtiene filas pendientes de procesar.

    Args:
        limit: Máximo de filas a retornar (None para todas)
        countries: Lista de países para filtrar (None para todos)

    Returns:
        Lista de entradas de estado pendientes
    """
    pending = [row for row in self.state["rows"] if not row["processed"]]

    # Aplicar filtro de países
    if countries:
      countries_lower = [c.lower() for c in countries]
      pending = [
        row
        for row in pending
        if any(country in [x.lower() for x in row["obtenido_de"]] for country in countries_lower)
      ]

    # Aplicar límite
    if limit is not None:
      return pending[:limit]
    return pending

  def save_row_result(
    self,
    index: int,
    representation: np.ndarray,
    tokens: Optional[list[str]] = None,
  ) -> None:
    """Guarda el resultado de una fila y actualiza el estado.

    Args:
        index: Índice de la fila en el CSV
        representation: Vector de representación extraído
        tokens: Tokens para VP-k (opcional)
    """
    # Agregar al buffer
    self.buffer.append((index, representation, tokens))

    # Flush si el buffer está lleno
    if len(self.buffer) >= self.buffer_size:
      self._flush_buffer()

  def _flush_buffer(self) -> None:
    """Escribe el buffer al archivo NPZ y actualiza el estado JSON."""
    if not self.buffer:
      return

    # Cargar datos existentes del NPZ si existe
    if os.path.exists(self.npz_file_path):
      existing = np.load(self.npz_file_path, allow_pickle=True)
      existing_raw = existing["representations_raw"]
      existing_indices = existing["row_indices"].tolist()
      existing_tokens = (
        existing["tokens"].tolist()
        if "tokens" in existing.files and existing["tokens"].size > 0
        else []
      )
    else:
      existing_raw = None
      existing_indices = []
      existing_tokens = []

    # Preparar nuevos datos
    new_representations = []
    new_indices = []
    new_tokens = []

    for index, representation, tokens in self.buffer:
      new_representations.append(representation)
      new_indices.append(index)
      if tokens:
        new_tokens.append(tokens)

      # Actualizar estado de la fila
      row = self.state["rows"][index]
      row["processed"] = True
      row["processed_at"] = datetime.now().isoformat()
      row["vector_index"] = len(existing_indices) + len(new_indices) - 1

    # Combinar con datos existentes
    new_raw_array = np.array(new_representations)
    if existing_raw is not None:
      combined_raw = np.vstack([existing_raw, new_raw_array])
    else:
      combined_raw = new_raw_array

    combined_indices = existing_indices + new_indices
    combined_tokens = existing_tokens + new_tokens if new_tokens else existing_tokens

    # Guardar NPZ
    np.savez(
      self.npz_file_path,
      representations_raw=combined_raw,
      representations_normalized=np.array([]),  # Se llena al normalizar
      row_indices=np.array(combined_indices),
      tokens=np.array(combined_tokens, dtype=object) if combined_tokens else np.array([]),
    )

    # Actualizar summary
    self.state["summary"]["processed"] += len(self.buffer)
    self.state["metadata"]["last_updated"] = datetime.now().isoformat()

    # Guardar estado JSON
    self._save_state()

    # Limpiar buffer
    self.buffer = []

  def _save_state(self) -> None:
    """Guarda el estado actual al archivo JSON de forma atómica."""
    temp_path = self.state_file_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
      json.dump(self.state, f, ensure_ascii=False, indent=2)

    os.replace(temp_path, self.state_file_path)

  def is_complete(self) -> bool:
    """Verifica si todas las filas han sido procesadas.

    Returns:
        True si todas las filas están procesadas
    """
    return self.state["summary"]["processed"] == self.state["summary"]["total_rows"]

  def is_normalized(self) -> bool:
    """Verifica si las representaciones ya fueron normalizadas.

    Returns:
        True si ya se aplicó normalización global
    """
    return self.state["summary"]["normalized"]

  def apply_global_normalization(self, extractor) -> None:
    """Aplica normalización global a todas las representaciones.

    Args:
        extractor: Instancia de FalconHiddenStatesExtractor con fit_normalizer()
    """
    if not self.is_complete():
      raise ValueError("No se puede normalizar: extracción incompleta")

    if self.is_normalized():
      print("Las representaciones ya están normalizadas")
      return

    # Flush buffer pendiente
    self._flush_buffer()

    # Cargar representaciones RAW
    npz_data = np.load(self.npz_file_path, allow_pickle=True)
    raw_representations = npz_data["representations_raw"]

    # Ajustar normalizador global
    method_key = "vpk" if self.method in ("vp-k", "vpk") else self.method
    extractor.fit_normalizer(method_key, raw_representations)

    # Transformar
    normalized = extractor.transform_with_normalizer(method_key, raw_representations)

    # Guardar NPZ actualizado
    np.savez(
      self.npz_file_path,
      representations_raw=raw_representations,
      representations_normalized=normalized,
      row_indices=npz_data["row_indices"],
      tokens=npz_data["tokens"] if "tokens" in npz_data.files else np.array([]),
    )

    # Actualizar metadata
    self.state["metadata"]["normalized_at"] = datetime.now().isoformat()
    self.state["summary"]["normalized"] = True
    self._save_state()

    print(f"Normalización global aplicada a {len(raw_representations)} representaciones")

  def get_representations(self, normalized: bool = True) -> np.ndarray:
    """Obtiene las representaciones extraídas.

    Args:
        normalized: Si True, retorna normalizadas (si disponibles)

    Returns:
        Array numpy con las representaciones
    """
    if not os.path.exists(self.npz_file_path):
      raise FileNotFoundError("No hay representaciones extraídas aún")

    npz_data = np.load(self.npz_file_path, allow_pickle=True)

    if normalized and self.is_normalized():
      return npz_data["representations_normalized"]
    return npz_data["representations_raw"]

  def get_summary(self) -> dict[str, Any]:
    """Obtiene estadísticas de progreso.

    Returns:
        Diccionario con total_rows, processed, normalized
    """
    return self.state["summary"].copy()

  def finalize(self) -> None:
    """Finaliza la sesión, asegurando que todo el buffer se guarde."""
    self._flush_buffer()
