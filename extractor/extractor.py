import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")


@dataclass
class FalconConfig:
  model_name: str = "tiiuae/Falcon3-3B-Instruct"
  num_layers: int = 22
  hidden_size: int = 3072
  vocab_size: int = 131072
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  dtype: torch.dtype = torch.bfloat16


class GlobalNormalizer:
  def __init__(self):
    self.scaler = MinMaxScaler()
    self.fitted = False

  def fit(self, all_representations: np.ndarray) -> None:
    if len(all_representations.shape) == 1:
      all_representations = all_representations.reshape(-1, 1)
    self.scaler.fit(all_representations)
    self.fitted = True
    print(f"✅ Normalizador ajustado con {len(all_representations)} muestras")
    print(f"Rango global: [{self.scaler.data_min_.min():.4f}, {self.scaler.data_max_.max():.4f}]")

  def transform(self, representation: np.ndarray) -> np.ndarray:
    if not self.fitted:
      raise ValueError("El normalizador debe ser ajustado primero con fit()")
    original_shape = representation.shape
    is_single_vector = len(original_shape) == 1
    if is_single_vector:
      representation = representation.reshape(1, -1)
    transformed = self.scaler.transform(representation)
    return transformed.flatten() if is_single_vector else transformed

  def fit_transform(self, all_representations: np.ndarray) -> np.ndarray:
    self.fit(all_representations)
    return self.scaler.transform(all_representations)


class FalconHiddenStatesExtractor:
  def __init__(self, config: Optional[FalconConfig] = None):
    self.config = config or FalconConfig()
    print(f"🚀 Inicializando Falcon3-3B-Instruct en {self.config.device}...")
    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
      self.config.model_name,
      torch_dtype=self.config.dtype,
      device_map=self.config.device,
      low_cpu_mem_usage=True,
    )
    self.model.eval()
    self.selected_layers = self._calculate_target_layers()
    print(f"✅ Modelo cargado. Capas seleccionadas: {self.selected_layers}")
    self.normalizers: Dict[str, Optional[GlobalNormalizer]] = {"hs": None, "vp": None, "vpk": None}

  def _calculate_target_layers(self) -> List[int]:
    L = self.config.num_layers
    base_layer = int(3 * L / 4)
    layers = [base_layer - 1, base_layer, base_layer + 1]
    return [max(0, min(L - 1, layer)) for layer in layers]

  def _tokenize_entity(self, entity: str, context: Optional[str] = None) -> Dict:
    text = (
      context if context and entity in context else (context + " " + entity if context else entity)
    )
    encoded = self.tokenizer(
      text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    encoded = {k: v.to(self.config.device) for k, v in encoded.items()}
    entity_tokens = self.tokenizer.encode(entity, add_special_tokens=False)
    input_ids = encoded["input_ids"][0].tolist()
    entity_end_pos = self._find_last_token_position(input_ids, entity_tokens, entity)
    encoded["entity_end_pos"] = entity_end_pos
    return encoded

  def _find_last_token_position(
    self, input_ids: List[int], entity_tokens: List[int], entity: str = ""
  ) -> int:
    if not entity_tokens:
      warnings.warn(f"Entidad '{entity}' no generó tokens. Usando último token.")
      return len(input_ids) - 1
    for i in range(len(input_ids) - len(entity_tokens), -1, -1):
      if input_ids[i : i + len(entity_tokens)] == entity_tokens:
        return i + len(entity_tokens) - 1
    if len(entity_tokens) > 1:
      for i in range(len(input_ids) - len(entity_tokens) + 1, -1, -1):
        if input_ids[i : i + len(entity_tokens) - 1] == entity_tokens[1:]:
          warnings.warn(f"Coincidencia parcial para entidad '{entity}' (sin primer token)")
          return i + len(entity_tokens) - 2
    warnings.warn(f"⚠️  Entidad '{entity}' no encontrada en tokens. Usando último token.")
    return len(input_ids) - 1

  def fit_normalizer(self, method: str, all_representations: np.ndarray) -> None:
    method = method.lower()
    if method not in self.normalizers:
      raise ValueError(f"Método '{method}' no válido. Use 'hs', 'vp', o 'vpk'")
    normalizer = GlobalNormalizer()
    normalizer.fit(all_representations)
    self.normalizers[method] = normalizer
    print(f"✅ Normalizador global ajustado para método {method.upper()}")

  def transform_with_normalizer(self, method: str, representation: np.ndarray) -> np.ndarray:
    method = method.lower()
    if method not in self.normalizers:
      raise ValueError(f"Método '{method}' no válido. Use 'hs', 'vp', o 'vpk'")
    normalizer = self.normalizers[method]
    if normalizer is None:
      raise ValueError(f"Normalizador para '{method}' no ajustado. Llame fit_normalizer() primero.")
    return normalizer.transform(representation)

  @torch.no_grad()
  def extract_hidden_states(
    self, entity: str, context: Optional[str] = None
  ) -> Dict[str, torch.Tensor]:
    inputs = self._tokenize_entity(entity, context)
    entity_pos = inputs["entity_end_pos"]
    outputs = self.model(
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      output_hidden_states=True,
      return_dict=True,
    )
    all_hidden_states = outputs.hidden_states
    selected_hidden_states = {}
    for layer_idx in self.selected_layers:
      layer_output = all_hidden_states[layer_idx + 1]
      entity_repr = layer_output[0, entity_pos, :]
      selected_hidden_states[f"layer_{layer_idx}"] = entity_repr.cpu()
    return selected_hidden_states

  def method_hs(
    self, entity: str, context: Optional[str] = None, normalize: bool = True
  ) -> np.ndarray:
    hidden_states = self.extract_hidden_states(entity, context)
    states_array = np.stack(
      [hidden_states[key].detach().float().numpy() for key in sorted(hidden_states.keys())]
    )
    if normalize:
      normalized_states = np.array(
        [MinMaxScaler().fit_transform(layer.reshape(-1, 1)).flatten() for layer in states_array]
      )
    else:
      normalized_states = states_array
    return normalized_states.mean(axis=0)

  def method_vp(
    self, entity: str, context: Optional[str] = None, normalize: bool = True
  ) -> np.ndarray:
    hidden_states = self.extract_hidden_states(entity, context)
    lm_head_weight = self.model.lm_head.weight
    ln_f = self.model.model.norm
    vocab_projections = []
    for layer_key in sorted(hidden_states.keys()):
      h = hidden_states[layer_key].to(self.config.device)
      h_norm = ln_f(h.unsqueeze(0)).squeeze(0)
      vocab_proj = torch.matmul(lm_head_weight, h_norm)
      vocab_projections.append(vocab_proj.detach().float().cpu().numpy())
    projections_array = np.stack(vocab_projections)
    if normalize:
      normalized_projections = np.array(
        [
          MinMaxScaler().fit_transform(layer.reshape(-1, 1)).flatten()
          for layer in projections_array
        ]
      )
    else:
      normalized_projections = projections_array
    return normalized_projections.mean(axis=0)

  def method_vp_k(
    self,
    entity: str,
    k: int = 50,
    context: Optional[str] = None,
    mode: str = "bidirectional",
    normalize: bool = True,
    use_abs: bool = False,
  ) -> Tuple[np.ndarray, List[str]]:
    if use_abs:
      warnings.warn("use_abs está deprecado. Use mode='abs' en su lugar.", DeprecationWarning)
      mode = "abs"
    if mode not in ["positive", "bidirectional", "abs"]:
      raise ValueError(f"mode debe ser 'positive', 'bidirectional', o 'abs'. Recibido: {mode}")
    hidden_states = self.extract_hidden_states(entity, context)
    lm_head_weight = self.model.lm_head.weight
    ln_f = self.model.model.norm
    top_k_features = []
    top_k_tokens = []
    for layer_key in sorted(hidden_states.keys()):
      h = hidden_states[layer_key].to(self.config.device)
      h_norm = ln_f(h.unsqueeze(0)).squeeze(0)
      vocab_proj = torch.matmul(lm_head_weight, h_norm)
      if mode == "positive":
        top_k_values, top_k_indices = torch.topk(vocab_proj, k=k, largest=True)
        layer_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_k_indices]
      elif mode == "bidirectional":
        k_half = k // 2
        top_pos_values, top_pos_indices = torch.topk(vocab_proj, k=k_half, largest=True)
        top_neg_values, top_neg_indices = torch.topk(vocab_proj, k=k_half, largest=False)
        top_k_values = torch.cat([top_pos_values, top_neg_values])
        pos_tokens = [f"+{self.tokenizer.decode([idx.item()])}" for idx in top_pos_indices]
        neg_tokens = [f"-{self.tokenizer.decode([idx.item()])}" for idx in top_neg_indices]
        layer_tokens = pos_tokens + neg_tokens
      else:
        warnings.warn(
          "Modo 'abs' no es recomendado. Use 'bidirectional' para mayor interpretabilidad."
        )
        top_k_values, top_k_indices = torch.topk(torch.abs(vocab_proj), k=k, largest=True)
        layer_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_k_indices]
      top_k_features.append(top_k_values.detach().float().cpu().numpy())
      top_k_tokens.extend(layer_tokens)
    features_array = np.concatenate(top_k_features)
    if normalize:
      z = MinMaxScaler().fit_transform(features_array.reshape(-1, 1)).flatten()
    else:
      z = features_array
    return z, top_k_tokens

  def get_entity_representation(
    self,
    entity: str,
    method: str = "hs",
    context: Optional[str] = None,
    k: int = 50,
    normalize: bool = True,
    vpk_mode: str = "bidirectional",
  ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    method = method.lower()
    if method == "hs":
      return self.method_hs(entity, context, normalize=normalize)
    elif method == "vp":
      return self.method_vp(entity, context, normalize=normalize)
    elif method in ["vp-k", "vpk"]:
      return self.method_vp_k(entity, k, context, mode=vpk_mode, normalize=normalize)
    else:
      raise ValueError(f"Método no válido: {method}. Use 'hs', 'vp', o 'vp-k'")


def extract_all_raw_representations(
  extractor: FalconHiddenStatesExtractor,
  entities: List[str],
  method: Literal["hs", "vp", "vp-k", "vpk"] = "hs",
  contexts: Optional[List[Optional[str]]] = None,
  k: int = 50,
  vpk_mode: str = "bidirectional",
) -> np.ndarray:
  contexts_list: List[Optional[str]] = contexts if contexts is not None else [None] * len(entities)
  if len(contexts_list) != len(entities):
    raise ValueError("contexts debe tener la misma longitud que entities")
  print(f"\n🔄 Extrayendo representaciones RAW de {len(entities)} entidades...")
  print(f"Método: {method.upper()}")
  all_representations = []
  for i, (entity, context) in enumerate(zip(entities, contexts_list)):
    print(f"[{i + 1}/{len(entities)}] {entity}")
    if method in ("vp-k", "vpk"):
      result = extractor.method_vp_k(entity, k, context, mode=vpk_mode, normalize=False)
      repr_vec = result[0]
    else:
      repr_vec = (
        extractor.method_hs(entity, context, normalize=False)
        if method == "hs"
        else extractor.method_vp(entity, context, normalize=False)
      )
    all_representations.append(repr_vec)
  all_representations_array = np.array(all_representations)
  print(f"\n✅ Representaciones RAW extraídas: {all_representations_array.shape}")
  return all_representations_array


def fit_and_transform_pipeline(
  extractor: FalconHiddenStatesExtractor,
  entities: List[str],
  method: Literal["hs", "vp", "vp-k", "vpk"] = "hs",
  contexts: Optional[List[Optional[str]]] = None,
  k: int = 50,
  vpk_mode: str = "bidirectional",
  save_normalizer: bool = False,
  normalizer_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[List[List[str]]]]:
  print("\n" + "=" * 80)
  print("PIPELINE KEEN: Extracción con Normalización Global")
  print("=" * 80)
  print("\n📊 PASO 1: Extrayendo representaciones RAW...")
  all_raw = extract_all_raw_representations(extractor, entities, method, contexts, k, vpk_mode)
  print("\n📊 PASO 2: Ajustando normalizador global...")
  method_key = "vpk" if method in ("vp-k", "vpk") else method
  extractor.fit_normalizer(method_key, all_raw)
  print("\n📊 PASO 3: Transformando representaciones con estadísticas globales...")
  all_normalized = extractor.normalizers[method_key].transform(all_raw)  # type: ignore
  tokens_list = None
  if method in ("vp-k", "vpk"):
    print("\n📊 Extrayendo tokens de VP-k para interpretabilidad...")
    tokens_list = []
    contexts_list = contexts or [None] * len(entities)
    for entity, context in zip(entities, contexts_list):
      _, tokens = extractor.method_vp_k(entity, k, context, mode=vpk_mode, normalize=False)
      tokens_list.append(tokens)
  if save_normalizer and normalizer_path:
    import pickle

    with open(normalizer_path, "wb") as f:
      pickle.dump(extractor.normalizers[method_key], f)
    print(f"\n💾 Normalizador guardado en: {normalizer_path}")
  print("\n" + "=" * 80)
  print("✅ PIPELINE COMPLETADO")
  print(f"Representaciones finales: {all_normalized.shape}")
  print(f"Rango global aplicado: [{all_normalized.min():.4f}, {all_normalized.max():.4f}]")
  print("=" * 80 + "\n")
  return all_normalized, tokens_list


if __name__ == "__main__":
  extractor = FalconHiddenStatesExtractor()
  entities = ["Napoleon Bonaparte", "Albert Einstein", "Bernardo O'Higgins"]
  contexts: List[Optional[str]] = [
    "Napoleon Bonaparte fue un militar y emperador francés.",
    "Albert Einstein desarrolló la teoría de la relatividad.",
    "Bernardo O'Higgins fue un militar y político chileno.",
  ]
  representations, _ = fit_and_transform_pipeline(
    extractor, entities, method="hs", contexts=contexts
  )
  print("\nResultados:")
  for i, entity in enumerate(entities):
    print(f"{entity}: shape={representations[i].shape}, mean={representations[i].mean():.4f}")
