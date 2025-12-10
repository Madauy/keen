import logging
import re
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
  MAX_NEW_TOKENS,
  SIMILARITY_MODEL_NAME,
  SIMILARITY_THRESHOLD,
  get_language_config,
)

logger = logging.getLogger(__name__)


class SimilarityEvaluator:
  """Handles semantic similarity evaluation of answers."""

  def __init__(self):
    """Initialize the similarity evaluator."""
    self.model: Optional[SentenceTransformer] = None
    self.model_name = SIMILARITY_MODEL_NAME

  def load_model(self) -> None:
    """Load the sentence transformer model."""
    if self.model is None:
      logger.info(f"Loading similarity model: {self.model_name}")
      self.model = SentenceTransformer(self.model_name)
      logger.info("Similarity model loaded successfully")

  def unload_model(self) -> None:
    """Unload the model to free memory."""
    if self.model is not None:
      del self.model
      self.model = None
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      logger.info("Similarity model unloaded")

  def _is_year_answer(self, text: str) -> bool:
    """Check if the answer is a 4-digit year.

    Args:
        text: The text to check

    Returns:
        True if the text is a 4-digit year
    """
    return bool(re.match(r"^\d{4}$", text.strip()))

  def _calculate_similarity(self, text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity score (0-1)
    """
    if self.model is None:
      raise RuntimeError("Similarity model not loaded. Call load_model() first.")

    embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity.item())

  def _normalize_text(self, text: str) -> str:
    """Normalize text for comparison.

    Args:
        text: Text to normalize

    Returns:
        Normalized text (lowercase, stripped)
    """
    return text.strip().lower()

  def evaluate(
    self,
    predicted: str,
    respuestas: list[str],
    aliases: list[list[str]],
  ) -> tuple[bool, Optional[float], Optional[str]]:
    """Evaluate if the predicted answer matches any expected answer.

    Args:
        predicted: The LLM's predicted answer
        respuestas: List of correct answers
        aliases: List of alias lists for each answer

    Returns:
        Tuple of (is_correct, similarity_score, matched_answer)
    """
    predicted_normalized = self._normalize_text(predicted)
    best_similarity = 0.0
    best_match: Optional[str] = None

    # Check against primary answers
    for answer in respuestas:
      answer_normalized = self._normalize_text(answer)

      # Exact match
      if predicted_normalized == answer_normalized:
        return True, 1.0, answer

      # For years, only exact match
      if self._is_year_answer(answer_normalized):
        if predicted_normalized == answer_normalized:
          return True, 1.0, answer
        continue

      # Similarity check for non-year answers
      similarity = self._calculate_similarity(predicted, answer)
      if similarity > best_similarity:
        best_similarity = similarity
        best_match = answer

      if similarity >= SIMILARITY_THRESHOLD:
        return True, similarity, answer

    # Check against aliases
    for alias_group in aliases:
      for alias in alias_group:
        alias_normalized = self._normalize_text(alias)

        # Exact match with alias
        if predicted_normalized == alias_normalized:
          return True, 1.0, alias

        # For years, only exact match
        if self._is_year_answer(alias_normalized):
          if predicted_normalized == alias_normalized:
            return True, 1.0, alias
          continue

        # Similarity check for non-year aliases
        similarity = self._calculate_similarity(predicted, alias)
        if similarity > best_similarity:
          best_similarity = similarity
          best_match = alias

        if similarity >= SIMILARITY_THRESHOLD:
          return True, similarity, alias

    # No match found
    return False, best_similarity if best_similarity > 0 else None, best_match


class Evaluator:
  """Main evaluator class for LLM inference and answer evaluation."""

  def __init__(self, model_name: str, language: str):
    """Initialize the evaluator.

    Args:
        model_name: HuggingFace model name/path
        language: Language code for prompts ('es' or 'en')
    """
    self.model_name = model_name
    self.language = language
    self.language_config = get_language_config(language)

    self.model = None
    self.tokenizer = None
    self.similarity_evaluator = SimilarityEvaluator()

  def load_model(self) -> None:
    """Load the LLM model and tokenizer."""
    logger.info(f"Loading LLM model: {self.model_name}")

    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
      self.model_name,
      torch_dtype="auto",
      device_map="auto",
    )

    # Set pad token if not set
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    logger.info("LLM model loaded successfully")

    # Also load similarity model
    self.similarity_evaluator.load_model()

  def unload_model(self) -> None:
    """Unload models to free memory."""
    if self.model is not None:
      del self.model
      self.model = None
    if self.tokenizer is not None:
      del self.tokenizer
      self.tokenizer = None

    if torch.cuda.is_available():
      torch.cuda.empty_cache()

    self.similarity_evaluator.unload_model()
    logger.info("LLM model unloaded")

  def _build_prompt(self, question: str) -> tuple[str, str]:
    """Build the prompt for the LLM.

    Args:
        question: The question to answer

    Returns:
        Tuple of (system_message, user_message)
    """
    system_message = self.language_config["system_message"]

    # Build few-shot examples
    examples_text = ""
    for example in self.language_config["few_shot_examples"]:
      examples_text += f"{example['pregunta']}\n{example['respuesta']}\n\n"

    user_message = f"{examples_text}{question}"

    return system_message, user_message

  def generate_response(self, question: str) -> str:
    """Generate a response from the LLM.

    Args:
        question: The question to answer

    Returns:
        The LLM's response (cleaned)
    """
    if self.model is None or self.tokenizer is None:
      raise RuntimeError("Model not loaded. Call load_model() first.")

    system_message, user_message = self._build_prompt(question)

    # Build chat messages
    messages = [
      {"role": "system", "content": system_message},
      {"role": "user", "content": user_message},
    ]

    # Apply chat template
    prompt = self.tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
    )

    # Tokenize
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    # Generate
    with torch.no_grad():
      outputs = self.model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=self.tokenizer.pad_token_id,
      )

    # Decode response (only the new tokens)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return self._clean_response(response)

  def _clean_response(self, response: str) -> str:
    """Clean the LLM response.

    Args:
        response: Raw response from the LLM

    Returns:
        Cleaned response
    """
    # Strip whitespace
    response = response.strip()

    # Take only the first line if multiple lines
    if "\n" in response:
      response = response.split("\n")[0].strip()

    return response

  def evaluate_answer(
    self,
    predicted: str,
    respuestas: list[str],
    aliases: list[list[str]],
  ) -> tuple[bool, Optional[float], Optional[str]]:
    """Evaluate if the predicted answer is correct.

    Args:
        predicted: The LLM's predicted answer
        respuestas: List of correct answers
        aliases: List of alias lists for each answer

    Returns:
        Tuple of (is_correct, similarity_score, matched_answer)
    """
    return self.similarity_evaluator.evaluate(predicted, respuestas, aliases)
