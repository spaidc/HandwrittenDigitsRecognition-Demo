"""Inference helpers for the MNIST demo app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.model_loader import (
    DEFAULT_MODELS_DIR,
    LoadedModel,
    ModelSpec,
    adapt_input_for_model,
    get_model_spec,
    load_model,
)
from src.preprocessing import InvertMode, PreprocessResult, preprocess_digit_image


@dataclass(frozen=True)
class TopPrediction:
    """One class entry for a ranked prediction list."""

    digit: int
    probability: float


@dataclass(frozen=True)
class PredictionResult:
    """Normalized prediction output suitable for UI rendering."""

    model_key: str
    model_name: str
    predicted_digit: int
    confidence: float
    probabilities: np.ndarray
    top_predictions: list[TopPrediction]
    model_input: np.ndarray
    raw_output: np.ndarray


@dataclass(frozen=True)
class ImageInferenceResult:
    """Full result when inference starts from an uploaded/camera image."""

    prediction: PredictionResult
    preprocessing: PreprocessResult


def predict_image(
    image: str | Path | Image.Image | np.ndarray,
    model_id: str,
    *,
    loaded_model: LoadedModel | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    invert: InvertMode = "auto",
    threshold: int = 30,
    padding: int = 4,
    top_k: int = 3,
) -> ImageInferenceResult:
    """Preprocess one image and run inference with the selected model."""

    preprocessing = preprocess_digit_image(
        image,
        invert=invert,
        threshold=threshold,
        padding=padding,
    )
    prediction = predict_preprocessed(
        preprocessing.model_input,
        model_id,
        loaded_model=loaded_model,
        models_dir=models_dir,
        top_k=top_k,
    )
    return ImageInferenceResult(prediction=prediction, preprocessing=preprocessing)


def predict_preprocessed(
    preprocessed_input: np.ndarray,
    model_id: str,
    *,
    loaded_model: LoadedModel | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    top_k: int = 3,
) -> PredictionResult:
    """Run inference on an already preprocessed image/tensor."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    loaded = loaded_model or load_model(model_id, models_dir=models_dir)
    requested_spec = get_model_spec(
        model_id,
        models_dir=models_dir,
        require_available=loaded_model is None,
    )
    if loaded.spec.key != requested_spec.key:
        raise ValueError(
            "loaded_model does not match model_id: "
            f"{loaded.spec.display_name} != {requested_spec.display_name}"
        )

    model_input = adapt_input_for_model(preprocessed_input, loaded.spec)
    raw_output = _run_model_prediction(loaded.model, model_input)
    probabilities = normalize_probabilities(raw_output)[0]

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return PredictionResult(
        model_key=loaded.spec.key,
        model_name=loaded.spec.display_name,
        predicted_digit=predicted_digit,
        confidence=confidence,
        probabilities=probabilities,
        top_predictions=rank_probabilities(probabilities, top_k=top_k),
        model_input=model_input,
        raw_output=raw_output,
    )


def normalize_probabilities(raw_output: np.ndarray) -> np.ndarray:
    """Return a valid probability matrix with shape ``(N, 10)``.

    Softmax Keras models already emit probabilities. This function also accepts
    logits or unnormalized scores, which keeps the UI robust if a future model
    is exported without a final softmax layer.
    """

    scores = np.asarray(raw_output, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    if scores.ndim != 2 or scores.shape[1] != 10:
        raise ValueError(
            "Expected model output with shape (10,) or (N, 10). "
            f"Received {scores.shape}."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model output contains NaN or infinite values.")

    row_sums = scores.sum(axis=1, keepdims=True)
    looks_like_probabilities = (
        np.all(scores >= 0.0)
        and np.all(row_sums > 0.0)
        and np.allclose(row_sums, 1.0, atol=1e-3)
    )

    if looks_like_probabilities:
        probabilities = scores / row_sums
    else:
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    return probabilities.astype(np.float32)


def rank_probabilities(
    probabilities: np.ndarray,
    *,
    top_k: int = 3,
) -> list[TopPrediction]:
    """Return the highest-probability digits in descending order."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    vector = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if vector.shape[0] != 10:
        raise ValueError(f"Expected 10 class probabilities. Received {vector.shape}.")

    k = min(top_k, 10)
    indices = np.argsort(vector)[::-1][:k]
    return [
        TopPrediction(digit=int(index), probability=float(vector[index]))
        for index in indices
    ]


def format_confidence(probability: float) -> str:
    """Format a probability as a UI-friendly percentage."""

    return f"{probability * 100:.2f}%"


def _run_model_prediction(model: Any, model_input: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        output = model.predict(model_input, verbose=0)
    else:
        output = model(model_input)
        if hasattr(output, "numpy"):
            output = output.numpy()

    return np.asarray(output)
