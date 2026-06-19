"""Model discovery and loading utilities for the MNIST demo app."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np


InputKind = Literal["image", "flat"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata needed by the UI and inference code."""

    key: str
    display_name: str
    filename: str
    input_kind: InputKind
    accuracy: float | None = None
    description: str = ""

    @property
    def path(self) -> Path:
        return DEFAULT_MODELS_DIR / self.filename


@dataclass(frozen=True)
class LoadedModel:
    """A loaded Keras model paired with its app metadata."""

    spec: ModelSpec
    model: object


MODEL_SPECS: dict[str, ModelSpec] = {
    "standard_cnn": ModelSpec(
        key="standard_cnn",
        display_name="Standard CNN",
        filename="best_standard_cnn.keras",
        input_kind="image",
        accuracy=97.72,
        description="Baseline convolutional neural network.",
    ),
    "resnet9": ModelSpec(
        key="resnet9",
        display_name="ResNet-9",
        filename="best_resnet9.keras",
        input_kind="image",
        accuracy=99.59,
        description="Best accuracy model from the notebook benchmark.",
    ),
    "tinyvit": ModelSpec(
        key="tinyvit",
        display_name="TinyViT",
        filename="best_tinyvit.keras",
        input_kind="image",
        accuracy=97.72,
        description="Small Vision Transformer-style model.",
    ),
    "tuned_mlp": ModelSpec(
        key="tuned_mlp",
        display_name="Tuned MLP",
        filename="tuned_mnist_mlp.keras",
        input_kind="flat",
        accuracy=97.64,
        description="Dense baseline using flattened 28x28 input.",
    ),
}


def list_model_specs(
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    available_only: bool = True,
) -> list[ModelSpec]:
    """Return known model specs, optionally filtered to files on disk."""

    models_path = Path(models_dir)
    specs = list(MODEL_SPECS.values())

    if available_only:
        specs = [spec for spec in specs if (models_path / spec.filename).exists()]

    return specs


def list_model_names(
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    available_only: bool = True,
) -> list[str]:
    """Return display names for UI selectors."""

    return [
        spec.display_name
        for spec in list_model_specs(
            models_dir=models_dir,
            available_only=available_only,
        )
    ]


def get_model_spec(
    model_id: str,
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    require_available: bool = True,
) -> ModelSpec:
    """Resolve a model by key, filename stem, filename, or display name."""

    normalized = _normalize_model_id(model_id)
    models_path = Path(models_dir)

    for spec in MODEL_SPECS.values():
        aliases = {
            _normalize_model_id(spec.key),
            _normalize_model_id(spec.display_name),
            _normalize_model_id(spec.filename),
            _normalize_model_id(Path(spec.filename).stem),
        }
        if normalized in aliases:
            if require_available and not (models_path / spec.filename).exists():
                raise FileNotFoundError(
                    f"Model file not found for {spec.display_name}: "
                    f"{models_path / spec.filename}"
                )
            return spec

    known = ", ".join(spec.display_name for spec in MODEL_SPECS.values())
    raise KeyError(f"Unknown model '{model_id}'. Known models: {known}.")


def load_model(
    model_id: str,
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> LoadedModel:
    """Load one Keras model and return it with metadata."""

    spec = get_model_spec(model_id, models_dir=models_dir)
    model = _load_keras_model(str(Path(models_dir) / spec.filename))
    return LoadedModel(spec=spec, model=model)


def load_all_models(
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> dict[str, LoadedModel]:
    """Load all available known model artifacts."""

    loaded = {}
    for spec in list_model_specs(models_dir=models_dir, available_only=True):
        loaded[spec.key] = load_model(spec.key, models_dir=models_dir)
    return loaded


def clear_model_cache() -> None:
    """Clear the in-process Keras model cache."""

    _load_keras_model.cache_clear()


def adapt_input_for_model(
    preprocessed_input: np.ndarray,
    spec_or_model_id: ModelSpec | str,
) -> np.ndarray:
    """Adapt preprocessing output to the selected model's expected shape.

    ``preprocessing.preprocess_for_model`` returns ``(1, 28, 28, 1)``. CNN,
    ResNet, and TinyViT use that directly. The MLP uses a flattened
    ``(1, 784)`` vector.
    """

    spec = (
        spec_or_model_id
        if isinstance(spec_or_model_id, ModelSpec)
        else get_model_spec(spec_or_model_id, require_available=False)
    )

    array = np.asarray(preprocessed_input, dtype=np.float32)
    if spec.input_kind == "image":
        return _as_image_batch(array)
    if spec.input_kind == "flat":
        image_batch = _as_image_batch(array)
        return image_batch.reshape(image_batch.shape[0], -1)

    raise ValueError(f"Unsupported model input kind: {spec.input_kind}")


@lru_cache(maxsize=None)
def _load_keras_model(model_path: str) -> object:
    """Load a Keras model once per process."""

    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required to load .keras models. Install project "
            "dependencies first, for example: pip install -r requirements.txt"
        ) from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as first_error:
        try:
            return tf.keras.models.load_model(
                path,
                compile=False,
                safe_mode=False,
            )
        except TypeError:
            raise first_error
        except Exception as second_error:
            raise RuntimeError(f"Could not load Keras model: {path}") from second_error


def _as_image_batch(array: np.ndarray) -> np.ndarray:
    if array.shape == (28, 28):
        array = array.reshape(1, 28, 28, 1)
    elif array.shape == (28, 28, 1):
        array = array.reshape(1, 28, 28, 1)
    elif array.ndim == 2 and array.shape[1] == 784:
        array = array.reshape(array.shape[0], 28, 28, 1)
    elif array.ndim == 3 and array.shape[1:] == (28, 28):
        array = array[..., np.newaxis]

    if array.ndim != 4 or array.shape[1:] != (28, 28, 1):
        raise ValueError(
            "Expected preprocessed input with shape (1,28,28,1), "
            "(N,28,28,1), (28,28), (28,28,1), or (N,784). "
            f"Received {array.shape}."
        )

    return array.astype(np.float32)


def _normalize_model_id(model_id: str) -> str:
    return (
        model_id.strip()
        .lower()
        .replace(".keras", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
