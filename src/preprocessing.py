"""Image preprocessing utilities for the MNIST Streamlit demo.

The trained models expect MNIST-like input: one grayscale channel, white digit
strokes on a black background, 28x28 pixels, normalized to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps


InvertMode = Literal["auto", "yes", "no"]


@dataclass(frozen=True)
class PreprocessResult:
    """Container returned by ``preprocess_digit_image``."""

    model_input: np.ndarray
    image_28x28: np.ndarray
    preview: Image.Image
    original_grayscale: Image.Image
    did_invert: bool
    bbox: tuple[int, int, int, int] | None


def load_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Load common image inputs into an RGB PIL image."""

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim == 2:
            return Image.fromarray(_to_uint8(array), mode="L").convert("RGB")
        if array.ndim == 3:
            if array.shape[2] == 4:
                return Image.fromarray(_to_uint8(array), mode="RGBA").convert("RGB")
            if array.shape[2] == 3:
                return Image.fromarray(_to_uint8(array), mode="RGB")

    raise TypeError(
        "image must be a path, PIL.Image.Image, or numpy array with shape "
        "(H, W), (H, W, 3), or (H, W, 4)."
    )


def preprocess_digit_image(
    image: str | Path | Image.Image | np.ndarray,
    *,
    invert: InvertMode = "auto",
    threshold: int = 30,
    padding: int = 4,
    output_size: int = 28,
) -> PreprocessResult:
    """Convert a handwritten digit image into MNIST model input.

    Parameters
    ----------
    image:
        Input from Streamlit uploader/camera, a file path, PIL image, or numpy
        array.
    invert:
        ``"auto"`` detects dark-on-light camera/upload images and converts them
        to MNIST style. ``"yes"`` always inverts. ``"no"`` never inverts.
    threshold:
        Minimum foreground intensity after inversion. Pixels below this value
        are treated as background for crop detection.
    padding:
        Number of pixels added around the detected digit before square resize.
    output_size:
        Final model image size. MNIST models should keep the default 28.
    """

    if invert not in {"auto", "yes", "no"}:
        raise ValueError("invert must be one of: 'auto', 'yes', 'no'.")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in the range [0, 255].")
    if padding < 0:
        raise ValueError("padding must be non-negative.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")

    original = load_image(image)
    grayscale = ImageOps.grayscale(original)
    gray_array = np.asarray(grayscale, dtype=np.uint8)

    did_invert = _should_invert(gray_array, invert)
    digit_array = 255 - gray_array if did_invert else gray_array.copy()

    digit_array = _suppress_low_intensity_background(digit_array, threshold)
    bbox = _foreground_bbox(digit_array, threshold)

    if bbox is None:
        cropped = digit_array
    else:
        cropped = digit_array[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    squared = _pad_to_square(cropped, padding=padding)
    resized = Image.fromarray(squared, mode="L").resize(
        (output_size, output_size),
        resample=Image.Resampling.LANCZOS,
    )

    resized_array = np.asarray(resized, dtype=np.float32) / 255.0
    resized_array = np.clip(resized_array, 0.0, 1.0)
    model_input = resized_array.reshape(1, output_size, output_size, 1).astype(
        np.float32
    )

    preview = Image.fromarray((resized_array * 255).astype(np.uint8), mode="L")

    return PreprocessResult(
        model_input=model_input,
        image_28x28=resized_array,
        preview=preview,
        original_grayscale=grayscale,
        did_invert=did_invert,
        bbox=bbox,
    )


def preprocess_for_model(
    image: str | Path | Image.Image | np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Return only the model tensor for inference code."""

    return preprocess_digit_image(image, **kwargs).model_input


def _to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if max_value <= 1.0:
            array = array * 255.0

    return np.clip(array, 0, 255).astype(np.uint8)


def _should_invert(array: np.ndarray, invert: InvertMode) -> bool:
    if invert == "yes":
        return True
    if invert == "no":
        return False

    border_values = np.concatenate(
        [
            array[0, :],
            array[-1, :],
            array[:, 0],
            array[:, -1],
        ]
    )
    center = array[
        array.shape[0] // 4 : array.shape[0] * 3 // 4,
        array.shape[1] // 4 : array.shape[1] * 3 // 4,
    ]

    border_mean = float(border_values.mean())
    center_mean = float(center.mean()) if center.size else float(array.mean())

    # Camera/upload photos are usually dark strokes on a bright page.
    # MNIST is the opposite: bright strokes on a dark background.
    return border_mean > 127 and center_mean < border_mean


def _suppress_low_intensity_background(array: np.ndarray, threshold: int) -> np.ndarray:
    clean = array.copy()
    clean[clean < threshold] = 0
    return clean


def _foreground_bbox(
    array: np.ndarray,
    threshold: int,
) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(array > threshold)
    if xs.size == 0 or ys.size == 0:
        return None

    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return left, top, right, bottom


def _pad_to_square(array: np.ndarray, padding: int) -> np.ndarray:
    height, width = array.shape
    side = max(height, width) + padding * 2
    canvas = np.zeros((side, side), dtype=np.uint8)

    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = array
    return canvas
