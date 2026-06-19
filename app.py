"""Streamlit app for handwritten digit recognition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st

from src.inference import format_confidence, predict_image
from src.model_loader import LoadedModel, ModelSpec, list_model_specs, load_model
from src.preprocessing import InvertMode


APP_ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def cached_load_model(model_key: str) -> LoadedModel:
    return load_model(model_key)


def main() -> None:
    st.set_page_config(
        page_title="MNIST Digit Recognition",
        layout="wide",
    )
    _inject_styles()

    specs = list_model_specs()
    if not specs:
        st.error("No model files were found in the models folder.")
        return

    selected_spec, input_mode, invert, threshold, padding, top_k = _render_sidebar(
        specs
    )

    st.title("Handwritten Digit Recognition")

    image = _read_user_image(input_mode)
    if image is None:
        _render_empty_state(input_mode)
        return

    try:
        loaded_model = cached_load_model(selected_spec.key)
    except Exception as exc:
        st.error(str(exc))
        return

    try:
        result = predict_image(
            image,
            selected_spec.key,
            loaded_model=loaded_model,
            invert=invert,
            threshold=threshold,
            padding=padding,
            top_k=top_k,
        )
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    _render_prediction(image, selected_spec, result)


def _render_sidebar(
    specs: list[ModelSpec],
) -> tuple[ModelSpec, str, InvertMode, int, int, int]:
    with st.sidebar:
        st.header("Controls")

        default_index = next(
            (index for index, spec in enumerate(specs) if spec.key == "resnet9"),
            0,
        )
        selected_spec = st.selectbox(
            "Model",
            specs,
            index=default_index,
            format_func=lambda spec: spec.display_name,
        )

        if selected_spec.accuracy is not None:
            st.caption(f"Notebook accuracy: {selected_spec.accuracy:.2f}%")
        st.caption(selected_spec.description)

        input_mode = st.radio(
            "Input",
            ["Upload image", "Camera snapshot", "Draw on canvas"],
            horizontal=False,
        )

        st.divider()
        st.subheader("Preprocessing")
        invert_label = st.selectbox(
            "Invert",
            ["Auto", "Force invert", "Keep as-is"],
            index=0,
        )
        invert_map: dict[str, InvertMode] = {
            "Auto": "auto",
            "Force invert": "yes",
            "Keep as-is": "no",
        }
        threshold = st.slider("Threshold", 0, 160, 30, 5)
        padding = st.slider("Padding", 0, 14, 4, 1)
        top_k = st.slider("Top predictions", 1, 10, 3, 1)

    return selected_spec, input_mode, invert_map[invert_label], threshold, padding, top_k


def _read_user_image(input_mode: str) -> Image.Image | None:
    if input_mode == "Upload image":
        uploaded_file = st.file_uploader(
            "Image file",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
        )
        if uploaded_file is None:
            return None
        return _open_uploaded_image(uploaded_file)

    if input_mode == "Camera snapshot":
        camera_file = st.camera_input("Camera")
        if camera_file is None:
            return None
        return _open_uploaded_image(camera_file)

    return _read_canvas_image()


def _open_uploaded_image(file) -> Image.Image | None:
    try:
        return Image.open(file).convert("RGB")
    except UnidentifiedImageError:
        st.error("The selected file is not a valid image.")
        return None


def _read_canvas_image() -> Image.Image | None:
    try:
        from streamlit_drawable_canvas import st_canvas
    except ModuleNotFoundError:
        st.error(
            "Canvas input requires streamlit-drawable-canvas. "
            "Run: python -m pip install -r requirements.txt"
        )
        return None

    st.caption("Draw one digit using the white pen.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
    )

    if canvas_result.image_data is None:
        return None

    canvas = np.asarray(canvas_result.image_data, dtype=np.uint8)
    if canvas.ndim != 3 or canvas.shape[2] < 3:
        return None

    rgb = canvas[:, :, :3]
    if np.count_nonzero(rgb > 20) < 20:
        return None

    return Image.fromarray(rgb, mode="RGB")


def _render_empty_state(input_mode: str) -> None:
    labels = {
        "Upload image": "upload an image",
        "Camera snapshot": "capture a photo",
        "Draw on canvas": "draw a digit on the canvas",
    }
    label = labels.get(input_mode, "provide an input")
    st.info(f"Ready to run once you {label}.")


def _render_prediction(image: Image.Image, selected_spec: ModelSpec, result) -> None:
    prediction = result.prediction
    preprocessing = result.preprocessing

    left, middle, right = st.columns([1.15, 0.85, 1.2], gap="large")

    with left:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    with middle:
        st.subheader("Processed")
        st.image(preprocessing.preview.resize((196, 196)), width=196)
        st.caption(
            f"Invert: {'yes' if preprocessing.did_invert else 'no'} | "
            f"Tensor: {tuple(prediction.model_input.shape)}"
        )

    with right:
        st.subheader("Prediction")
        st.markdown(
            f"""
            <div class="prediction-box">
              <div class="prediction-label">Digit</div>
              <div class="prediction-digit">{prediction.predicted_digit}</div>
              <div class="prediction-confidence">{format_confidence(prediction.confidence)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Model: {selected_spec.display_name}")

    st.divider()

    top_col, prob_col = st.columns([0.9, 1.5], gap="large")
    with top_col:
        st.subheader("Top Classes")
        for item in prediction.top_predictions:
            st.metric(
                label=f"Digit {item.digit}",
                value=format_confidence(item.probability),
            )

    with prob_col:
        st.subheader("Class Probabilities")
        _render_probability_bars(prediction.probabilities)


def _render_probability_bars(probabilities) -> None:
    for digit, probability in enumerate(probabilities):
        probability = float(probability)
        cols = st.columns([0.18, 1.0, 0.24], gap="small")
        cols[0].markdown(f"**{digit}**")
        cols[1].progress(min(max(probability, 0.0), 1.0))
        cols[2].markdown(f"{probability * 100:.1f}%")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
          }
          .prediction-box {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 8px;
            padding: 18px 20px;
            background: #ffffff;
          }
          .prediction-label {
            color: #4b5563;
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
          }
          .prediction-digit {
            color: #111827;
            font-size: 5rem;
            line-height: 1;
            font-weight: 800;
          }
          .prediction-confidence {
            color: #2563eb;
            font-size: 1.2rem;
            font-weight: 700;
            margin-top: 0.4rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
