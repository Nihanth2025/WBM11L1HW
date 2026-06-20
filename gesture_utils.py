from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tensorflow as tf

TM_IMAGE_SIZE = (224, 224)


def load_labels(labels_file: str | Path) -> list[str]:
    lines = Path(labels_file).read_text(encoding="utf-8").splitlines()
    labels: list[str] = []
    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        parts = clean.split(maxsplit=1)
        labels.append(parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else clean)
    if not labels:
        raise ValueError("labels.txt is empty.")
    return labels


def load_local_teachable_machine_model(model_file: str | Path, labels_file: str | Path):
    model_path = Path(model_file)
    labels_path = Path(labels_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    labels = load_labels(labels_path)
    return model, labels


def preprocess_image_for_model(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB").resize(TM_IMAGE_SIZE), dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def get_top_predictions(probabilities: np.ndarray, labels: list[str], top_k: int = 3) -> list[dict[str, Any]]:
    scored = sorted(
        enumerate(probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"label": labels[i], "confidence": float(score)}
        for i, score in scored[: min(top_k, len(labels))]
    ]


def normalise_label(label: str, label_map: dict[str, str] | None = None) -> str:
    cleaned = label.strip()
    return label_map.get(cleaned, cleaned) if label_map else cleaned


def predict_gesture_from_image(model, labels: list[str], image: Image.Image, label_map: dict[str, str] | None = None):
    raw = model.predict(preprocess_image_for_model(image), verbose=0)[0]
    best_idx = int(np.argmax(raw))
    top_predictions = get_top_predictions(raw, labels)
    if label_map:
        top_predictions = [
            {"label": normalise_label(item["label"], label_map), "confidence": item["confidence"]}
            for item in top_predictions
        ]
    best_label = labels[best_idx]
    if label_map:
        best_label = normalise_label(best_label, label_map)
    return {
        "label": best_label,
        "confidence": float(raw[best_idx]),
        "top_predictions": top_predictions,
    }