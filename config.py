from pathlib import Path

APP_TITLE = "Gesture to Power Mapper"
APP_ICON = "🪄"
APP_SUBTITLE = (
    "Recognise a hand gesture from your webcam or an uploaded image and map it to a game power, "
    "action, or ability."
)
CAMERA_HELP = "Show one gesture clearly in good lighting for the best prediction."

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "keras_model.h5"
LABELS_PATH = BASE_DIR / "models" / "labels.txt"