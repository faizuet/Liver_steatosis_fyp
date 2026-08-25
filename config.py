"""Single source of truth for paths, hyperparameters, and clinical feature metadata."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Liver_steatosis"
CSV_PATH = ROOT / "ILP4SO.csv"
SAVE_DIR = ROOT / "models"

EFF_MODEL_PATH = SAVE_DIR / "efficientnet_liver_model.pth"
CLASS_NAMES_PATH = SAVE_DIR / "class_names.json"
ADA_MODEL_PATH = SAVE_DIR / "adaboost_model.pkl"
SCALER_PATH = SAVE_DIR / "clinical_scaler.pkl"
CLINICAL_META_PATH = SAVE_DIR / "clinical_meta.json"
CNN_PLOT_PATH = SAVE_DIR / "efficientnet_training.png"

# Ultrasound image gate (rejects typical photos before the CNN runs).
IMAGE_MIN_SIDE = 96
IMAGE_MAX_BYTES = 25 * 1024 * 1024
# Training scans in this project are grayscale with a large dark field.
IMAGE_MAX_COLORFULNESS = 20.0
IMAGE_MAX_COLORFULNESS_HARD = 80.0
IMAGE_MIN_DARK_RATIO = 0.15
IMAGE_MAX_MEAN_LUMA = 140.0
IMAGE_MIN_CONTRAST = 12.0
IMAGE_LOW_CONFIDENCE = 0.62

BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-4
SEED = 42


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ImageNet mean/std used by EfficientNet-B0 pretraining — must match inference.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLINICAL_FEATURES = [
    "Age",
    "Direct_Bilrubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]
TARGET_COL = "Outcome"

# CSV Outcome encoding used by this dataset (not the original ILPD 1/2 coding).
CLINICAL_LABEL_MAP = {0: "Normal", 1: "Disease"}

# Stats used when ILP4SO.csv lab columns were z-scored from raw ILPD-style units.
# Age in the CSV was left in years and is NOT converted with these stats.
# Applied at inference only, to map user-entered raw labs into CSV feature space.
RAW_LAB_STATS = {
    "Direct_Bilrubin": (1.73, 3.19),
    "Alkaline_Phosphotase": (313.10, 259.31),
    "Alamine_Aminotransferase": (102.83, 221.45),
    "Aspartate_Aminotransferase": (142.64, 355.68),
    "Albumin": (3.14, 0.78),
    "Albumin_and_Globulin_Ratio": (0.93, 0.31),
}

# GUI labels and plausible raw-unit ranges (user input, not z-scores).
FEATURE_META = {
    "Age": {
        "label": "Age",
        "unit": "years",
        "min": 1.0,
        "max": 120.0,
    },
    "Direct_Bilrubin": {
        "label": "Direct bilirubin",
        "unit": "mg/dL",
        "min": 0.0,
        "max": 30.0,
    },
    "Alkaline_Phosphotase": {
        "label": "Alkaline phosphatase (ALP)",
        "unit": "IU/L",
        "min": 20.0,
        "max": 2500.0,
    },
    "Alamine_Aminotransferase": {
        "label": "ALT (SGPT)",
        "unit": "IU/L",
        "min": 1.0,
        "max": 2500.0,
    },
    "Aspartate_Aminotransferase": {
        "label": "AST (SGOT)",
        "unit": "IU/L",
        "min": 1.0,
        "max": 5000.0,
    },
    "Albumin": {
        "label": "Albumin",
        "unit": "g/dL",
        "min": 0.5,
        "max": 6.5,
    },
    "Albumin_and_Globulin_Ratio": {
        "label": "Albumin / globulin ratio",
        "unit": "",
        "min": 0.1,
        "max": 4.0,
    },
}

DISCLAIMER = (
    "Research prototype for academic use only. Not a medical device. "
    "Predictions are not a diagnosis — consult a qualified clinician."
)

SAFETY_NORMAL = (
    "Safety advice: A normal score is not a medical clearance and does not mean "
    "you are healthy. If you have symptoms (pain, yellowing of the eyes, swelling, "
    "or unusual tiredness) or a doctor has asked for tests, still see a clinician. "
    "Do not skip follow-up care because of this app."
)

SAFETY_DISEASE = (
    "Safety advice: This is not a confirmed diagnosis. Do not start, stop, or "
    "change any medicine on your own. Please consult a qualified doctor promptly "
    "for proper examination and tests. Seek urgent care for severe abdominal pain, "
    "jaundice, persistent vomiting, or sudden worsening."
)
