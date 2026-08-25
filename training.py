"""Train and load the image CNN and clinical AdaBoost models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    CLINICAL_FEATURES,
    CLINICAL_LABEL_MAP,
    CLINICAL_META_PATH,
    CNN_PLOT_PATH,
    CSV_PATH,
    DATA_DIR,
    EFF_MODEL_PATH,
    ADA_MODEL_PATH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LR,
    NUM_EPOCHS,
    RAW_LAB_STATS,
    SAVE_DIR,
    SCALER_PATH,
    SEED,
    TARGET_COL,
    get_device,
)


def _torch_modules():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import datasets, models, transforms

    return torch, nn, optim, DataLoader, Dataset, random_split, datasets, models, transforms


def _transformed_subset_class(Dataset):
    class TransformedSubset(Dataset):
        """Apply a transform to a Subset of an ImageFolder that itself has no transform."""

        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, index):
            image, label = self.subset.dataset[self.subset.indices[index]]
            if self.transform is not None:
                image = self.transform(image)
            return image, label

    return TransformedSubset


def _set_seed(seed: int = SEED) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def save_class_names(class_names: list[str]) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )


def load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if DATA_DIR.exists():
        names = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
        if names:
            save_class_names(names)
            return names
    return ["Diseased", "Normal"]


def _build_efficientnet(num_classes: int):
    import torch.nn as nn
    from torchvision import models

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train_efficientnet(device=None):
    torch, nn, optim, DataLoader, Dataset, random_split, datasets, models, _transforms = (
        _torch_modules()
    )
    TransformedSubset = _transformed_subset_class(Dataset)
    device = device or get_device()
    _set_seed()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading image dataset...")
    raw_ds = datasets.ImageFolder(DATA_DIR)
    class_names = list(raw_ds.classes)
    print(f" Dataset loaded: {len(raw_ds)} images, classes: {class_names}")

    train_size = int(0.8 * len(raw_ds))
    val_size = len(raw_ds) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_raw, val_raw = random_split(
        raw_ds, [train_size, val_size], generator=generator
    )
    train_ds = TransformedSubset(train_raw, train_transform())
    val_ds = TransformedSubset(val_raw, eval_transform())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_acc_hist, val_acc_hist = [], []
    print("\nTraining EfficientNet...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        epoch_train_acc = correct / total
        train_acc_hist.append(epoch_train_acc)

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        epoch_val_acc = correct / total if total else 0.0
        val_acc_hist.append(epoch_val_acc)
        print(
            f"Epoch {epoch:02}/{NUM_EPOCHS} | "
            f"Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}"
        )

    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in val_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())
    print("\nValidation classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    torch.save(model.state_dict(), EFF_MODEL_PATH)
    save_class_names(class_names)
    print(f"EfficientNet saved to {EFF_MODEL_PATH}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_acc_hist, label="Train Accuracy")
    plt.plot(range(1, NUM_EPOCHS + 1), val_acc_hist, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EfficientNet Training Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig(CNN_PLOT_PATH, bbox_inches="tight")
    plt.close()
    print(f"Training plot saved to {CNN_PLOT_PATH}\n")

    return model, class_names


def load_efficientnet(device=None):
    import torch

    device = device or get_device()
    class_names = load_class_names()
    print("Loading EfficientNet model...")
    model = _build_efficientnet(len(class_names))
    state = torch.load(EFF_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, class_names


def load_or_train_efficientnet(device=None):
    device = device or get_device()
    if EFF_MODEL_PATH.exists():
        return load_efficientnet(device)
    print("No EfficientNet checkpoint found — training a new model.")
    return train_efficientnet(device)


def load_clinical_dataframe() -> pd.DataFrame:
    """Load ILP4SO.csv, coerce numerics, drop incomplete/invalid rows."""
    raw = pd.read_csv(CSV_PATH, encoding="latin1")
    n_raw = len(raw)
    cols = CLINICAL_FEATURES + [TARGET_COL]
    missing = [c for c in cols if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    data = raw[cols].apply(pd.to_numeric, errors="coerce")
    cleaned = data.dropna().copy()
    cleaned[TARGET_COL] = cleaned[TARGET_COL].astype(int)

    dropped = n_raw - len(cleaned)
    print(f"Clinical rows: {n_raw} read, {dropped} dropped, {len(cleaned)} kept")

    counts = cleaned[TARGET_COL].value_counts().sort_index()
    for label_id, name in CLINICAL_LABEL_MAP.items():
        n = int(counts.get(label_id, 0))
        print(f"  Outcome {label_id} ({name}): {n}")
    return cleaned


def raw_clinical_to_csv_space(values: dict) -> dict:
    """Map user-entered raw lab units into the already-z-scored CSV space. Age stays in years."""
    converted = {}
    for key in CLINICAL_FEATURES:
        value = float(values[key])
        if key in RAW_LAB_STATS:
            mean, std = RAW_LAB_STATS[key]
            converted[key] = (value - mean) / std
        else:
            converted[key] = value
    return converted


def train_adaboost():
    """Train AdaBoost on CSV-space features. Labs in the CSV are already z-scored; Age is raw.

    A StandardScaler is fit on the training split only (including Age) and saved
    so inference uses the same transform. The CSV is NOT z-scored a second time
    with RAW_LAB_STATS — that conversion is for raw GUI input only.
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print("\nLoading clinical dataset...")
    data = load_clinical_dataframe()

    X = data[CLINICAL_FEATURES]
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=CLINICAL_FEATURES, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=CLINICAL_FEATURES, index=X_test.index
    )

    print("Training AdaBoost...")
    sample_weight = compute_sample_weight("balanced", y_train)
    model = AdaBoostClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train_s, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_disease = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=[CLINICAL_LABEL_MAP[0], CLINICAL_LABEL_MAP[1]],
        zero_division=0,
    )

    print(f" AdaBoost accuracy: {acc:.4f}")
    print(f" Macro F1: {f1:.4f} | Disease-class F1: {f1_disease:.4f}")
    print(" Confusion matrix (rows=true, cols=pred):\n", cm)
    print(report)

    joblib.dump(model, ADA_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    meta = {
        "features": CLINICAL_FEATURES,
        "label_map": {str(k): v for k, v in CLINICAL_LABEL_MAP.items()},
        "n_samples": int(len(data)),
        "metrics": {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "disease_f1": float(f1_disease),
            "confusion_matrix": cm.tolist(),
        },
        "note": (
            "CSV lab columns are pre-z-scored; Age is in years. "
            "StandardScaler was fit on the training split only. "
            "At inference, convert raw labs with RAW_LAB_STATS then apply the scaler."
        ),
    }
    CLINICAL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"AdaBoost saved to {ADA_MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}\n")
    return model, scaler, meta


def load_adaboost():
    if not ADA_MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "AdaBoost model or clinical scaler is missing. Run train_models.py."
        )
    print("Loading AdaBoost model and clinical scaler...")
    model = joblib.load(ADA_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = {}
    if CLINICAL_META_PATH.exists():
        meta = json.loads(CLINICAL_META_PATH.read_text(encoding="utf-8"))
    return model, scaler, meta


def load_or_train_adaboost():
    if ADA_MODEL_PATH.exists() and SCALER_PATH.exists():
        return load_adaboost()
    print("No AdaBoost checkpoint/scaler pair found — training a new model.")
    return train_adaboost()


def predict_from_image(img_path: str | Path, model, class_names, device=None):
    import torch

    from image_gate import to_model_image, validate_image_file

    device = device or get_device()
    image = to_model_image(validate_image_file(img_path))
    tensor = eval_transform()(image).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        confidence = float(probs[idx].item())

    cls = class_names[idx]
    risk = "High Risk" if cls.lower() == "diseased" else "Low Risk"
    return cls, risk, confidence


def predict_from_clinical(raw_values: dict, model, scaler):
    csv_space = raw_clinical_to_csv_space(raw_values)
    frame = pd.DataFrame([csv_space], columns=CLINICAL_FEATURES)
    scaled = pd.DataFrame(scaler.transform(frame), columns=CLINICAL_FEATURES)
    pred = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    classes = list(model.classes_)
    prob_map = {int(c): float(p) for c, p in zip(classes, proba)}
    confidence = float(prob_map.get(pred, max(prob_map.values())))
    label = CLINICAL_LABEL_MAP.get(pred, str(pred))
    return pred, label, confidence, prob_map
