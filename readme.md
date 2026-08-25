# Liver Diagnosis System

Final Year Project (FYP) — a **desktop research prototype** that screens for liver disease using two independent machine-learning models:

1. **Ultrasound images** → EfficientNet-B0 (Diseased vs Normal)
2. **Clinical laboratory values** → AdaBoost (Normal vs Disease)

This is **not a medical device**. Predictions are for academic demonstration only and are not a diagnosis.

---

## 1. Project Overview

Liver disease is often assessed with imaging (for example ultrasound) and blood tests. This FYP builds a **local Python application** that lets a user:

- upload a liver ultrasound image, or
- type standard lab values (bilirubin, enzymes, albumin, and so on)

and receive a **model prediction with a confidence score**.

**Problem it addresses:** a simple, inspectable demo that connects real ML inference to a usable interface — without a hospital system, cloud API, or web backend.

**Objective:** train, save, and run two binary classifiers, and present results clearly in a desktop GUI, with honest limitations (small image set, processed lab CSV, research-only use).

The two models are **not combined**. Each screen uses its own model.

---

## 2. Key Features

- Desktop GUI (CustomTkinter) with Ultrasound and Clinical labs views
- Image classification with softmax confidence
- Ultrasound image gate: reject typical non-scan photos before a disease label is shown
- Clinical prediction with class probabilities
- Input checks on lab fields (required numbers and allowed ranges)
- Shared training/inference code so the GUI and CLI cannot drift apart
- Saved artifacts: CNN weights, AdaBoost model, clinical scaler, class names
- Research disclaimer on every screen

Not included (by design): login, multi-user roles, REST API, or a database.

---

## 3. How the System Works

1. Run `python main.py`.
2. A splash screen **loads** saved models from `models/` (or **trains** them if a checkpoint is missing).
3. The main window opens with two workspace pages:
   - **Ultrasound** — select an image, run analysis, see class + confidence.
   - **Clinical labs** — enter raw lab units, run analysis, see Normal/Disease + probabilities.
4. Closing the app does **not** store prediction history. Results exist only in that session.

Optional: `python train_models.py` trains models from the command line without opening the GUI.

---

## 4. Technology Stack

| Area | What this project uses |
|---|---|
| Language | Python 3.12 (3.11 also fine if Tcl/Tk is installed) |
| Desktop UI | CustomTkinter, Tkinter, Pillow |
| Image ML | PyTorch, torchvision (EfficientNet-B0) |
| Clinical ML | scikit-learn (AdaBoost, StandardScaler) |
| Data handling | pandas, NumPy, joblib |
| Plots (training only) | matplotlib (saved to file, not shown as a blocking window) |
| Database | **None** — files only |
| Web / REST API | **None** — in-process function calls |
| Cloud / extra services | **None** |

Pinned versions are in `requirements.txt`.

---

## 5. Project Structure

```
Liver_Diagnosis_System/
├── main.py              # Start the GUI
├── train_models.py      # Train models (CLI)
├── config.py            # Paths, features, lab ranges, label map
├── training.py          # Train, load, and predict
├── image_gate.py        # Reject non-ultrasound images before CNN
├── liver_gui.py         # Desktop interface
├── ui_theme.py          # Colours, icons, layout tokens
├── requirements.txt
├── ILP4SO.csv           # Clinical dataset
├── Liver_steatosis/     # Ultrasound images (Diseased/, Normal/)
└── models/              # Saved checkpoints
    ├── efficientnet_liver_model.pth
    ├── class_names.json
    ├── adaboost_model.pkl
    ├── clinical_scaler.pkl
    └── clinical_meta.json
```

`main.py` only launches the app. All ML logic lives in `training.py`.

---

## 6. System / Data Flow

```
User
  ├─ Ultrasound image ──► file + appearance checks (image_gate.py)
  │                         ──► if rejected: no class label
  │                         ──► resize 224×224, ImageNet normalize
  │                         ──► EfficientNet-B0 (eval mode)
  │                         ──► grayscale (match training) + EfficientNet-B0
  │                         ──► class + confidence  (or withhold)
  │
  └─ Raw lab values ──► range check
                          ──► map labs into CSV space (RAW_LAB_STATS)
                          ──► saved StandardScaler
                          ──► AdaBoost
                          ──► class + probabilities
```

**Ultrasound**

- Dataset folders: `Diseased`, `Normal`. Names are stored in `class_names.json` (ImageFolder order is alphabetical: Diseased = 0, Normal = 1).
- Before prediction, `image_gate.py` rejects typical colour photos. Real ultrasound PNGs (including machine text/calipers) are allowed. Low confidence is shown as a caution, not a withheld result.
- This is not a full modality detector: colour Doppler or odd crops may be rejected; other grayscale medical images may still pass.
- Training uses random flip/rotation; **validation and inference do not**.
- Split is 80/20, seeded (`SEED = 42`).
- The image set is small (~72 augmented images). Treat accuracy as a demo metric, not a clinical claim.

**Clinical**

- Source: `ILP4SO.csv` (ILPD-style table). Lab columns are **already z-scored**; **Age is in years**.
- GUI users type **raw units** (mg/dL, IU/L, g/dL). Those labs are converted with `RAW_LAB_STATS` in `config.py`, then the **training-split scaler** is applied.
- Labels in this CSV: `0` = Normal, `1` = Disease (not the original ILPD 1/2 coding).
- Incomplete rows are dropped. Last recorded training snapshot in `clinical_meta.json`: 199 rows, stratified split, balanced sample weights.

There is no API gateway and no database write in this path.

---

## 7. Installation & Setup

**Prerequisites**

- Windows, macOS, or Linux
- Python 3.11+ with **Tcl/Tk** (needed for the GUI). On Windows, repair Python and enable **tcl/tk and IDLE** if `init.tcl` is missing.
- Optional: NVIDIA GPU + CUDA for faster CNN training (CPU works)

**Setup (Windows PowerShell)**

```powershell
cd "path\to\Liver_Diagnosis_System"
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Environment / database**

- No `.env` file and no database setup.
- Paths are taken from `config.py` (project folder), not from the current working directory.

**Retrain models**

```bash
python train_models.py
python train_models.py --clinical-only
python train_models.py --image-only
```

If `adaboost_model.pkl` exists without `clinical_scaler.pkl`, the app trains AdaBoost again. The scaler must always pair with the clinical model.

---

## 8. Usage

1. Start the app with `python main.py`.
2. Wait until the splash screen finishes loading models.
3. **Ultrasound:** choose an image (JPG/PNG/BMP/TIFF) → **Run analysis**.
4. **Clinical labs:** fill every field in raw units → **Run analysis**. Invalid or out-of-range values are highlighted on the form.
5. Read the result card (class, confidence, and — for labs — both class probabilities).
6. Treat every output as a **research prototype**, not a clinical report.

---

## 9. API Overview

This project **does not provide HTTP endpoints**. The GUI calls these functions in the same process (`training.py`):

| Function | Role |
|---|---|
| `load_or_train_efficientnet()` | Load CNN weights, or train if missing |
| `load_or_train_adaboost()` | Load AdaBoost + scaler, or train if missing |
| `predict_from_image(...)` | Image → class, risk label, confidence |
| `predict_from_clinical(...)` | Raw labs → class, label, confidence, probabilities |

There is no `/predict` URL, JWT, or CORS layer to document.

---

## 10. FYP Objective / Expected Outcome

The project is meant to show a **complete, defensible pipeline**:

- two real trained models (not placeholder/fake predictions)
- a consistent train/inference path
- a usable desktop interface
- documented data conventions and limitations

**Expected outcome for evaluation:** an examiner can install the app, run both analysis paths, and explain how an ultrasound or a lab vector becomes a labelled prediction — including why this remains a **prototype**, not a clinical system.

---

## Disclaimer

For academic use only. Not a substitute for professional medical advice, diagnosis, or treatment.
