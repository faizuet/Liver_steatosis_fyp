# Liver Diagnosis System

Final Year Project — a **desktop research prototype** that screens for liver disease using two independent machine-learning models:

1. **Liver ultrasound image** → EfficientNet-B0 (`Diseased` vs `Normal`)
2. **Clinical laboratory values** → AdaBoost (`Normal` vs `Disease`)

This is **not a medical device**. Outputs are for academic demonstration only and are not a diagnosis.

The two models are **not fused**. Each workspace page runs its own model and shows its own result.

---

## Project Overview

Liver assessment in practice often uses imaging and blood tests. This project is a **local Python desktop app** that lets a user:

- upload a liver ultrasound, or
- type standard lab values (bilirubin, enzymes, albumin, and related fields)

and receive a **labelled prediction with a confidence score**.

**Purpose:** show a complete, inspectable train-and-infer pipeline with a usable interface — without a hospital system, cloud service, or web backend.

**Objective:** train, save, and run two binary classifiers, validate inputs before they reach the models, and present results clearly, including safety text and known limitations (small image set, processed lab CSV, research-only use).

---

## Key Features

### Ultrasound analysis

- User selects a scan (JPG, PNG, BMP, or TIFF).
- The app shows a preview and runs **file checks** (type, size, emptiness, corruption, resolution).
- It then checks whether the image **looks like an ultrasound** (dark field, limited colour, scan texture) so a typical colour photo is not given a disease label.
- If the file is accepted, **EfficientNet-B0** scores `Diseased` vs `Normal` with softmax confidence.
- Real clinical PNGs with machine overlays (text, calipers) are allowed. If confidence is moderate, the result still appears with a **caution**, not a hard reject.
- Unsupported files never reach the CNN. The same checks run again at prediction time, so the UI cannot skip them.

### Clinical laboratory analysis

- User enters **raw clinical units** (for example mg/dL, IU/L, g/dL), not z-scores.
- Every field is required, must be numeric, and must fall in the range shown on the form.
- **AdaBoost** returns `Normal` or `Disease`, overall confidence, and **both class probabilities**.
- Invalid fields are highlighted; analysis does not run until they are fixed.

### Desktop workspace

- CustomTkinter GUI with a light clinical layout.
- Splash screen loads both models (or trains them if a checkpoint is missing).
- Sidebar switches between **Ultrasound** and **Clinical labs**.
- Result cards cover idle, loading, success, and error states.
- After a successful score, **safety advice** is shown (a normal score is not a clearance; a disease score is not a confirmed diagnosis).
- A disclaimer is always visible in the footer.

### Training CLI

- `python train_models.py` retrains from the command line without opening the GUI.
- Optional flags: `--image-only`, `--clinical-only`.

**Not included (by design):** login, user accounts, REST/HTTP API, database, prediction history, or a web frontend.

---

## Application Flow

1. Run `python main.py`.
2. On Windows, `main.py` may set `TCL_LIBRARY` / `TK_LIBRARY` so Tkinter can find Tcl/Tk inside the base Python install.
3. Splash screen:
   - load EfficientNet-B0 from `models/efficientnet_liver_model.pth` (train if missing)
   - load AdaBoost + scaler from `models/` (train if the pair is missing)
4. Main window opens. The user stays on one machine; there is no network call.

**Ultrasound path**

1. User selects an image (or clicks **Run analysis** with no file → error).
2. `inspect_image_path` checks the file is present, non-empty, an allowed type/format, under 25 MB, and between 96 px (shortest side) and 8192 px (longest side). Damaged files are rejected.
3. `assess_ultrasound_image` rejects typical photographs.
4. User clicks **Run analysis**.
5. `predict_from_image` runs the **full gate again**, converts the image to grayscale RGB (to match training), resizes to 224×224, applies ImageNet normalization, and runs EfficientNet-B0 in eval mode.
6. The result card shows class, a simple risk label (`High Risk` / `Low Risk`), confidence, and safety advice. Nothing is saved to disk.

**Clinical path**

1. User fills all seven fields and clicks **Run analysis** (Enter in a field also submits).
2. The GUI range-checks every value.
3. Labs (not Age) are mapped into the CSV feature space using `RAW_LAB_STATS` in `config.py`.
4. The saved `StandardScaler` is applied, then AdaBoost predicts.
5. The result card shows class, confidence, both probabilities, and safety advice. Nothing is saved to disk.

Closing the app discards the session. There is no history store.

```
User
  ├─ Ultrasound file
  │     ► inspect (type / size / integrity / dimensions)
  │     ► ultrasound appearance check
  │     ► grayscale RGB → 224×224 → ImageNet normalize
  │     ► EfficientNet-B0 (eval)
  │     ► class + confidence (+ caution if confidence is moderate)
  │
  └─ Raw lab values
        ► required / numeric / range checks
        ► RAW_LAB_STATS (labs only; Age stays in years)
        ► StandardScaler (fit on train split)
        ► AdaBoost
        ► class + probabilities
```

---

## AI/ML Features

### EfficientNet-B0 (ultrasound)

| | |
|---|---|
| **Role** | Binary image classifier: `Diseased` vs `Normal` |
| **Library** | PyTorch / torchvision |
| **Input** | Validated ultrasound image |
| **Processing** | Grayscale→RGB, resize 224×224, ImageNet mean/std |
| **Output** | Predicted class name, risk label, softmax confidence |
| **Training** | ImageFolder on `Liver_steatosis/` (`Diseased/`, `Normal/`). Pretrained ImageNet weights, classifier head replaced for 2 classes. 80/20 split, seed 42. Train-time flip/rotation; validation and inference have **no** random augment. 30 epochs, batch 16, Adam, learning rate `1e-4`. |
| **Artifacts** | `models/efficientnet_liver_model.pth`, `models/class_names.json` |

Class names follow folder order (alphabetical): index `0` = Diseased, `1` = Normal.

The image set is small (on the order of **72** augmented scans). Treat accuracy as a demo metric, not a clinical claim.

**Image gate (not a neural net):** rule-based checks so the binary CNN is not asked to label a selfie or a random photo. It is **not** a full modality detector: colour Doppler or unusual crops may be rejected; other grayscale medical images may still pass.

### AdaBoost (clinical labs)

| | |
|---|---|
| **Role** | Binary tabular classifier: `0` = Normal, `1` = Disease |
| **Library** | scikit-learn (`AdaBoostClassifier`, 100 estimators, balanced sample weights) |
| **Input** | Seven fields listed below, typed in **raw units** |
| **Processing** | Convert labs with `RAW_LAB_STATS` → apply `clinical_scaler.pkl` |
| **Output** | Class id, label, confidence, probability of both classes |
| **Training** | `ILP4SO.csv`, incomplete rows dropped. Stratified 80/20 split, seed 42. `StandardScaler` fit on the **training split only** (including Age). The CSV lab columns are **already z-scored**; they are **not** z-scored a second time during training. |
| **Artifacts** | `models/adaboost_model.pkl`, `models/clinical_scaler.pkl`, `models/clinical_meta.json` |

Last recorded snapshot in `clinical_meta.json`: **199** rows; test accuracy **0.90**; macro F1 **~0.86**; disease-class F1 **~0.78**. Retraining will overwrite these numbers.

---

## API and Backend Functionality

This project **does not expose HTTP endpoints**. There is no Flask/FastAPI server, no `/predict` URL, and no JWT/CORS layer.

The GUI and the training CLI call **in-process Python functions**. That is the “backend”:

| Module | Responsibility |
|---|---|
| `main.py` | Entry point; Windows Tcl/Tk path fix; start the GUI |
| `config.py` | Paths, image limits, lab ranges, label map, safety text |
| `image_gate.py` | File validation + ultrasound appearance check; grayscale conversion for the CNN |
| `training.py` | Train, load, and predict for both models |
| `train_models.py` | CLI wrapper around training only |
| `liver_gui.py` | Desktop UI and user-facing validation messages |
| `ui_theme.py` | Colours, icons, shared button styles |

Prediction functions used by the GUI:

| Function | Role |
|---|---|
| `load_or_train_efficientnet()` | Load CNN weights, or train if the checkpoint is missing |
| `load_or_train_adaboost()` | Load AdaBoost + scaler, or train if the pair is missing |
| `inspect_image_path(...)` | Technical image checks (used when the user picks a file) |
| `validate_image_file(...)` | Technical checks **plus** ultrasound appearance (used at inference) |
| `predict_from_image(...)` | Validated image → class, risk label, confidence |
| `predict_from_clinical(...)` | Raw labs → class, label, confidence, probabilities |

---

## Database and Data Management

There is **no database** (no SQLite, no PostgreSQL, no ORM). Data is files on disk.

| Store | What it is | How it is used |
|---|---|---|
| `Liver_steatosis/Diseased/` and `Normal/` | Training/inference image folders | ImageFolder class labels; user uploads are **not** written here |
| `ILP4SO.csv` | Clinical table (ILPD-style) | AdaBoost training only. Labs are pre-z-scored; Age is in years; `Outcome` is `0`/`1` (not original ILPD 1/2 coding) |
| `models/` | Saved weights, scaler, class names, metrics JSON | Loaded at startup; optional retrain overwrites them |
| GUI session | Preview image + last result | Memory only; closed when the app exits |

**Relationships (logical, not SQL):**

- Image class names in `class_names.json` must match the `Liver_steatosis` folder names used at training.
- AdaBoost and `clinical_scaler.pkl` are a **pair**. If the model exists without the scaler, the app retrains the clinical pipeline.
- GUI lab fields must match `CLINICAL_FEATURES` in `config.py` (same names as CSV columns).

No user table, no patient IDs, and no stored reports.

### Clinical inputs (what the user types)

| Field on screen | CSV column | Unit |
|---|---|---|
| Age | `Age` | years |
| Direct bilirubin | `Direct_Bilrubin` | mg/dL |
| Alkaline phosphatase (ALP) | `Alkaline_Phosphotase` | IU/L |
| ALT (SGPT) | `Alamine_Aminotransferase` | IU/L |
| AST (SGOT) | `Aspartate_Aminotransferase` | IU/L |
| Albumin | `Albumin` | g/dL |
| Albumin / globulin ratio | `Albumin_and_Globulin_Ratio` | — |

---

## Project Structure

```
Liver_Diagnosis_System/
├── main.py              # Launch the GUI
├── train_models.py      # Train models from the CLI
├── config.py            # Paths, features, ranges, image limits
├── training.py          # Train, load, predict
├── image_gate.py        # Image validation before the CNN
├── liver_gui.py         # Desktop interface
├── ui_theme.py          # Visual tokens and icons
├── requirements.txt     # Pinned Python dependencies
├── ILP4SO.csv           # Clinical dataset
├── Liver_steatosis/     # Ultrasound images (Diseased/, Normal/)
└── models/              # Saved checkpoints
    ├── efficientnet_liver_model.pth
    ├── class_names.json
    ├── adaboost_model.pkl
    ├── clinical_scaler.pkl
    └── clinical_meta.json
```

`main.py` only starts the app. Shared ML logic lives in `training.py` so the GUI and CLI cannot drift apart.

---

## Installation and Setup

### Prerequisites

- Windows, macOS, or Linux
- **Python 3.12** recommended (3.11 is fine if Tcl/Tk is installed)
- **Tcl/Tk** for the GUI. On Windows, repair Python and enable **tcl/tk and IDLE** if you see an `init.tcl` error
- Optional: NVIDIA GPU + CUDA for faster CNN training (CPU works)

### Environment and dependencies

No `.env` file. Paths come from `config.py` (project folder), not from the current working directory.

**Windows PowerShell**

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

Pinned packages include PyTorch, torchvision, scikit-learn, pandas, NumPy, Pillow, joblib, matplotlib, and CustomTkinter (`requirements.txt`).

### Database setup

**None.** Skip this step. Do not install or configure a database.

### Retrain models (optional)

```bash
python train_models.py
python train_models.py --clinical-only
python train_models.py --image-only
```

If you only want to train, you do not need to open the GUI.

---

## Usage Guide

1. Start with `python main.py` and wait until the splash screen finishes.
2. **Ultrasound**
   - Click the upload area or **Select image**.
   - Use a liver ultrasound (JPG / PNG / BMP / TIFF, max 25 MB).
   - A green “ready to analyse” caption means file + appearance checks passed.
   - A red caption or result card means the file was rejected (wrong type, damaged, too large/small, or not an ultrasound).
   - Click **Run analysis** to see EfficientNet-B0 output.
3. **Clinical labs**
   - Open **Clinical labs** in the sidebar.
   - Fill every field using the units and ranges on the form.
   - **Clear form** wipes the fields and the last result.
   - **Run analysis** scores the vector with AdaBoost.
4. Read the result as a **research prototype**, not a radiology or lab report. Follow the on-screen safety advice.

---

## Limitations

- Small ultrasound training set; real-world scans can look different from the cropped grayscale examples.
- Image gate is heuristic, not a certified modality detector.
- Clinical CSV uses processed (z-scored) labs; the GUI conversion depends on `RAW_LAB_STATS`.
- No model ensemble: an image result and a lab result are independent.
- No persistence of predictions.

**Expected evaluation outcome:** an examiner can install the app, run both paths, and explain how an ultrasound or a lab vector becomes a labelled score — including why this remains a prototype, not a clinical system.

---

## Disclaimer

For academic use only. Not a substitute for professional medical advice, diagnosis, or treatment. Do not start, stop, or change medication based on this application.
