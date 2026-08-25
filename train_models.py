"""Train models and save artifacts. Does not launch the GUI.

Examples:
    python train_models.py
    python train_models.py --clinical-only
    python train_models.py --image-only
"""

import argparse

from training import train_adaboost, train_efficientnet


def main():
    parser = argparse.ArgumentParser(description="Train Liver Diagnosis System models.")
    parser.add_argument(
        "--clinical-only",
        action="store_true",
        help="Train AdaBoost and the clinical scaler only.",
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Train EfficientNet only.",
    )
    args = parser.parse_args()
    if args.clinical_only and args.image_only:
        parser.error("Use only one of --clinical-only or --image-only.")

    print("\n=============================")
    print("   Liver Diagnosis Training")
    print("=============================\n")

    if args.clinical_only:
        train_adaboost()
    elif args.image_only:
        train_efficientnet()
    else:
        train_efficientnet()
        train_adaboost()

    print("Training complete. Models are ready for the GUI.")


if __name__ == "__main__":
    main()
