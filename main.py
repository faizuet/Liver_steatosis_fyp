"""Launch the Liver Diagnosis System desktop GUI.

Models are loaded (or trained if missing) from the splash screen.
Use train_models.py if you only want to train and save artifacts.
"""

import os
import sys
from pathlib import Path


def _ensure_tcl_tk() -> None:
    """Point Tkinter at the base interpreter's Tcl/Tk library (Windows venv fix)."""
    base = Path(getattr(sys, "base_prefix", sys.prefix))
    tcl = base / "tcl" / "tcl8.6"
    tk = base / "tcl" / "tk8.6"
    if tcl.is_dir() and "TCL_LIBRARY" not in os.environ:
        os.environ["TCL_LIBRARY"] = str(tcl)
    if tk.is_dir() and "TK_LIBRARY" not in os.environ:
        os.environ["TK_LIBRARY"] = str(tk)


_ensure_tcl_tk()

from liver_gui import run_app  # noqa: E402

if __name__ == "__main__":
    print("\n=============================")
    print("   Liver Diagnosis System")
    print("=============================\n")
    print("Starting application...\n")
    try:
        run_app()
    except Exception as exc:
        if "init.tcl" in str(exc) or type(exc).__name__ == "TclError":
            print(
                "Tkinter could not start because Tcl/Tk is missing from this Python "
                "install.\nRepair Python 3.12 and enable the 'tcl/tk and IDLE' "
                "optional feature, then run this command again."
            )
        raise
