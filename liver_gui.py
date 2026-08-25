"""CustomTkinter desktop UI for the Liver Diagnosis System."""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image, ImageTk

from config import CLINICAL_FEATURES, DISCLAIMER, FEATURE_META, SAFETY_DISEASE, SAFETY_NORMAL
from training import (
    load_or_train_adaboost,
    load_or_train_efficientnet,
    predict_from_clinical,
    predict_from_image,
)
from image_gate import UnsupportedImageError, assess_ultrasound_image
from ui_theme import (
    ACCENT,
    ACCENT_HOVER,
    ACCENT_SOFT,
    BG,
    BORDER,
    BORDER_FOCUS,
    BREAKPOINT,
    CARD,
    DANGER,
    DANGER_SOFT,
    ELEVATED,
    FAINT,
    INSET,
    MUTED,
    OK,
    OK_SOFT,
    SIDEBAR,
    SIDEBAR_NARROW,
    SIDEBAR_WIDE,
    SURFACE,
    TEXT,
    TEXT_SECONDARY,
    TOPBAR,
    WARN,
    WHITE,
    center,
    empty_art,
    f,
    icon,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class NavItem(ctk.CTkFrame):
    def __init__(self, master, key: str, label: str, icon_name: str, command):
        super().__init__(master, fg_color="transparent", height=44, cursor="hand2")
        self.key = key
        self.label_text = label
        self.icon_name = icon_name
        self.command = command
        self.selected = False
        self.collapsed = False

        self.bar = ctk.CTkFrame(self, width=3, height=24, fg_color="transparent", corner_radius=2)
        self.bar.pack(side="left", padx=(8, 8))
        self.bar.pack_propagate(False)

        self.btn = ctk.CTkButton(
            self,
            text=label,
            image=icon(icon_name, 18, MUTED),
            compound="left",
            anchor="w",
            height=40,
            fg_color="transparent",
            hover_color=ELEVATED,
            text_color=MUTED,
            font=f(13),
            command=self.command,
        )
        self.btn.pack(side="left", fill="x", expand=True, padx=(0, 10))

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        color = ACCENT if selected else MUTED
        self.bar.configure(fg_color=ACCENT if selected else "transparent")
        self.btn.configure(
            image=icon(self.icon_name, 18, color),
            text_color=WHITE if selected else MUTED,
            fg_color=ACCENT_SOFT if selected else "transparent",
            font=f(13, "bold" if selected else "normal"),
        )
        self._apply_collapse()

    def set_collapsed(self, collapsed: bool) -> None:
        self.collapsed = collapsed
        self._apply_collapse()

    def _apply_collapse(self) -> None:
        self.btn.configure(text="" if self.collapsed else self.label_text)


class FormField(ctk.CTkFrame):
    def __init__(self, master, feature: str, meta: dict, on_submit=None):
        super().__init__(master, fg_color="transparent")
        self.feature = feature
        self.meta = meta
        self.grid_columnconfigure(0, weight=1)

        unit = f" · {meta['unit']}" if meta["unit"] else ""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text=meta["label"],
            font=f(12, "bold"),
            text_color=TEXT_SECONDARY,
            anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            header,
            text=f"{meta['min']:g}–{meta['max']:g}{unit}",
            font=f(11),
            text_color=FAINT,
            anchor="e",
        ).pack(side="right")

        self.entry = ctk.CTkEntry(
            self,
            height=40,
            corner_radius=10,
            border_width=1,
            border_color=BORDER,
            fg_color=INSET,
            text_color=TEXT,
            font=f(14),
            placeholder_text=f"Enter {meta['label'].lower()}",
        )
        self.entry.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.entry.bind("<FocusIn>", lambda _e: self._focus(True))
        self.entry.bind("<FocusOut>", lambda _e: self._focus(False))
        if on_submit:
            self.entry.bind("<Return>", lambda _e: on_submit())

        self.error = ctk.CTkLabel(
            self, text="", font=f(11), text_color=DANGER, anchor="w"
        )
        self.error.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.error.grid_remove()

    def _focus(self, active: bool) -> None:
        if self.error.cget("text"):
            return
        self.entry.configure(border_color=BORDER_FOCUS if active else BORDER)

    def get_raw(self) -> str:
        return self.entry.get().strip()

    def set_error(self, message: str | None) -> None:
        if message:
            self.error.configure(text=message)
            self.error.grid()
            self.entry.configure(border_color=DANGER)
        else:
            self.error.configure(text="")
            self.error.grid_remove()
            self.entry.configure(border_color=BORDER)

    def clear(self) -> None:
        self.entry.delete(0, "end")
        self.set_error(None)


class ResultCard(ctk.CTkFrame):
    def __init__(self, master, empty_title: str, empty_body: str, art_kind: str):
        super().__init__(
            master, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER
        )
        self.empty_title = empty_title
        self.empty_body = empty_body
        self.art_kind = art_kind
        self._pulse_job = None
        self._pulse_dir = 1

        pad = ctk.CTkFrame(self, fg_color="transparent")
        pad.pack(fill="both", expand=True, padx=24, pady=22)

        eyebrow = ctk.CTkFrame(pad, fg_color="transparent")
        eyebrow.pack(fill="x")
        ctk.CTkLabel(
            eyebrow,
            text="RESULT",
            font=f(11, "bold"),
            text_color=FAINT,
            anchor="w",
        ).pack(side="left")
        self.pill = ctk.CTkLabel(
            eyebrow,
            text="Idle",
            font=f(11, "bold"),
            text_color=MUTED,
            fg_color=INSET,
            corner_radius=10,
            padx=10,
            pady=3,
        )
        self.pill.pack(side="right")

        self.icon_label = ctk.CTkLabel(pad, text="", image=empty_art(art_kind, 108))
        self.icon_label.pack(pady=(28, 12))

        self.title = ctk.CTkLabel(
            pad, text=empty_title, font=f(20, "bold"), text_color=TEXT, wraplength=320
        )
        self.title.pack()
        self.body = ctk.CTkLabel(
            pad,
            text=empty_body,
            font=f(13),
            text_color=MUTED,
            wraplength=320,
            justify="center",
        )
        self.body.pack(pady=(6, 18))

        self.meter_wrap = ctk.CTkFrame(pad, fg_color="transparent")
        self.meter_wrap.pack(fill="x")
        head = ctk.CTkFrame(self.meter_wrap, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text="Confidence", font=f(12), text_color=MUTED, anchor="w").pack(
            side="left"
        )
        self.conf_value = ctk.CTkLabel(
            head, text="—", font=f(12, "bold"), text_color=TEXT, anchor="e"
        )
        self.conf_value.pack(side="right")
        self.bar = ctk.CTkProgressBar(
            self.meter_wrap,
            height=10,
            corner_radius=5,
            progress_color=ACCENT,
            fg_color=INSET,
        )
        self.bar.pack(fill="x", pady=(8, 0))
        self.bar.set(0)

        self.prob_wrap = ctk.CTkFrame(pad, fg_color="transparent")
        self.prob_wrap.pack(fill="x", pady=(16, 0))
        self.p_normal = self._prob_row(self.prob_wrap, "Normal", OK)
        self.p_disease = self._prob_row(self.prob_wrap, "Disease", DANGER)

        self.safety_wrap = ctk.CTkFrame(
            pad,
            fg_color=INSET,
            corner_radius=10,
            border_width=1,
            border_color=BORDER,
        )
        ctk.CTkLabel(
            self.safety_wrap,
            text="SAFETY ADVICE",
            font=f(11, "bold"),
            text_color=FAINT,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 0))
        self.safety = ctk.CTkLabel(
            self.safety_wrap,
            text="",
            font=f(12),
            text_color=TEXT_SECONDARY,
            wraplength=300,
            justify="left",
            anchor="w",
        )
        self.safety.pack(fill="x", padx=12, pady=(4, 12))

        self.footnote = ctk.CTkLabel(
            pad,
            text="Academic demonstration only. Not a clinical diagnosis.",
            font=f(11),
            text_color=FAINT,
            wraplength=320,
            justify="center",
        )
        self.footnote.pack(side="bottom", pady=(18, 0))
        self.meter_wrap.pack_forget()
        self.prob_wrap.pack_forget()
        self.show_idle()

    def _prob_row(self, master, label: str, color: str):
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.pack(fill="x", pady=5)
        top = ctk.CTkFrame(row, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=label, font=f(12), text_color=MUTED, anchor="w").pack(side="left")
        value = ctk.CTkLabel(top, text="—", font=f(12, "bold"), text_color=TEXT)
        value.pack(side="right")
        bar = ctk.CTkProgressBar(
            row, height=6, corner_radius=3, progress_color=color, fg_color=INSET
        )
        bar.pack(fill="x", pady=(4, 0))
        bar.set(0)
        return {"row": row, "value": value, "bar": bar}

    def _stop_pulse(self):
        if self._pulse_job:
            self.after_cancel(self._pulse_job)
            self._pulse_job = None

    def _pulse(self):
        cur = self.bar.get()
        nxt = cur + 0.035 * self._pulse_dir
        if nxt >= 0.86:
            self._pulse_dir = -1
            nxt = 0.86
        elif nxt <= 0.12:
            self._pulse_dir = 1
            nxt = 0.12
        self.bar.set(nxt)
        self._pulse_job = self.after(30, self._pulse)

    def _set_pill(self, text: str, fg: str, bg: str):
        self.pill.configure(text=text, text_color=fg, fg_color=bg)

    def show_idle(self):
        self._stop_pulse()
        self._set_pill("Idle", MUTED, INSET)
        self.icon_label.configure(image=empty_art(self.art_kind, 108), text="")
        self.title.configure(text=self.empty_title, text_color=TEXT)
        self.body.configure(text=self.empty_body)
        self.bar.set(0)
        self.bar.configure(progress_color=ACCENT)
        self.conf_value.configure(text="—")
        self.meter_wrap.pack_forget()
        self.prob_wrap.pack_forget()
        self.safety_wrap.pack_forget()

    def show_loading(self, message: str):
        self._stop_pulse()
        self._set_pill("Running", ACCENT, ACCENT_SOFT)
        self.icon_label.configure(image=icon("spark", 36, ACCENT), text="")
        self.title.configure(text="Analysing", text_color=ACCENT)
        self.body.configure(text=message)
        self.bar.configure(progress_color=ACCENT)
        self.conf_value.configure(text="…")
        self.meter_wrap.pack(fill="x")
        self.prob_wrap.pack_forget()
        self.safety_wrap.pack_forget()
        self._pulse_dir = 1
        self._pulse()

    def show_error(self, message: str):
        self._stop_pulse()
        self._set_pill("Failed", DANGER, DANGER_SOFT)
        self.icon_label.configure(image=icon("alert", 36, DANGER), text="")
        self.title.configure(text="Could not complete", text_color=DANGER)
        self.body.configure(text=message)
        self.bar.set(0)
        self.conf_value.configure(text="—")
        self.meter_wrap.pack_forget()
        self.prob_wrap.pack_forget()
        self.safety_wrap.pack_forget()

    def show_success(
        self,
        title: str,
        body: str,
        confidence: float,
        positive: bool,
        probabilities: dict | None = None,
    ):
        self._stop_pulse()
        color = DANGER if positive else OK
        soft = DANGER_SOFT if positive else OK_SOFT
        self._set_pill("Complete", color, soft)
        self.icon_label.configure(
            image=icon("alert" if positive else "check", 40, color), text=""
        )
        self.title.configure(text=title, text_color=color)
        self.body.configure(text=body)
        self.bar.set(max(0.0, min(1.0, confidence)))
        self.bar.configure(progress_color=color)
        self.conf_value.configure(text=f"{confidence:.0%}")
        self.meter_wrap.pack(fill="x")
        if probabilities:
            self.prob_wrap.pack(fill="x", pady=(16, 0))
            n = probabilities.get(0, 0.0)
            d = probabilities.get(1, 0.0)
            self.p_normal["bar"].set(n)
            self.p_normal["value"].configure(text=f"{n:.0%}")
            self.p_disease["bar"].set(d)
            self.p_disease["value"].configure(text=f"{d:.0%}")
        else:
            self.prob_wrap.pack_forget()
        self.safety.configure(text=SAFETY_DISEASE if positive else SAFETY_NORMAL)
        self.safety_wrap.pack(fill="x", pady=(14, 0))


class Banner(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=DANGER_SOFT, corner_radius=10, border_width=1, border_color=DANGER)
        self.label = ctk.CTkLabel(
            self,
            text="",
            image=icon("alert", 16, DANGER),
            compound="left",
            font=f(12),
            text_color=TEXT,
            wraplength=560,
            justify="left",
            anchor="w",
        )
        self.label.pack(fill="x", padx=12, pady=8)

    def hide(self):
        self.pack_forget()


class LiverDiagnosisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.cnn_model = None
        self.class_names = None
        self.adaboost_model = None
        self.scaler = None
        self.filepath = None
        self._source_pil = None
        self._preview_photo = None
        self._image_supported = False
        self._image_reject_reason = ""
        self._busy = False
        self._narrow = False
        self._icon_only = False
        self.page = "scan"
        self.fields: dict[str, FormField] = {}

        self.title("Liver Diagnosis System")
        self.configure(fg_color=BG)
        self._set_window_icon()
        self._show_splash()
        self.after(80, self._load_cnn)

    def _show_splash(self):
        self.resizable(False, False)
        center(self, 560, 380)
        self.splash = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.splash.pack(fill="both", expand=True)

        card = ctk.CTkFrame(
            self.splash, fg_color=CARD, corner_radius=18, border_width=1, border_color=BORDER
        )
        card.pack(expand=True, fill="both", padx=26, pady=26)

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=28, pady=(28, 8))
        ctk.CTkLabel(head, text="", image=icon("mark", 40)).pack(side="left")
        titles = ctk.CTkFrame(head, fg_color="transparent")
        titles.pack(side="left", padx=12)
        ctk.CTkLabel(
            titles, text="Liver Diagnosis System", font=f(20, "bold"), text_color=WHITE, anchor="w"
        ).pack(anchor="w")
        ctk.CTkLabel(
            titles, text="Preparing the workspace", font=f(13), text_color=MUTED, anchor="w"
        ).pack(anchor="w")

        self.steps = {}
        for key, label in (
            ("cnn", "Ultrasound model  ·  EfficientNet-B0"),
            ("ada", "Clinical model  ·  AdaBoost + scaler"),
            ("ui", "Opening interface"),
        ):
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=28, pady=6)
            mark = ctk.CTkLabel(row, text="", image=icon("dot", 10, FAINT), width=18)
            mark.pack(side="left")
            text = ctk.CTkLabel(row, text=label, font=f(13), text_color=MUTED, anchor="w")
            text.pack(side="left", padx=8)
            self.steps[key] = (mark, text)

        self.splash_progress = ctk.CTkProgressBar(
            card, height=6, corner_radius=3, progress_color=ACCENT, fg_color=INSET
        )
        self.splash_progress.pack(fill="x", padx=28, pady=(18, 8))
        self.splash_progress.set(0.08)
        self.splash_status = ctk.CTkLabel(card, text="Starting…", font=f(12), text_color=FAINT)
        self.splash_status.pack(pady=(0, 8))

    def _set_step(self, key: str, state: str):
        mark, text = self.steps[key]
        if state == "active":
            mark.configure(image=icon("spark", 14, ACCENT))
            text.configure(text_color=TEXT)
        elif state == "done":
            mark.configure(image=icon("check", 16, OK))
            text.configure(text_color=TEXT_SECONDARY)
        else:
            mark.configure(image=icon("dot", 10, FAINT))
            text.configure(text_color=MUTED)

    def _load_cnn(self):
        self._set_step("cnn", "active")
        self.splash_status.configure(text="Loading ultrasound model…")
        self.splash_progress.set(0.28)
        self.update_idletasks()
        try:
            self.cnn_model, self.class_names = load_or_train_efficientnet()
        except Exception as exc:
            self.splash_status.configure(
                text=f"Could not load the image model:\n{exc}", text_color=DANGER
            )
            return
        self._set_step("cnn", "done")
        self.after(80, self._load_adaboost)

    def _load_adaboost(self):
        self._set_step("ada", "active")
        self.splash_status.configure(text="Loading clinical model…")
        self.splash_progress.set(0.72)
        self.update_idletasks()
        try:
            self.adaboost_model, self.scaler, _meta = load_or_train_adaboost()
        except Exception as exc:
            self.splash_status.configure(
                text=f"Could not load the clinical model:\n{exc}", text_color=DANGER
            )
            return
        self._set_step("ada", "done")
        self._set_step("ui", "active")
        self.splash_progress.set(1.0)
        self.splash_status.configure(text="Opening workspace…")
        self.after(120, self._open_workspace)

    def _open_workspace(self):
        self._set_step("ui", "done")
        self.splash.destroy()
        self.resizable(True, True)
        self.minsize(980, 680)
        center(self, 1280, 820)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_workspace()
        self.bind("<Configure>", self._on_resize)
        self._goto("scan")

    def _set_window_icon(self):
        try:
            pil = icon("mark", 32)._light_image  # noqa: SLF001
            self._wm_icon = ImageTk.PhotoImage(pil)
            self.iconphoto(True, self._wm_icon)
        except Exception:
            pass

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, fg_color=SIDEBAR, width=SIDEBAR_WIDE, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.sidebar.grid_propagate(False)

        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=16, pady=(22, 28))
        ctk.CTkLabel(brand, text="", image=icon("mark", 34)).pack(side="left")
        self.brand_text = ctk.CTkFrame(brand, fg_color="transparent")
        self.brand_text.pack(side="left", padx=10)
        ctk.CTkLabel(
            self.brand_text, text="LDS", font=f(16, "bold"), text_color=WHITE, anchor="w"
        ).pack(anchor="w")
        ctk.CTkLabel(
            self.brand_text, text="Clinical AI", font=f(11), text_color=FAINT, anchor="w"
        ).pack(anchor="w")

        self.nav_scan = NavItem(self.sidebar, "scan", "Ultrasound", "scan", lambda: self._goto("scan"))
        self.nav_labs = NavItem(self.sidebar, "labs", "Clinical labs", "labs", lambda: self._goto("labs"))
        self.nav_scan.pack(fill="x", padx=8, pady=2)
        self.nav_labs.pack(fill="x", padx=8, pady=2)

        foot = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        foot.pack(side="bottom", fill="x", padx=16, pady=18)
        self.side_status = ctk.CTkLabel(
            foot,
            text="  Models ready",
            image=icon("dot", 10, OK),
            compound="left",
            font=f(12),
            text_color=MUTED,
            anchor="w",
        )
        self.side_status.pack(anchor="w")
        self.side_ver = ctk.CTkLabel(
            foot, text="Research build", font=f(11), text_color=FAINT, anchor="w"
        )
        self.side_ver.pack(anchor="w", pady=(4, 0))

    def _build_workspace(self):
        self.workspace = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.workspace.grid(row=0, column=1, sticky="nsew")
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(self.workspace, fg_color=TOPBAR, height=72, corner_radius=0)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_propagate(False)
        top.grid_columnconfigure(0, weight=1)

        titles = ctk.CTkFrame(top, fg_color="transparent")
        titles.grid(row=0, column=0, sticky="w", padx=28, pady=14)
        self.page_kicker = ctk.CTkLabel(
            titles, text="WORKSPACE", font=f(11, "bold"), text_color=FAINT, anchor="w"
        )
        self.page_kicker.pack(anchor="w")
        self.page_title = ctk.CTkLabel(
            titles, text="Ultrasound analysis", font=f(20, "bold"), text_color=WHITE, anchor="w"
        )
        self.page_title.pack(anchor="w")

        self.model_chip = ctk.CTkLabel(
            top,
            text="  EfficientNet-B0",
            image=icon("spark", 14, ACCENT),
            compound="left",
            font=f(12),
            text_color=TEXT_SECONDARY,
            fg_color=SURFACE,
            corner_radius=12,
            padx=12,
            pady=6,
        )
        self.model_chip.grid(row=0, column=1, sticky="e", padx=24)

        self.body = ctk.CTkFrame(self.workspace, fg_color=BG)
        self.body.grid(row=1, column=0, sticky="nsew", padx=20, pady=16)
        self.body.grid_columnconfigure(0, weight=3)
        self.body.grid_columnconfigure(1, weight=2)
        self.body.grid_rowconfigure(0, weight=1)

        self.page_scan = ctk.CTkFrame(self.body, fg_color="transparent")
        self.page_labs = ctk.CTkFrame(self.body, fg_color="transparent")
        for page in (self.page_scan, self.page_labs):
            page.grid(row=0, column=0, columnspan=2, sticky="nsew")
            page.grid_columnconfigure(0, weight=3)
            page.grid_columnconfigure(1, weight=2)
            page.grid_rowconfigure(0, weight=1)

        self._build_scan_page()
        self._build_labs_page()

        footer = ctk.CTkFrame(self.workspace, fg_color=TOPBAR, height=44, corner_radius=0)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_propagate(False)
        ctk.CTkLabel(
            footer, text=DISCLAIMER, font=f(11), text_color=MUTED, wraplength=980
        ).pack(expand=True, padx=20)

    def _card(self, master) -> ctk.CTkFrame:
        return ctk.CTkFrame(
            master, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER
        )

    def _build_scan_page(self):
        left = self._card(self.page_scan)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_rowconfigure(3, weight=1)
        self.scan_input = left

        ctk.CTkLabel(
            left, text="INPUT", font=f(11, "bold"), text_color=FAINT, anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(18, 0))
        ctk.CTkLabel(
            left,
            text="Liver ultrasound",
            font=f(18, "bold"),
            text_color=TEXT,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", padx=22, pady=(2, 0))
        ctk.CTkLabel(
            left,
            text="Drop-in a scan to classify Diseased vs Normal. Supported: JPG, PNG, BMP, TIFF.",
            font=f(13),
            text_color=MUTED,
            wraplength=560,
            justify="left",
            anchor="w",
        ).grid(row=2, column=0, sticky="w", padx=22, pady=(4, 12))

        self.preview_frame = ctk.CTkFrame(
            left, fg_color=INSET, corner_radius=14, cursor="hand2"
        )
        self.preview_frame.grid(row=3, column=0, sticky="nsew", padx=22, pady=(0, 8))
        self.preview_frame.bind("<Button-1>", lambda _e: self.select_image())
        self.preview_frame.bind("<Configure>", lambda _e: self._render_preview())

        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="  Click to choose an image",
            image=icon("image", 28, MUTED),
            compound="top",
            font=f(13),
            text_color=FAINT,
        )
        self.preview_label.pack(expand=True)
        self.preview_label.bind("<Button-1>", lambda _e: self.select_image())

        self.file_caption = ctk.CTkLabel(
            left, text="No file selected", font=f(12), text_color=FAINT, anchor="w"
        )
        self.file_caption.grid(row=4, column=0, sticky="w", padx=22, pady=(0, 10))

        actions = ctk.CTkFrame(left, fg_color="transparent")
        actions.grid(row=5, column=0, sticky="ew", padx=22, pady=(0, 20))
        self.select_btn = ctk.CTkButton(
            actions,
            text="  Select image",
            image=icon("image", 16, TEXT),
            compound="left",
            height=42,
            width=160,
            corner_radius=10,
            fg_color=ELEVATED,
            hover_color=BORDER,
            border_width=1,
            border_color=BORDER,
            text_color=TEXT,
            font=f(13, "bold"),
            command=self.select_image,
        )
        self.select_btn.pack(side="left")
        self.predict_img_btn = ctk.CTkButton(
            actions,
            text="  Run analysis",
            image=icon("spark", 16, WHITE),
            compound="left",
            height=42,
            width=168,
            corner_radius=10,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            font=f(13, "bold"),
            command=self.predict_from_image_clicked,
        )
        self.predict_img_btn.pack(side="right")

        self.img_result = ResultCard(
            self.page_scan,
            "No scan analysed yet",
            "Select an ultrasound image, then run analysis to see the model output.",
            "scan",
        )
        self.img_result.grid(row=0, column=1, sticky="nsew")

    def _build_labs_page(self):
        left = self._card(self.page_labs)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.labs_input = left

        ctk.CTkLabel(
            left, text="INPUT", font=f(11, "bold"), text_color=FAINT, anchor="w"
        ).pack(fill="x", padx=22, pady=(18, 0))
        ctk.CTkLabel(
            left, text="Laboratory values", font=f(18, "bold"), text_color=TEXT, anchor="w"
        ).pack(fill="x", padx=22, pady=(2, 0))
        ctk.CTkLabel(
            left,
            text="Enter raw clinical units. Each field is range-checked before scoring.",
            font=f(13),
            text_color=MUTED,
            wraplength=600,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=22, pady=(4, 12))

        self.labs_banner_host = ctk.CTkFrame(left, fg_color="transparent")
        self.labs_banner_host.pack(fill="x", padx=22)
        self.labs_banner = Banner(self.labs_banner_host)

        form = ctk.CTkFrame(left, fg_color="transparent")
        form.pack(fill="both", expand=True, padx=16, pady=(4, 8))
        form.grid_columnconfigure((0, 1), weight=1)

        for i, feat in enumerate(CLINICAL_FEATURES):
            row, col = divmod(i, 2)
            field = FormField(
                form, feat, FEATURE_META[feat], on_submit=self.predict_from_data_clicked
            )
            field.grid(row=row, column=col, sticky="ew", padx=8, pady=8)
            self.fields[feat] = field

        actions = ctk.CTkFrame(left, fg_color="transparent")
        actions.pack(fill="x", padx=22, pady=(4, 20))
        self.clear_btn = ctk.CTkButton(
            actions,
            text="Clear form",
            height=42,
            width=130,
            corner_radius=10,
            fg_color=ELEVATED,
            hover_color=BORDER,
            border_width=1,
            border_color=BORDER,
            text_color=TEXT,
            font=f(13, "bold"),
            command=self._clear_clinical_form,
        )
        self.clear_btn.pack(side="left")
        self.predict_data_btn = ctk.CTkButton(
            actions,
            text="  Run analysis",
            image=icon("spark", 16, WHITE),
            compound="left",
            height=42,
            width=168,
            corner_radius=10,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            font=f(13, "bold"),
            command=self.predict_from_data_clicked,
        )
        self.predict_data_btn.pack(side="right")

        self.data_result = ResultCard(
            self.page_labs,
            "No labs analysed yet",
            "Fill every field, then run analysis to see Normal vs Disease probabilities.",
            "labs",
        )
        self.data_result.grid(row=0, column=1, sticky="nsew")

    def _goto(self, page: str):
        self.page = page
        self.nav_scan.set_selected(page == "scan")
        self.nav_labs.set_selected(page == "labs")
        if page == "scan":
            self.page_title.configure(text="Ultrasound analysis")
            self.page_kicker.configure(text="IMAGING")
            self.model_chip.configure(text="  EfficientNet-B0")
            self.page_scan.tkraise()
        else:
            self.page_title.configure(text="Clinical laboratory analysis")
            self.page_kicker.configure(text="BIOMARKERS")
            self.model_chip.configure(text="  AdaBoost")
            self.page_labs.tkraise()

    def _on_resize(self, event):
        if event.widget is not self:
            return
        narrow = event.width < BREAKPOINT
        icon_only = event.width < 1040
        if narrow != self._narrow:
            self._narrow = narrow
            if narrow:
                self.scan_input.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=0)
                self.img_result.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=(12, 0))
                self.labs_input.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=0)
                self.data_result.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=(12, 0))
                for page in (self.page_scan, self.page_labs):
                    page.grid_columnconfigure(0, weight=1)
                    page.grid_rowconfigure(0, weight=1)
                    page.grid_rowconfigure(1, weight=1)
            else:
                self.scan_input.grid(row=0, column=0, columnspan=1, sticky="nsew", padx=(0, 12), pady=0)
                self.img_result.grid(row=0, column=1, columnspan=1, sticky="nsew", padx=0, pady=0)
                self.labs_input.grid(row=0, column=0, columnspan=1, sticky="nsew", padx=(0, 12), pady=0)
                self.data_result.grid(row=0, column=1, columnspan=1, sticky="nsew", padx=0, pady=0)
                for page in (self.page_scan, self.page_labs):
                    page.grid_columnconfigure(0, weight=3)
                    page.grid_columnconfigure(1, weight=2)
                    page.grid_rowconfigure(1, weight=0)
        if icon_only != self._icon_only:
            self._icon_only = icon_only
            self.sidebar.configure(width=SIDEBAR_NARROW if icon_only else SIDEBAR_WIDE)
            if icon_only:
                self.brand_text.pack_forget()
                self.side_status.configure(text="")
                self.side_ver.configure(text="")
            else:
                self.brand_text.pack(side="left", padx=10)
                self.side_status.configure(text="  Models ready")
                self.side_ver.configure(text="Research build")
            self.nav_scan.set_collapsed(icon_only)
            self.nav_labs.set_collapsed(icon_only)

    def _render_preview(self):
        if self._source_pil is None:
            return
        fw = max(self.preview_frame.winfo_width() - 28, 160)
        fh = max(self.preview_frame.winfo_height() - 28, 120)
        img = self._source_pil.copy()
        img.thumbnail((fw, fh))
        photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self._preview_photo = photo
        self.preview_label.configure(image=photo, text="")

    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self._source_pil = img
            self.filepath = path
            self._render_preview()
            ok, reason = assess_ultrasound_image(img)
            self._image_supported = ok
            self._image_reject_reason = reason
            if ok:
                self.file_caption.configure(text=Path(path).name, text_color=TEXT_SECONDARY)
                self.img_result.show_idle()
            else:
                self.file_caption.configure(
                    text=f"{Path(path).name}  ·  not a supported ultrasound",
                    text_color=WARN,
                )
                self.img_result.show_error(reason)
        except Exception:
            self.filepath = None
            self._source_pil = None
            self.img_result.show_error("The selected file could not be opened as an image.")

    def predict_from_image_clicked(self):
        if not self.filepath:
            self.img_result.show_error("Select an ultrasound image before running analysis.")
            return
        if not self._image_supported:
            self.img_result.show_error(
                self._image_reject_reason or "This does not appear to be a liver ultrasound."
            )
            return
        if self._busy:
            return
        self._set_busy(True)
        self.img_result.show_loading("EfficientNet-B0 is scoring the selected scan.")
        path = self.filepath
        threading.Thread(target=self._image_worker, args=(path,), daemon=True).start()

    def _image_worker(self, path):
        try:
            cls, risk, conf = predict_from_image(path, self.cnn_model, self.class_names)
            self.after(0, lambda: self._show_image_result(cls, risk, conf))
        except UnsupportedImageError as exc:
            self.after(0, lambda e=exc: self._reject_image(str(e)))
        except Exception as exc:
            self.after(0, lambda e=exc: self._fail(e, "img"))

    def _reject_image(self, message: str):
        self.img_result.show_error(message)
        self._set_busy(False)

    def _show_image_result(self, cls, risk, conf):
        from config import IMAGE_LOW_CONFIDENCE

        positive = cls.lower() == "diseased"
        title = "Disease indicated" if positive else "Normal pattern"
        body = f"{cls}  ·  {risk}\nThis is a model score, not a radiology report."
        if conf < IMAGE_LOW_CONFIDENCE:
            body += (
                "\nConfidence is moderate: this scan differs from the small training set "
                "(for example overlays or a different crop)."
            )
        self.img_result.show_success(title, body, conf, positive)
        self._set_busy(False)

    def _clear_clinical_form(self):
        for field in self.fields.values():
            field.clear()
        self.labs_banner.pack_forget()
        self.data_result.show_idle()

    def predict_from_data_clicked(self):
        if self._busy:
            return
        vals = {}
        has_error = False
        for key, field in self.fields.items():
            raw = field.get_raw()
            meta = FEATURE_META[key]
            if not raw:
                field.set_error("Required")
                has_error = True
                continue
            try:
                value = float(raw)
            except ValueError:
                field.set_error("Enter a number")
                has_error = True
                continue
            if value < meta["min"] or value > meta["max"]:
                unit = f" {meta['unit']}" if meta["unit"] else ""
                field.set_error(f"Must be {meta['min']:g}–{meta['max']:g}{unit}")
                has_error = True
                continue
            field.set_error(None)
            vals[key] = value

        if has_error:
            self.labs_banner.label.configure(text="  Fix the highlighted fields, then run again.")
            self.labs_banner.pack(fill="x", pady=(0, 8))
            return
        self.labs_banner.pack_forget()
        self._set_busy(True)
        self.data_result.show_loading("AdaBoost is scoring the laboratory vector.")
        threading.Thread(target=self._clinical_worker, args=(vals,), daemon=True).start()

    def _clinical_worker(self, vals):
        try:
            pred, label, confidence, prob_map = predict_from_clinical(
                vals, self.adaboost_model, self.scaler
            )
            self.after(0, lambda: self._show_clinical_result(pred, label, confidence, prob_map))
        except Exception as exc:
            self.after(0, lambda e=exc: self._fail(e, "data"))

    def _show_clinical_result(self, pred, label, confidence, prob_map):
        positive = pred != 0
        title = "Disease indicated" if positive else "Normal pattern"
        body = f"{label}  ·  AdaBoost classifier"
        self.data_result.show_success(title, body, confidence, positive, probabilities=prob_map)
        self._set_busy(False)

    def _fail(self, exc, which: str):
        print(f"Prediction error: {exc!r}")
        panel = self.img_result if which == "img" else self.data_result
        panel.show_error("Something went wrong while scoring. Check the input and try again.")
        self._set_busy(False)

    def _set_busy(self, busy: bool):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for widget in (
            self.predict_img_btn,
            self.predict_data_btn,
            self.select_btn,
            self.clear_btn,
        ):
            widget.configure(state=state)
        self.nav_scan.btn.configure(state=state)
        self.nav_labs.btn.configure(state=state)


def run_app():
    app = LiverDiagnosisApp()
    app.mainloop()
