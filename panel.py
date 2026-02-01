# -*- coding: utf-8 -*-
# FILE: panel.py | version: 2026-01-18.panel_v6_inference_lab
#
# Tkinter GUI for Domino AI pipeline
# Adds: Inference Lab tab (IM4.2) to automate:
#   - Collect strategic INF1 dataset via evaluate.py --infer_out_dir (open-loop)
#   - Train inference model via tools/infer_tool.py
#   - Run A/B eval on two fixed seeds with DOMINO_INFER=0/1 and show deltas
#
# Patch (IM4.3 GUI):
# - Add "Train Pair (self+strat) + Install Ensemble" workflow (no manual copying required).
#
# Design goals:
# - No copy/paste of manifests or models
# - Stable env handling (DOMINO_* read by Rust OnceLock per process)
# - Smart logging for A/B (show summary, not noise)

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox


ROOT = Path(__file__).resolve().parent
FACTORY_MIN = ROOT / "factory_min.py"
EVAL = ROOT / "evaluate.py"
INFER_TOOL = ROOT / "tools" / "infer_tool.py"
SETTINGS_PATH = ROOT / "panel_settings.json"

# Patterns
PROGRESS_RE = re.compile(r"^PROGRESS\s+(.*)$")
KV_RE = re.compile(r"(\w+)=([-\w\.]+)")
EVAL_JSON_RE = re.compile(r'^\{.*"results"\s*:\s*\{.*\}.*\}$')
EVAL_PROGRESS_RE = re.compile(r"^\[eval\]\s+progress\s+matches_done=(\d+)/(\d+)\s+mps=([0-9\.]+)")
INFER_JSON_RE = re.compile(r'^\{.*"op"\s*:\s*"infer_out_(init|done)".*\}$')


HELP_TEXTS = {
    "cycle": """Cycle Mode Help
═══════════════════════════
Cycle runs: generate → check_shards → train → eval

Train Modes:
• both: train policy/value + spike (legacy)
• pv_only: train policy/value only (no spike loss)
• spike_only: train spike head only (freeze PV/trunk)

Replay:
• replay_last_n: trains on last N manifests in same out_dir

ISMCTS Knobs:
• opp/me_mix_greedy: opponent/self greedy mixing
• leaf_value_weight: neural value weight at leaves
• gift_penalty_weight: penalty for gifting moves
• pessimism_alpha_max: pessimistic value estimation
""",
    "data": """Data Generation Help
═══════════════════════════
Generate shards only (no train/eval).

Modes:
• selfplay: standard self-play generation
• endgame_mine: mine endgame positions (record-only)

Use endgame_mine to collect specialized endgame data.
""",
    "eval": """Evaluation Help
═══════════════════════════
Run evaluation matches against opponent.

• me: your policy (rust, ai, etc.)
• opp: opponent policy (strategic, greedy, ...)
• level: AI difficulty level (quick, ...)
• jobs: parallel match workers
""",
    "infer": """Inference Lab Help (IM4.2)
═══════════════════════════
This tab automates the manual workflow:

1) Collect (Open-loop):
   - Runs evaluate.py with opp=strategic, jobs=1
   - Forces DOMINO_INFER=0 and DOMINO_AVOID_BETA=0
   - Writes INF1 files + infer_strat_*.manifest.json

2) Train:
   - Runs tools/infer_tool.py train on the chosen manifest
   - Optionally copies output to inference_model.json

3) A/B Magic:
   - Runs 4 evals: (seed1, seed2) × (INFER=0, INFER=1)
   - Shows side-by-side metrics and deltas.
""",
    "diag": """Diagnostics Help
═══════════════════════════
Use this tab before long runs.

• Run Tests: quick pytest sanity check
• Run Gates: full conformance + gates + bench + train-smoke
• Smoke Cycle: tiny end-to-end cycle test
"""
}


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    out_dir: str
    matches: int
    det: int
    think_ms: int
    temp: float
    mode: str
    end_close: int
    end_bone: int
    end_hand: int


PRESETS: List[Preset] = [
    Preset("small",   "Small (Smoke)", "runs_small",   3000, 10,  800, 0.95, "selfplay",     20, 2, 6),
    Preset("full",    "Full (Stable)", "runs_full",   15000, 12, 1200, 0.85, "selfplay",     20, 2, 6),
    Preset("scale",   "Scale (30k)",   "runs_scale",  30000, 14, 1400, 0.80, "selfplay",     20, 2, 6),
    Preset("endmine", "EndMine (15k)", "runs_endmine", 15000, 12, 1200, 0.85, "endgame_mine", 20, 2, 6),
]


class PanelApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Domino AI Panel")
        self.root.geometry("1120x780")
        self.root.minsize(950, 650)

        self._setup_styles()

        self.output_queue: "queue.Queue[str]" = queue.Queue()
        self.current_process: Optional[subprocess.Popen] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.running_var = tk.BooleanVar(value=False)
        self.cancel_requested = False

        self.last_eval: Optional[Dict[str, Any]] = None

        self.status_var = tk.StringVar(value="Ready")
        self.progress_text_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0.0)

        # ----------------------------
        # Cycle vars
        # ----------------------------
        self.cycle_preset = tk.StringVar(value=PRESETS[0].key if PRESETS else "small")
        self.cycle_out_dir = tk.StringVar(value="runs_small")
        self.cycle_n = tk.IntVar(value=1)
        self.cycle_matches = tk.IntVar(value=3000)
        self.cycle_det = tk.IntVar(value=10)
        self.cycle_think_ms = tk.IntVar(value=800)
        self.cycle_temp = tk.DoubleVar(value=0.95)
        self.cycle_opp_mix_greedy = tk.DoubleVar(value=1.0)
        self.cycle_me_mix_greedy = tk.DoubleVar(value=1.0)
        self.cycle_leaf_value_weight = tk.DoubleVar(value=0.0)
        self.cycle_gift_penalty_weight = tk.DoubleVar(value=0.15)
        self.cycle_pessimism_alpha_max = tk.DoubleVar(value=0.0)

        self.cycle_mode = tk.StringVar(value="selfplay")
        self.cycle_end_close = tk.IntVar(value=20)
        self.cycle_end_bone = tk.IntVar(value=2)
        self.cycle_end_hand = tk.IntVar(value=6)

        self.cycle_eval_matches = tk.IntVar(value=500)
        self.cycle_eval_me = tk.StringVar(value="rust")
        self.cycle_eval_opp = tk.StringVar(value="strategic")
        # NEW: ensure cycle eval is "max strength" by default
        self.cycle_eval_det = tk.IntVar(value=20)
        self.cycle_eval_think_ms = tk.IntVar(value=2000)
        self.cycle_eval_jobs = tk.IntVar(value=6)
        self.cycle_seed = tk.IntVar(value=99999)

        self.cycle_train_mode = tk.StringVar(value="both")
        self.cycle_replay_last_n = tk.IntVar(value=4)
        self.cycle_spike_pos_w = tk.DoubleVar(value=5.0)

        self.cycle_epochs = tk.IntVar(value=0)
        self.cycle_steps_per_epoch = tk.IntVar(value=0)
        self.cycle_batch = tk.IntVar(value=256)
        self.cycle_lr = tk.DoubleVar(value=1e-3)
        self.cycle_l2 = tk.DoubleVar(value=1e-4)
        self.cycle_shuffle_shards_each_epoch = tk.BooleanVar(value=True)
        self.cycle_shuffle_buffer = tk.IntVar(value=8192)

        # ----------------------------
        # Data vars
        # ----------------------------
        self.data_preset = tk.StringVar(value="endmine")
        self.data_out_dir = tk.StringVar(value="runs_endmine")
        self.data_matches = tk.IntVar(value=15000)
        self.data_det = tk.IntVar(value=12)
        self.data_think_ms = tk.IntVar(value=1200)
        self.data_temp = tk.DoubleVar(value=0.85)
        self.data_opp_mix_greedy = tk.DoubleVar(value=1.0)
        self.data_me_mix_greedy = tk.DoubleVar(value=1.0)
        self.data_leaf_value_weight = tk.DoubleVar(value=0.0)
        self.data_gift_penalty_weight = tk.DoubleVar(value=0.15)
        self.data_pessimism_alpha_max = tk.DoubleVar(value=0.0)

        self.data_mode = tk.StringVar(value="endgame_mine")
        self.data_end_close = tk.IntVar(value=20)
        self.data_end_bone = tk.IntVar(value=2)
        self.data_end_hand = tk.IntVar(value=6)
        self.data_seed = tk.IntVar(value=99999)

        # ----------------------------
        # Eval vars
        # ----------------------------
        self.eval_matches = tk.IntVar(value=1000)
        self.eval_target = tk.IntVar(value=150)
        self.eval_seed = tk.IntVar(value=12345)
        self.eval_me = tk.StringVar(value="rust")
        self.eval_level = tk.StringVar(value="quick")
        self.eval_det = tk.IntVar(value=20)
        self.eval_think_ms = tk.IntVar(value=2000)
        self.eval_opp = tk.StringVar(value="strategic")
        self.eval_jobs = tk.IntVar(value=6)
        # NEW: expose rust knobs in Eval tab (production defaults)
        self.eval_rust_opp_mix_greedy = tk.DoubleVar(value=1.0)
        self.eval_rust_me_mix_greedy = tk.DoubleVar(value=1.0)
        self.eval_rust_leaf_value_weight = tk.DoubleVar(value=0.0)
        self.eval_rust_gift_penalty_weight = tk.DoubleVar(value=0.15)
        self.eval_rust_pessimism_alpha_max = tk.DoubleVar(value=0.0)

        # ----------------------------
        # Inference Lab vars (IM4.2)
        # ----------------------------
        self.infer_out_dir = tk.StringVar(value="runs_infer_strat_pilot")
        self.infer_collect_matches = tk.IntVar(value=2000)
        self.infer_collect_seed = tk.IntVar(value=99999)
        self.infer_collect_det = tk.IntVar(value=8)
        self.infer_collect_think_ms = tk.IntVar(value=120)
        self.infer_collect_opp = tk.StringVar(value="strategic")
        self.infer_collect_gift_w = tk.DoubleVar(value=0.15)
        self.infer_collect_progress_every = tk.IntVar(value=200)

        # Strategic INF1 manifest (output of Collect step)
        self.infer_manifest_path = tk.StringVar(value="")
        self.infer_samples = tk.IntVar(value=0)
        self.infer_out_errors = tk.IntVar(value=0)

        # IM4.3: Self-play manifest selection (for training inference_model_self.json)
        self.infer_selfplay_dir = tk.StringVar(value="runs_small")
        self.infer_self_manifest_path = tk.StringVar(value="")

        self.infer_train_steps = tk.IntVar(value=2000)
        self.infer_train_batch = tk.IntVar(value=256)
        self.infer_train_hidden = tk.IntVar(value=256)
        self.infer_train_lr = tk.DoubleVar(value=1e-3)
        self.infer_train_l2 = tk.DoubleVar(value=1e-4)
        self.infer_train_val_samples = tk.IntVar(value=4000)
        # Single-train output (legacy button)
        self.infer_train_out = tk.StringVar(value="inference_model_strat.json")
        self.infer_copy_to_default = tk.BooleanVar(value=True)

        # IM4.3: Pair-train outputs (fixed names recommended)
        self.infer_train_out_self = tk.StringVar(value="inference_model_self.json")
        self.infer_train_out_strat = tk.StringVar(value="inference_model_strat.json")

        self.infer_ab_matches = tk.IntVar(value=2000)
        self.infer_ab_det = tk.IntVar(value=8)
        self.infer_ab_think_ms = tk.IntVar(value=120)
        self.infer_ab_jobs = tk.IntVar(value=6)
        self.infer_ab_seed1 = tk.IntVar(value=314159)
        self.infer_ab_seed2 = tk.IntVar(value=99999)
        self.infer_ab_opp = tk.StringVar(value="strategic")
        self.infer_ab_level = tk.StringVar(value="quick")
        self.infer_ab_gift_w = tk.DoubleVar(value=0.15)
        self.infer_ab_assert_every = tk.IntVar(value=10)

        # A/B result storage
        self._infer_ab_runs: List[Dict[str, Any]] = []

        # UI widgets
        self.log_text: Optional[scrolledtext.ScrolledText] = None
        self.eval_result_text: Optional[scrolledtext.ScrolledText] = None

        self.infer_table: Optional[ttk.Treeview] = None
        self.infer_summary_text: Optional[scrolledtext.ScrolledText] = None

        # build UI
        self._build_ui()
        self._load_settings()
        self._poll_output_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------------------
    # UI setup
    # ----------------------------
    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", font=("Consolas", 9))
        style.configure("GroupTitle.TLabelframe.Label", font=("Segoe UI", 9, "bold"))

    def _build_ui(self) -> None:
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        self._build_header(main_container)

        self.paned = ttk.PanedWindow(main_container, orient=tk.VERTICAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 0))

        notebook_frame = ttk.Frame(self.paned)
        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_cycle = ttk.Frame(self.notebook)
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_eval = ttk.Frame(self.notebook)
        self.tab_infer = ttk.Frame(self.notebook)
        self.tab_diag = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_cycle, text="  ⚡ Cycle  ")
        self.notebook.add(self.tab_data, text="  📊 Data  ")
        self.notebook.add(self.tab_eval, text="  📈 Eval  ")
        self.notebook.add(self.tab_infer, text="  🧠 Inference Lab  ")
        self.notebook.add(self.tab_diag, text="  🔧 Diagnostics  ")

        self._build_cycle_tab(self.tab_cycle)
        self._build_data_tab(self.tab_data)
        self._build_eval_tab(self.tab_eval)
        self._build_infer_tab(self.tab_infer)
        self._build_diag_tab(self.tab_diag)

        self.paned.add(notebook_frame, weight=3)

        log_frame = self._build_log_panel()
        self.paned.add(log_frame, weight=2)

        self._build_status_bar(main_container)

    def _build_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, padding=(10, 8))
        header.pack(fill=tk.X)

        ttk.Label(header, text="Domino AI Panel", style="Header.TLabel").pack(side=tk.LEFT)

        btn_frame = ttk.Frame(header)
        btn_frame.pack(side=tk.RIGHT)

        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel_clicked, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Separator(btn_frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Button(btn_frame, text="Copy", command=self._copy_log, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Save", command=self._save_log, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear", command=self._clear_log, width=8).pack(side=tk.LEFT, padx=2)

    def _build_log_panel(self) -> ttk.Frame:
        frame = ttk.LabelFrame(self.paned, text=" Log ", padding=4)

        self.log_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD,
            font=("Consolas", 9), bg="#1a1a2e", fg="#eee",
            insertbackground="#fff", selectbackground="#4a4a6a"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        return frame

    def _build_status_bar(self, parent: ttk.Frame) -> None:
        status_frame = ttk.Frame(parent, padding=(10, 6))
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_indicator = tk.Label(status_frame, text="●", font=("Segoe UI", 12), fg="#4caf50")
        self.status_indicator.pack(side=tk.LEFT)

        ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.LEFT, padx=(4, 15))

        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100.0, length=300, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(status_frame, textvariable=self.progress_text_var, style="Status.TLabel").pack(side=tk.LEFT)

    # ----------------------------
    # Tabs
    # ----------------------------
    def _build_cycle_tab(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        content = ttk.Frame(scroll_frame, padding=10)
        content.pack(fill=tk.BOTH, expand=True)

        preset_frame = ttk.Frame(content)
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)
        ttk.Combobox(
            preset_frame, textvariable=self.cycle_preset,
            values=[p.key for p in PRESETS], width=14, state="readonly"
        ).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(preset_frame, text="Apply", command=self._apply_cycle_preset, width=7).pack(side=tk.LEFT)

        ttk.Label(preset_frame, text="out_dir:").pack(side=tk.LEFT, padx=(20, 4))
        ttk.Entry(preset_frame, textvariable=self.cycle_out_dir, width=18).pack(side=tk.LEFT)

        ttk.Label(preset_frame, text="n:").pack(side=tk.LEFT, padx=(15, 4))
        ttk.Entry(preset_frame, textvariable=self.cycle_n, width=4).pack(side=tk.LEFT)

        ttk.Label(preset_frame, text="seed:").pack(side=tk.LEFT, padx=(15, 4))
        ttk.Entry(preset_frame, textvariable=self.cycle_seed, width=8).pack(side=tk.LEFT)

        ttk.Button(preset_frame, text="?", command=lambda: self._show_help("cycle"), width=2).pack(side=tk.RIGHT)

        gen_frame = ttk.LabelFrame(content, text=" Generation ", padding=8, style="GroupTitle.TLabelframe")
        gen_frame.pack(fill=tk.X, pady=(0, 8))

        gen_row1 = ttk.Frame(gen_frame)
        gen_row1.pack(fill=tk.X, pady=2)
        self._add_field(gen_row1, "matches", self.cycle_matches, 7)
        self._add_field(gen_row1, "det", self.cycle_det, 4)
        self._add_field(gen_row1, "think_ms", self.cycle_think_ms, 5)
        self._add_field(gen_row1, "temp", self.cycle_temp, 5)
        ttk.Label(gen_row1, text="mode:").pack(side=tk.LEFT, padx=(15, 4))
        ttk.Combobox(gen_row1, textvariable=self.cycle_mode, values=["selfplay", "endgame_mine"], width=12, state="readonly").pack(side=tk.LEFT)

        gen_row2 = ttk.Frame(gen_frame)
        gen_row2.pack(fill=tk.X, pady=2)
        self._add_field(gen_row2, "end_close", self.cycle_end_close, 4)
        self._add_field(gen_row2, "end_bone", self.cycle_end_bone, 4)
        self._add_field(gen_row2, "end_hand", self.cycle_end_hand, 4)

        ismcts_frame = ttk.LabelFrame(content, text=" ISMCTS Knobs ", padding=8, style="GroupTitle.TLabelframe")
        ismcts_frame.pack(fill=tk.X, pady=(0, 8))

        ismcts_row = ttk.Frame(ismcts_frame)
        ismcts_row.pack(fill=tk.X, pady=2)
        self._add_field(ismcts_row, "opp_mix", self.cycle_opp_mix_greedy, 5)
        self._add_field(ismcts_row, "me_mix", self.cycle_me_mix_greedy, 5)
        self._add_field(ismcts_row, "leaf_w", self.cycle_leaf_value_weight, 5)
        self._add_field(ismcts_row, "gift_w", self.cycle_gift_penalty_weight, 5)
        self._add_field(ismcts_row, "pessimism", self.cycle_pessimism_alpha_max, 5)

        train_frame = ttk.LabelFrame(content, text=" Training ", padding=8, style="GroupTitle.TLabelframe")
        train_frame.pack(fill=tk.X, pady=(0, 8))

        train_row1 = ttk.Frame(train_frame); train_row1.pack(fill=tk.X, pady=2)
        ttk.Label(train_row1, text="mode:").pack(side=tk.LEFT)
        ttk.Combobox(train_row1, textvariable=self.cycle_train_mode, values=["both", "pv_only", "spike_only"], width=10, state="readonly").pack(side=tk.LEFT, padx=(4, 15))
        self._add_field(train_row1, "epochs", self.cycle_epochs, 4)
        self._add_field(train_row1, "steps/ep", self.cycle_steps_per_epoch, 5)
        self._add_field(train_row1, "batch", self.cycle_batch, 5)

        train_row2 = ttk.Frame(train_frame); train_row2.pack(fill=tk.X, pady=2)
        self._add_field(train_row2, "lr", self.cycle_lr, 10)
        self._add_field(train_row2, "l2", self.cycle_l2, 10)
        self._add_field(train_row2, "replay_n", self.cycle_replay_last_n, 4)
        self._add_field(train_row2, "spike_pos_w", self.cycle_spike_pos_w, 6)
        self._add_field(train_row2, "shuf_buf", self.cycle_shuffle_buffer, 7)

        train_row3 = ttk.Frame(train_frame); train_row3.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(train_row3, text="shuffle_shards_each_epoch", variable=self.cycle_shuffle_shards_each_epoch).pack(side=tk.LEFT)

        eval_frame = ttk.LabelFrame(content, text=" Cycle Eval ", padding=8, style="GroupTitle.TLabelframe")
        eval_frame.pack(fill=tk.X, pady=(0, 8))

        eval_row = ttk.Frame(eval_frame); eval_row.pack(fill=tk.X, pady=2)
        self._add_field(eval_row, "matches", self.cycle_eval_matches, 6)
        self._add_field(eval_row, "me", self.cycle_eval_me, 8)
        self._add_field(eval_row, "opp", self.cycle_eval_opp, 10)
        self._add_field(eval_row, "det", self.cycle_eval_det, 5)
        self._add_field(eval_row, "think_ms", self.cycle_eval_think_ms, 7)
        self._add_field(eval_row, "jobs", self.cycle_eval_jobs, 4)

        btn_frame = ttk.Frame(content)
        btn_frame.pack(fill=tk.X, pady=(10, 5))
        ttk.Button(btn_frame, text="▶ Run Cycle", command=self.on_run_cycle_clicked, style="Run.TButton", width=20).pack()

    def _build_data_tab(self, parent: ttk.Frame) -> None:
        content = ttk.Frame(parent, padding=10)
        content.pack(fill=tk.BOTH, expand=True)

        preset_frame = ttk.Frame(content)
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)
        ttk.Combobox(preset_frame, textvariable=self.data_preset, values=[p.key for p in PRESETS], width=14, state="readonly").pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(preset_frame, text="Apply", command=self._apply_data_preset, width=7).pack(side=tk.LEFT)

        ttk.Label(preset_frame, text="out_dir:").pack(side=tk.LEFT, padx=(20, 4))
        ttk.Entry(preset_frame, textvariable=self.data_out_dir, width=18).pack(side=tk.LEFT)

        ttk.Label(preset_frame, text="seed:").pack(side=tk.LEFT, padx=(15, 4))
        ttk.Entry(preset_frame, textvariable=self.data_seed, width=8).pack(side=tk.LEFT)

        ttk.Button(preset_frame, text="?", command=lambda: self._show_help("data"), width=2).pack(side=tk.RIGHT)

        gen_frame = ttk.LabelFrame(content, text=" Generation ", padding=8)
        gen_frame.pack(fill=tk.X, pady=(0, 8))

        gen_row1 = ttk.Frame(gen_frame); gen_row1.pack(fill=tk.X, pady=2)
        self._add_field(gen_row1, "matches", self.data_matches, 7)
        self._add_field(gen_row1, "det", self.data_det, 4)
        self._add_field(gen_row1, "think_ms", self.data_think_ms, 6)
        self._add_field(gen_row1, "temp", self.data_temp, 6)
        ttk.Label(gen_row1, text="mode:").pack(side=tk.LEFT, padx=(15, 4))
        ttk.Combobox(gen_row1, textvariable=self.data_mode, values=["selfplay", "endgame_mine"], width=12, state="readonly").pack(side=tk.LEFT)

        gen_row2 = ttk.Frame(gen_frame); gen_row2.pack(fill=tk.X, pady=2)
        self._add_field(gen_row2, "end_close", self.data_end_close, 4)
        self._add_field(gen_row2, "end_bone", self.data_end_bone, 4)
        self._add_field(gen_row2, "end_hand", self.data_end_hand, 4)

        ismcts_frame = ttk.LabelFrame(content, text=" ISMCTS Knobs ", padding=8)
        ismcts_frame.pack(fill=tk.X, pady=(0, 8))

        ismcts_row = ttk.Frame(ismcts_frame); ismcts_row.pack(fill=tk.X, pady=2)
        self._add_field(ismcts_row, "opp_mix", self.data_opp_mix_greedy, 5)
        self._add_field(ismcts_row, "me_mix", self.data_me_mix_greedy, 5)
        self._add_field(ismcts_row, "leaf_w", self.data_leaf_value_weight, 5)
        self._add_field(ismcts_row, "gift_w", self.data_gift_penalty_weight, 5)
        self._add_field(ismcts_row, "pessimism", self.data_pessimism_alpha_max, 5)

        btn_frame = ttk.Frame(content)
        btn_frame.pack(fill=tk.X, pady=(15, 5))
        ttk.Button(btn_frame, text="▶ Generate Shards", command=self.on_run_generate_clicked, style="Run.TButton", width=20).pack()

    def _build_eval_tab(self, parent: ttk.Frame) -> None:
        content = ttk.Frame(parent, padding=10)
        content.pack(fill=tk.BOTH, expand=True)

        settings_frame = ttk.LabelFrame(content, text=" Eval Settings ", padding=8)
        settings_frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(settings_frame); row1.pack(fill=tk.X, pady=2)
        self._add_field(row1, "matches", self.eval_matches, 7)
        self._add_field(row1, "target", self.eval_target, 5)
        self._add_field(row1, "seed", self.eval_seed, 10)
        self._add_field(row1, "jobs", self.eval_jobs, 5)

        row2 = ttk.Frame(settings_frame); row2.pack(fill=tk.X, pady=2)
        self._add_field(row2, "me", self.eval_me, 10)
        self._add_field(row2, "opp", self.eval_opp, 10)
        self._add_field(row2, "level", self.eval_level, 10)
        self._add_field(row2, "det", self.eval_det, 5)
        self._add_field(row2, "think_ms", self.eval_think_ms, 6)

        # Rust knobs (max strength defaults)
        row3 = ttk.Frame(settings_frame); row3.pack(fill=tk.X, pady=2)
        self._add_field(row3, "opp_mix", self.eval_rust_opp_mix_greedy, 6)
        self._add_field(row3, "me_mix", self.eval_rust_me_mix_greedy, 6)
        self._add_field(row3, "leaf_w", self.eval_rust_leaf_value_weight, 6)
        self._add_field(row3, "gift_w", self.eval_rust_gift_penalty_weight, 6)
        self._add_field(row3, "pess", self.eval_rust_pessimism_alpha_max, 6)

        btn_row = ttk.Frame(content)
        btn_row.pack(fill=tk.X, pady=(8, 10))
        ttk.Button(btn_row, text="▶ Run Eval", command=self.on_run_eval_clicked, style="Run.TButton", width=15).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="?", command=lambda: self._show_help("eval"), width=2).pack(side=tk.LEFT, padx=(8, 0))

        results_frame = ttk.LabelFrame(content, text=" Last Eval Results ", padding=8)
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.eval_result_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, font=("Consolas", 9), bg="#1a1a2e", fg="#eee", height=12)
        self.eval_result_text.pack(fill=tk.BOTH, expand=True)
        self.eval_result_text.insert("1.0", "No eval results yet.\nRun an evaluation to see results here.")
        self.eval_result_text.config(state=tk.DISABLED)

    def _build_infer_tab(self, parent: ttk.Frame) -> None:
        content = ttk.Frame(parent, padding=10)
        content.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(content)
        top.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(top, text="Inference Lab (IM4.2)", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Button(top, text="?", command=lambda: self._show_help("infer"), width=2).pack(side=tk.RIGHT)

        # 1) Collect
        collect = ttk.LabelFrame(content, text=" 1) Collect strategic INF1 dataset (open-loop) ", padding=8)
        collect.pack(fill=tk.X, pady=(0, 8))

        r1 = ttk.Frame(collect); r1.pack(fill=tk.X, pady=2)
        self._add_field(r1, "out_dir", self.infer_out_dir, 26)
        self._add_field(r1, "matches", self.infer_collect_matches, 7)
        self._add_field(r1, "seed", self.infer_collect_seed, 10)
        ttk.Button(r1, text="▶ Collect", command=self.on_infer_collect_clicked, width=12).pack(side=tk.RIGHT)

        r2 = ttk.Frame(collect); r2.pack(fill=tk.X, pady=2)
        self._add_field(r2, "det", self.infer_collect_det, 5)
        self._add_field(r2, "think_ms", self.infer_collect_think_ms, 6)
        self._add_field(r2, "opp", self.infer_collect_opp, 10)
        self._add_field(r2, "gift_w", self.infer_collect_gift_w, 6)
        self._add_field(r2, "progress_every", self.infer_collect_progress_every, 7)

        r3 = ttk.Frame(collect); r3.pack(fill=tk.X, pady=2)
        ttk.Label(r3, text="manifest:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(r3, textvariable=self.infer_manifest_path, width=72).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(r3, text="Browse", command=self.on_infer_browse_manifest, width=8).pack(side=tk.LEFT)

        r4 = ttk.Frame(collect); r4.pack(fill=tk.X, pady=2)
        ttk.Label(r4, text="infer_samples:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(r4, textvariable=self.infer_samples, width=10).pack(side=tk.LEFT)
        ttk.Label(r4, text="infer_out_errors:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(r4, textvariable=self.infer_out_errors, width=10).pack(side=tk.LEFT)

        # 2) Train
        train = ttk.LabelFrame(content, text=" 2) Train inference model ", padding=8)
        train.pack(fill=tk.X, pady=(0, 8))

        # IM4.3: selfplay manifest (for inference_model_self.json)
        sp0 = ttk.Frame(train); sp0.pack(fill=tk.X, pady=2)
        ttk.Label(sp0, text="selfplay_dir:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(sp0, textvariable=self.infer_selfplay_dir, width=26).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(sp0, text="Pick Latest", command=self.on_infer_pick_latest_self_manifest, width=12).pack(side=tk.LEFT)

        sp1 = ttk.Frame(train); sp1.pack(fill=tk.X, pady=2)
        ttk.Label(sp1, text="self_manifest:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(sp1, textvariable=self.infer_self_manifest_path, width=72).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(sp1, text="Browse", command=self.on_infer_browse_self_manifest, width=8).pack(side=tk.LEFT)

        tr1 = ttk.Frame(train); tr1.pack(fill=tk.X, pady=2)
        self._add_field(tr1, "steps", self.infer_train_steps, 7)
        self._add_field(tr1, "batch", self.infer_train_batch, 6)
        self._add_field(tr1, "hidden", self.infer_train_hidden, 6)
        self._add_field(tr1, "lr", self.infer_train_lr, 10)
        self._add_field(tr1, "l2", self.infer_train_l2, 10)
        self._add_field(tr1, "val_samples", self.infer_train_val_samples, 8)

        tr2 = ttk.Frame(train); tr2.pack(fill=tk.X, pady=2)
        self._add_field(tr2, "out(self)", self.infer_train_out_self, 28)
        self._add_field(tr2, "out(strat)", self.infer_train_out_strat, 28)
        ttk.Button(tr2, text="▶ Train Pair + Install", command=self.on_infer_train_pair_clicked, width=20).pack(side=tk.RIGHT)

        tr3 = ttk.Frame(train); tr3.pack(fill=tk.X, pady=2)
        ttk.Label(tr3, text="(legacy) out:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(tr3, textvariable=self.infer_train_out, width=28).pack(side=tk.LEFT)
        ttk.Checkbutton(tr3, text="copy to inference_model.json", variable=self.infer_copy_to_default).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(tr3, text="▶ Train (single)", command=self.on_infer_train_clicked, width=14).pack(side=tk.RIGHT)

        # 3) A/B
        ab = ttk.LabelFrame(content, text=" 3) A/B Magic (two seeds) ", padding=8)
        ab.pack(fill=tk.X, pady=(0, 8))

        ab1 = ttk.Frame(ab); ab1.pack(fill=tk.X, pady=2)
        self._add_field(ab1, "matches", self.infer_ab_matches, 7)
        self._add_field(ab1, "det", self.infer_ab_det, 5)
        self._add_field(ab1, "think_ms", self.infer_ab_think_ms, 6)
        self._add_field(ab1, "jobs", self.infer_ab_jobs, 5)
        self._add_field(ab1, "level", self.infer_ab_level, 10)

        ab2 = ttk.Frame(ab); ab2.pack(fill=tk.X, pady=2)
        self._add_field(ab2, "seed1", self.infer_ab_seed1, 10)
        self._add_field(ab2, "seed2", self.infer_ab_seed2, 10)
        self._add_field(ab2, "opp", self.infer_ab_opp, 10)
        self._add_field(ab2, "gift_w", self.infer_ab_gift_w, 6)
        self._add_field(ab2, "assert_every", self.infer_ab_assert_every, 6)

        ab3 = ttk.Frame(ab); ab3.pack(fill=tk.X, pady=2)
        ttk.Button(ab3, text="▶ Run A/B (4 runs)", command=self.on_infer_ab_clicked, width=18).pack(side=tk.LEFT)
        ttk.Button(ab3, text="Copy Summary", command=self.on_infer_copy_summary_clicked, width=14).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(ab3, text="Copy CSV", command=self.on_infer_copy_csv_clicked, width=10).pack(side=tk.LEFT, padx=(10, 0))

        # Results area: table + summary
        res = ttk.LabelFrame(content, text=" A/B Results (auto) ", padding=8)
        res.pack(fill=tk.BOTH, expand=True)

        cols = ("seed", "infer", "win_rate", "avg_best_opp_reply", "gift15", "gift_wn_exists", "delta_win", "delta_reply", "delta_gift15", "delta_wn")
        self.infer_table = ttk.Treeview(res, columns=cols, show="headings", height=8)
        headings = {
            "seed": "Seed",
            "infer": "INFER",
            "win_rate": "Win",
            "avg_best_opp_reply": "OppReply",
            "gift15": "Gift15",
            "gift_wn_exists": "WN_Exists",
            "delta_win": "ΔWin",
            "delta_reply": "ΔOppReply",
            "delta_gift15": "ΔGift15",
            "delta_wn": "ΔWN",
        }
        widths = {
            "seed": 90, "infer": 60, "win_rate": 70, "avg_best_opp_reply": 90, "gift15": 70, "gift_wn_exists": 80,
            "delta_win": 70, "delta_reply": 90, "delta_gift15": 80, "delta_wn": 70,
        }
        for c in cols:
            self.infer_table.heading(c, text=headings.get(c, c))
            self.infer_table.column(c, width=widths.get(c, 80), anchor="center")
        self.infer_table.pack(fill=tk.X, expand=False)

        self.infer_summary_text = scrolledtext.ScrolledText(res, wrap=tk.WORD, font=("Consolas", 9), bg="#1a1a2e", fg="#eee", height=6)
        self.infer_summary_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self._infer_summary_set("No A/B results yet.\n")

    def _build_diag_tab(self, parent: ttk.Frame) -> None:
        content = ttk.Frame(parent, padding=15)
        content.pack(fill=tk.BOTH, expand=True)

        ttk.Label(content, text="Diagnostics", style="Header.TLabel").pack(anchor="w", pady=(0, 15))

        btn_frame = ttk.LabelFrame(content, text=" Quick Actions ", padding=10)
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        btn_row = ttk.Frame(btn_frame)
        btn_row.pack(fill=tk.X)

        ttk.Button(btn_row, text="Run Tests (pytest)", command=self.on_run_tests_clicked, width=22).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="Run Gates (domino_tool)", command=self.on_run_gates_clicked, width=25).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="Smoke Cycle", command=self.on_smoke_cycle_clicked, width=15).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="?", command=lambda: self._show_help("diag"), width=2).pack(side=tk.RIGHT)

        desc_frame = ttk.LabelFrame(content, text=" Description ", padding=10)
        desc_frame.pack(fill=tk.BOTH, expand=True)

        desc_text = tk.Text(desc_frame, wrap=tk.WORD, height=8, font=("Segoe UI", 10), bg="#f5f5f5")
        desc_text.insert("1.0", """Use this tab before starting long training runs.

• Run Tests: Quick pytest sanity check
• Run Gates: Conformance + feature gates + data gates + bench + train-smoke
• Smoke Cycle: End-to-end pipeline test

Tip: Always run Smoke Cycle after code changes before large runs.""")
        desc_text.config(state=tk.DISABLED)
        desc_text.pack(fill=tk.BOTH, expand=True)

    # ----------------------------
    # Helpers / Settings
    # ----------------------------
    def _add_field(self, parent: ttk.Frame, label: str, var: tk.Variable, width: int) -> None:
        ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(parent, textvariable=var, width=width).pack(side=tk.LEFT)

    def _show_help(self, topic: str) -> None:
        messagebox.showinfo(f"Help: {topic}", HELP_TEXTS.get(topic, "No help available."))

    def _update_status_color(self, state: str) -> None:
        colors = {"ready": "#4caf50", "running": "#2196f3", "error": "#f44336", "cancel": "#ff9800"}
        self.status_indicator.config(fg=colors.get(state, "#4caf50"))

    def _python_exe(self) -> str:
        vpy = ROOT / ".venv" / "Scripts" / "python.exe"
        if vpy.exists():
            return str(vpy)
        return sys.executable

    def _apply_cycle_preset(self) -> None:
        key = self.cycle_preset.get().strip().lower()
        p = next((x for x in PRESETS if x.key == key), None)
        if not p:
            return
        self.cycle_out_dir.set(p.out_dir)
        self.cycle_matches.set(p.matches)
        self.cycle_det.set(p.det)
        self.cycle_think_ms.set(p.think_ms)
        self.cycle_temp.set(p.temp)
        self.cycle_mode.set(p.mode)
        self.cycle_end_close.set(p.end_close)
        self.cycle_end_bone.set(p.end_bone)
        self.cycle_end_hand.set(p.end_hand)
        self._append_log(f"[preset] cycle applied: {p.key}\n")
        self._save_settings()

    def _apply_data_preset(self) -> None:
        key = self.data_preset.get().strip().lower()
        p = next((x for x in PRESETS if x.key == key), None)
        if not p:
            return
        self.data_out_dir.set(p.out_dir)
        self.data_matches.set(p.matches)
        self.data_det.set(p.det)
        self.data_think_ms.set(p.think_ms)
        self.data_temp.set(p.temp)
        self.data_mode.set(p.mode)
        self.data_end_close.set(p.end_close)
        self.data_end_bone.set(p.end_bone)
        self.data_end_hand.set(p.end_hand)
        self._append_log(f"[preset] data applied: {p.key}\n")
        self._save_settings()

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            self._apply_cycle_preset()
            self._apply_data_preset()
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            self._apply_cycle_preset()
            self._apply_data_preset()
            return

        def get(name, default):
            return data.get(name, default)

        # Cycle
        self.cycle_preset.set(str(get("cycle_preset", self.cycle_preset.get())))
        self.cycle_out_dir.set(str(get("cycle_out_dir", self.cycle_out_dir.get())))
        self.cycle_n.set(int(get("cycle_n", self.cycle_n.get())))
        self.cycle_matches.set(int(get("cycle_matches", self.cycle_matches.get())))
        self.cycle_det.set(int(get("cycle_det", self.cycle_det.get())))
        self.cycle_think_ms.set(int(get("cycle_think_ms", self.cycle_think_ms.get())))
        self.cycle_temp.set(float(get("cycle_temp", self.cycle_temp.get())))
        self.cycle_opp_mix_greedy.set(float(get("cycle_opp_mix_greedy", self.cycle_opp_mix_greedy.get())))
        self.cycle_me_mix_greedy.set(float(get("cycle_me_mix_greedy", self.cycle_me_mix_greedy.get())))
        self.cycle_leaf_value_weight.set(float(get("cycle_leaf_value_weight", self.cycle_leaf_value_weight.get())))
        self.cycle_gift_penalty_weight.set(float(get("cycle_gift_penalty_weight", self.cycle_gift_penalty_weight.get())))
        self.cycle_pessimism_alpha_max.set(float(get("cycle_pessimism_alpha_max", self.cycle_pessimism_alpha_max.get())))
        self.cycle_mode.set(str(get("cycle_mode", self.cycle_mode.get())))
        self.cycle_end_close.set(int(get("cycle_end_close", self.cycle_end_close.get())))
        self.cycle_end_bone.set(int(get("cycle_end_bone", self.cycle_end_bone.get())))
        self.cycle_end_hand.set(int(get("cycle_end_hand", self.cycle_end_hand.get())))
        self.cycle_eval_matches.set(int(get("cycle_eval_matches", self.cycle_eval_matches.get())))
        self.cycle_eval_me.set(str(get("cycle_eval_me", self.cycle_eval_me.get())))
        self.cycle_eval_opp.set(str(get("cycle_eval_opp", self.cycle_eval_opp.get())))
        self.cycle_eval_det.set(int(get("cycle_eval_det", self.cycle_eval_det.get())))
        self.cycle_eval_think_ms.set(int(get("cycle_eval_think_ms", self.cycle_eval_think_ms.get())))
        self.cycle_eval_jobs.set(int(get("cycle_eval_jobs", self.cycle_eval_jobs.get())))
        self.cycle_seed.set(int(get("cycle_seed", self.cycle_seed.get())))

        self.cycle_train_mode.set(str(get("cycle_train_mode", self.cycle_train_mode.get())))
        self.cycle_replay_last_n.set(int(get("cycle_replay_last_n", self.cycle_replay_last_n.get())))
        self.cycle_spike_pos_w.set(float(get("cycle_spike_pos_w", self.cycle_spike_pos_w.get())))
        self.cycle_epochs.set(int(get("cycle_epochs", self.cycle_epochs.get())))
        self.cycle_steps_per_epoch.set(int(get("cycle_steps_per_epoch", self.cycle_steps_per_epoch.get())))
        self.cycle_batch.set(int(get("cycle_batch", self.cycle_batch.get())))
        self.cycle_lr.set(float(get("cycle_lr", self.cycle_lr.get())))
        self.cycle_l2.set(float(get("cycle_l2", self.cycle_l2.get())))
        self.cycle_shuffle_shards_each_epoch.set(bool(get("cycle_shuffle_shards_each_epoch", self.cycle_shuffle_shards_each_epoch.get())))
        self.cycle_shuffle_buffer.set(int(get("cycle_shuffle_buffer", self.cycle_shuffle_buffer.get())))

        # Data
        self.data_preset.set(str(get("data_preset", self.data_preset.get())))
        self.data_out_dir.set(str(get("data_out_dir", self.data_out_dir.get())))
        self.data_matches.set(int(get("data_matches", self.data_matches.get())))
        self.data_det.set(int(get("data_det", self.data_det.get())))
        self.data_think_ms.set(int(get("data_think_ms", self.data_think_ms.get())))
        self.data_temp.set(float(get("data_temp", self.data_temp.get())))
        self.data_opp_mix_greedy.set(float(get("data_opp_mix_greedy", self.data_opp_mix_greedy.get())))
        self.data_me_mix_greedy.set(float(get("data_me_mix_greedy", self.data_me_mix_greedy.get())))
        self.data_leaf_value_weight.set(float(get("data_leaf_value_weight", self.data_leaf_value_weight.get())))
        self.data_gift_penalty_weight.set(float(get("data_gift_penalty_weight", self.data_gift_penalty_weight.get())))
        self.data_pessimism_alpha_max.set(float(get("data_pessimism_alpha_max", self.data_pessimism_alpha_max.get())))
        self.data_mode.set(str(get("data_mode", self.data_mode.get())))
        self.data_end_close.set(int(get("data_end_close", self.data_end_close.get())))
        self.data_end_bone.set(int(get("data_end_bone", self.data_end_bone.get())))
        self.data_end_hand.set(int(get("data_end_hand", self.data_end_hand.get())))
        self.data_seed.set(int(get("data_seed", self.data_seed.get())))

        # Eval
        self.eval_matches.set(int(get("eval_matches", self.eval_matches.get())))
        self.eval_target.set(int(get("eval_target", self.eval_target.get())))
        self.eval_seed.set(int(get("eval_seed", self.eval_seed.get())))
        self.eval_me.set(str(get("eval_me", self.eval_me.get())))
        self.eval_level.set(str(get("eval_level", self.eval_level.get())))
        self.eval_det.set(int(get("eval_det", self.eval_det.get())))
        self.eval_think_ms.set(int(get("eval_think_ms", self.eval_think_ms.get())))
        self.eval_opp.set(str(get("eval_opp", self.eval_opp.get())))
        self.eval_jobs.set(int(get("eval_jobs", self.eval_jobs.get())))
        self.eval_rust_opp_mix_greedy.set(float(get("eval_rust_opp_mix_greedy", self.eval_rust_opp_mix_greedy.get())))
        self.eval_rust_me_mix_greedy.set(float(get("eval_rust_me_mix_greedy", self.eval_rust_me_mix_greedy.get())))
        self.eval_rust_leaf_value_weight.set(float(get("eval_rust_leaf_value_weight", self.eval_rust_leaf_value_weight.get())))
        self.eval_rust_gift_penalty_weight.set(float(get("eval_rust_gift_penalty_weight", self.eval_rust_gift_penalty_weight.get())))
        self.eval_rust_pessimism_alpha_max.set(float(get("eval_rust_pessimism_alpha_max", self.eval_rust_pessimism_alpha_max.get())))

        # Inference Lab
        self.infer_out_dir.set(str(get("infer_out_dir", self.infer_out_dir.get())))
        self.infer_collect_matches.set(int(get("infer_collect_matches", self.infer_collect_matches.get())))
        self.infer_collect_seed.set(int(get("infer_collect_seed", self.infer_collect_seed.get())))
        self.infer_collect_det.set(int(get("infer_collect_det", self.infer_collect_det.get())))
        self.infer_collect_think_ms.set(int(get("infer_collect_think_ms", self.infer_collect_think_ms.get())))
        self.infer_collect_opp.set(str(get("infer_collect_opp", self.infer_collect_opp.get())))
        self.infer_collect_gift_w.set(float(get("infer_collect_gift_w", self.infer_collect_gift_w.get())))
        self.infer_collect_progress_every.set(int(get("infer_collect_progress_every", self.infer_collect_progress_every.get())))
        self.infer_manifest_path.set(str(get("infer_manifest_path", self.infer_manifest_path.get())))
        self.infer_selfplay_dir.set(str(get("infer_selfplay_dir", self.infer_selfplay_dir.get())))
        self.infer_self_manifest_path.set(str(get("infer_self_manifest_path", self.infer_self_manifest_path.get())))
        self.infer_train_steps.set(int(get("infer_train_steps", self.infer_train_steps.get())))
        self.infer_train_batch.set(int(get("infer_train_batch", self.infer_train_batch.get())))
        self.infer_train_hidden.set(int(get("infer_train_hidden", self.infer_train_hidden.get())))
        self.infer_train_lr.set(float(get("infer_train_lr", self.infer_train_lr.get())))
        self.infer_train_l2.set(float(get("infer_train_l2", self.infer_train_l2.get())))
        self.infer_train_val_samples.set(int(get("infer_train_val_samples", self.infer_train_val_samples.get())))
        self.infer_train_out.set(str(get("infer_train_out", self.infer_train_out.get())))
        self.infer_copy_to_default.set(bool(get("infer_copy_to_default", self.infer_copy_to_default.get())))
        self.infer_train_out_self.set(str(get("infer_train_out_self", self.infer_train_out_self.get())))
        self.infer_train_out_strat.set(str(get("infer_train_out_strat", self.infer_train_out_strat.get())))
        self.infer_ab_matches.set(int(get("infer_ab_matches", self.infer_ab_matches.get())))
        self.infer_ab_det.set(int(get("infer_ab_det", self.infer_ab_det.get())))
        self.infer_ab_think_ms.set(int(get("infer_ab_think_ms", self.infer_ab_think_ms.get())))
        self.infer_ab_jobs.set(int(get("infer_ab_jobs", self.infer_ab_jobs.get())))
        self.infer_ab_seed1.set(int(get("infer_ab_seed1", self.infer_ab_seed1.get())))
        self.infer_ab_seed2.set(int(get("infer_ab_seed2", self.infer_ab_seed2.get())))
        self.infer_ab_opp.set(str(get("infer_ab_opp", self.infer_ab_opp.get())))
        self.infer_ab_level.set(str(get("infer_ab_level", self.infer_ab_level.get())))
        self.infer_ab_gift_w.set(float(get("infer_ab_gift_w", self.infer_ab_gift_w.get())))
        self.infer_ab_assert_every.set(int(get("infer_ab_assert_every", self.infer_ab_assert_every.get())))

    def _save_settings(self) -> None:
        data = {
            # Cycle
            "cycle_preset": self.cycle_preset.get(),
            "cycle_out_dir": self.cycle_out_dir.get(),
            "cycle_n": int(self.cycle_n.get()),
            "cycle_matches": int(self.cycle_matches.get()),
            "cycle_det": int(self.cycle_det.get()),
            "cycle_think_ms": int(self.cycle_think_ms.get()),
            "cycle_temp": float(self.cycle_temp.get()),
            "cycle_opp_mix_greedy": float(self.cycle_opp_mix_greedy.get()),
            "cycle_me_mix_greedy": float(self.cycle_me_mix_greedy.get()),
            "cycle_leaf_value_weight": float(self.cycle_leaf_value_weight.get()),
            "cycle_gift_penalty_weight": float(self.cycle_gift_penalty_weight.get()),
            "cycle_pessimism_alpha_max": float(self.cycle_pessimism_alpha_max.get()),
            "cycle_mode": self.cycle_mode.get(),
            "cycle_end_close": int(self.cycle_end_close.get()),
            "cycle_end_bone": int(self.cycle_end_bone.get()),
            "cycle_end_hand": int(self.cycle_end_hand.get()),
            "cycle_eval_matches": int(self.cycle_eval_matches.get()),
            "cycle_eval_me": self.cycle_eval_me.get(),
            "cycle_eval_opp": self.cycle_eval_opp.get(),
            "cycle_eval_det": int(self.cycle_eval_det.get()),
            "cycle_eval_think_ms": int(self.cycle_eval_think_ms.get()),
            "cycle_eval_jobs": int(self.cycle_eval_jobs.get()),
            "cycle_seed": int(self.cycle_seed.get()),

            "cycle_train_mode": self.cycle_train_mode.get(),
            "cycle_replay_last_n": int(self.cycle_replay_last_n.get()),
            "cycle_spike_pos_w": float(self.cycle_spike_pos_w.get()),
            "cycle_epochs": int(self.cycle_epochs.get()),
            "cycle_steps_per_epoch": int(self.cycle_steps_per_epoch.get()),
            "cycle_batch": int(self.cycle_batch.get()),
            "cycle_lr": float(self.cycle_lr.get()),
            "cycle_l2": float(self.cycle_l2.get()),
            "cycle_shuffle_shards_each_epoch": bool(self.cycle_shuffle_shards_each_epoch.get()),
            "cycle_shuffle_buffer": int(self.cycle_shuffle_buffer.get()),

            # Data
            "data_preset": self.data_preset.get(),
            "data_out_dir": self.data_out_dir.get(),
            "data_matches": int(self.data_matches.get()),
            "data_det": int(self.data_det.get()),
            "data_think_ms": int(self.data_think_ms.get()),
            "data_temp": float(self.data_temp.get()),
            "data_opp_mix_greedy": float(self.data_opp_mix_greedy.get()),
            "data_me_mix_greedy": float(self.data_me_mix_greedy.get()),
            "data_leaf_value_weight": float(self.data_leaf_value_weight.get()),
            "data_gift_penalty_weight": float(self.data_gift_penalty_weight.get()),
            "data_pessimism_alpha_max": float(self.data_pessimism_alpha_max.get()),
            "data_mode": self.data_mode.get(),
            "data_end_close": int(self.data_end_close.get()),
            "data_end_bone": int(self.data_end_bone.get()),
            "data_end_hand": int(self.data_end_hand.get()),
            "data_seed": int(self.data_seed.get()),

            # Eval
            "eval_matches": int(self.eval_matches.get()),
            "eval_target": int(self.eval_target.get()),
            "eval_seed": int(self.eval_seed.get()),
            "eval_me": self.eval_me.get(),
            "eval_level": self.eval_level.get(),
            "eval_det": int(self.eval_det.get()),
            "eval_think_ms": int(self.eval_think_ms.get()),
            "eval_opp": self.eval_opp.get(),
            "eval_jobs": int(self.eval_jobs.get()),
            "eval_rust_opp_mix_greedy": float(self.eval_rust_opp_mix_greedy.get()),
            "eval_rust_me_mix_greedy": float(self.eval_rust_me_mix_greedy.get()),
            "eval_rust_leaf_value_weight": float(self.eval_rust_leaf_value_weight.get()),
            "eval_rust_gift_penalty_weight": float(self.eval_rust_gift_penalty_weight.get()),
            "eval_rust_pessimism_alpha_max": float(self.eval_rust_pessimism_alpha_max.get()),

            # Inference Lab
            "infer_out_dir": self.infer_out_dir.get(),
            "infer_collect_matches": int(self.infer_collect_matches.get()),
            "infer_collect_seed": int(self.infer_collect_seed.get()),
            "infer_collect_det": int(self.infer_collect_det.get()),
            "infer_collect_think_ms": int(self.infer_collect_think_ms.get()),
            "infer_collect_opp": self.infer_collect_opp.get(),
            "infer_collect_gift_w": float(self.infer_collect_gift_w.get()),
            "infer_collect_progress_every": int(self.infer_collect_progress_every.get()),
            "infer_manifest_path": self.infer_manifest_path.get(),
            "infer_selfplay_dir": self.infer_selfplay_dir.get(),
            "infer_self_manifest_path": self.infer_self_manifest_path.get(),
            "infer_train_steps": int(self.infer_train_steps.get()),
            "infer_train_batch": int(self.infer_train_batch.get()),
            "infer_train_hidden": int(self.infer_train_hidden.get()),
            "infer_train_lr": float(self.infer_train_lr.get()),
            "infer_train_l2": float(self.infer_train_l2.get()),
            "infer_train_val_samples": int(self.infer_train_val_samples.get()),
            "infer_train_out": self.infer_train_out.get(),
            "infer_copy_to_default": bool(self.infer_copy_to_default.get()),
            "infer_train_out_self": self.infer_train_out_self.get(),
            "infer_train_out_strat": self.infer_train_out_strat.get(),
            "infer_ab_matches": int(self.infer_ab_matches.get()),
            "infer_ab_det": int(self.infer_ab_det.get()),
            "infer_ab_think_ms": int(self.infer_ab_think_ms.get()),
            "infer_ab_jobs": int(self.infer_ab_jobs.get()),
            "infer_ab_seed1": int(self.infer_ab_seed1.get()),
            "infer_ab_seed2": int(self.infer_ab_seed2.get()),
            "infer_ab_opp": self.infer_ab_opp.get(),
            "infer_ab_level": self.infer_ab_level.get(),
            "infer_ab_gift_w": float(self.infer_ab_gift_w.get()),
            "infer_ab_assert_every": int(self.infer_ab_assert_every.get()),
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ----------------------------
    # Actions
    # ----------------------------
    def on_run_cycle_clicked(self) -> None:
        if self.running_var.get():
            messagebox.showwarning("Running", "Process is running. Press Cancel to stop.")
            return
        if not FACTORY_MIN.exists():
            messagebox.showerror("Missing", f"missing {FACTORY_MIN}")
            return

        self._save_settings()

        args = [
            self._python_exe(), str(FACTORY_MIN), "cycle",
            "--n", str(int(self.cycle_n.get())),
            "--out", self.cycle_out_dir.get().strip() or "runs_small",
            "--matches", str(int(self.cycle_matches.get())),
            "--det", str(int(self.cycle_det.get())),
            "--think_ms", str(int(self.cycle_think_ms.get())),
            "--temp", str(float(self.cycle_temp.get())),
            "--opp_mix_greedy", str(float(self.cycle_opp_mix_greedy.get())),
            "--me_mix_greedy", str(float(self.cycle_me_mix_greedy.get())),
            "--leaf_value_weight", str(float(self.cycle_leaf_value_weight.get())),
            "--gift_penalty_weight", str(float(self.cycle_gift_penalty_weight.get())),
            "--pessimism_alpha_max", str(float(self.cycle_pessimism_alpha_max.get())),
            "--eval_matches", str(int(self.cycle_eval_matches.get())),
            "--eval_me", str(self.cycle_eval_me.get()),
            "--eval_opp", str(self.cycle_eval_opp.get()),
            "--eval_det", str(int(self.cycle_eval_det.get())),
            "--eval_think_ms", str(int(self.cycle_eval_think_ms.get())),
            "--eval_jobs", str(int(self.cycle_eval_jobs.get())),
            "--seed", str(int(self.cycle_seed.get())),
            "--mode", str(self.cycle_mode.get()),
            "--end_close", str(int(self.cycle_end_close.get())),
            "--end_bone", str(int(self.cycle_end_bone.get())),
            "--end_hand", str(int(self.cycle_end_hand.get())),
            "--train_mode", str(self.cycle_train_mode.get()),
            "--replay_last_n", str(int(self.cycle_replay_last_n.get())),
            "--spike_pos_w", str(float(self.cycle_spike_pos_w.get())),
            "--epochs", str(int(self.cycle_epochs.get())),
            "--steps_per_epoch", str(int(self.cycle_steps_per_epoch.get())),
            "--batch", str(int(self.cycle_batch.get())),
            "--lr", str(float(self.cycle_lr.get())),
            "--l2", str(float(self.cycle_l2.get())),
            "--shuffle_buffer", str(int(self.cycle_shuffle_buffer.get())),
        ]
        if bool(self.cycle_shuffle_shards_each_epoch.get()):
            args.append("--shuffle_shards_each_epoch")

        self._start_command(args, desc="factory_min.py cycle")

    def on_run_generate_clicked(self) -> None:
        if self.running_var.get():
            messagebox.showwarning("Running", "Process is running. Press Cancel to stop.")
            return
        if not FACTORY_MIN.exists():
            messagebox.showerror("Missing", f"missing {FACTORY_MIN}")
            return

        self._save_settings()

        args = [
            self._python_exe(), str(FACTORY_MIN), "generate",
            "--out", self.data_out_dir.get().strip() or "runs_endmine",
            "--matches", str(int(self.data_matches.get())),
            "--det", str(int(self.data_det.get())),
            "--think_ms", str(int(self.data_think_ms.get())),
            "--temp", str(float(self.data_temp.get())),
            "--opp_mix_greedy", str(float(self.data_opp_mix_greedy.get())),
            "--me_mix_greedy", str(float(self.data_me_mix_greedy.get())),
            "--leaf_value_weight", str(float(self.data_leaf_value_weight.get())),
            "--gift_penalty_weight", str(float(self.data_gift_penalty_weight.get())),
            "--pessimism_alpha_max", str(float(self.data_pessimism_alpha_max.get())),
            "--seed", str(int(self.data_seed.get())),
            "--mode", str(self.data_mode.get()),
            "--end_close", str(int(self.data_end_close.get())),
            "--end_bone", str(int(self.data_end_bone.get())),
            "--end_hand", str(int(self.data_end_hand.get())),
        ]
        self._start_command(args, desc="factory_min.py generate")

    def on_run_eval_clicked(self) -> None:
        if self.running_var.get():
            messagebox.showwarning("Running", "Process is running. Press Cancel to stop.")
            return
        if not EVAL.exists():
            messagebox.showerror("Missing", f"missing {EVAL}")
            return

        self._save_settings()

        args = [
            self._python_exe(), str(EVAL),
            "--matches", str(int(self.eval_matches.get())),
            "--target", str(int(self.eval_target.get())),
            "--seed", str(int(self.eval_seed.get())),
            "--me", str(self.eval_me.get()),
            "--level", str(self.eval_level.get()),
            "--det", str(int(self.eval_det.get())),
            "--think_ms", str(int(self.eval_think_ms.get())),
            "--opp", str(self.eval_opp.get()),
            "--jobs", str(int(self.eval_jobs.get())),
            "--assert_every", "10",
            "--progress_every", "50",
            "--rust_opp_mix_greedy", str(float(self.eval_rust_opp_mix_greedy.get())),
            "--rust_me_mix_greedy", str(float(self.eval_rust_me_mix_greedy.get())),
            "--rust_leaf_value_weight", str(float(self.eval_rust_leaf_value_weight.get())),
            "--rust_gift_penalty_weight", str(float(self.eval_rust_gift_penalty_weight.get())),
            "--rust_pessimism_alpha_max", str(float(self.eval_rust_pessimism_alpha_max.get())),
        ]
        self._start_command(args, desc="evaluate.py")

    def on_run_tests_clicked(self) -> None:
        if self.running_var.get():
            return
        args = [self._python_exe(), "-m", "pytest", "-q"]
        self._start_command(args, desc="pytest")

    def on_run_gates_clicked(self) -> None:
        if self.running_var.get():
            return
        args = [self._python_exe(), str(ROOT / "tools" / "domino_tool.py"), "check"]
        self._start_command(args, desc="domino_tool check")

    def on_smoke_cycle_clicked(self) -> None:
        if self.running_var.get():
            return
        args = [
            self._python_exe(), str(FACTORY_MIN), "cycle",
            "--preset", "small",
            "--n", "1",
            "--eval_matches", "200",
            "--eval_me", "rust",
            "--eval_opp", "strategic",
            "--eval_jobs", "1",
        ]
        self._start_command(args, desc="smoke cycle")

    # Inference Lab actions
    def on_infer_browse_manifest(self) -> None:
        p = filedialog.askopenfilename(
            title="Select infer_strat_*.manifest.json",
            initialdir=str(ROOT),
            filetypes=[("Manifest JSON", "*.manifest.json"), ("All", "*.*")],
        )
        if p:
            self.infer_manifest_path.set(str(Path(p).resolve()))

    def on_infer_browse_self_manifest(self) -> None:
        p = filedialog.askopenfilename(
            title="Select selfplay run_*.manifest.json",
            initialdir=str(ROOT),
            filetypes=[("Manifest JSON", "*.manifest.json"), ("All", "*.*")],
        )
        if p:
            self.infer_self_manifest_path.set(str(Path(p).resolve()))

    def _infer_pick_latest_selfplay_manifest(self) -> Optional[str]:
        try:
            d = (ROOT / (self.infer_selfplay_dir.get().strip() or "runs_small")).resolve()
            if not d.exists():
                return None
            mans = sorted(d.glob("run_*.manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if mans:
                return str(mans[0])
        except Exception:
            return None
        return None

    def on_infer_pick_latest_self_manifest(self) -> None:
        p = self._infer_pick_latest_selfplay_manifest()
        if p:
            self.infer_self_manifest_path.set(str(Path(p).resolve()))
            self._append_log(f"[infer] picked selfplay manifest: {p}\n")
            self._save_settings()
        else:
            messagebox.showwarning("Not Found", "Could not find run_*.manifest.json in selfplay_dir. Generate selfplay shards first.")

    def _infer_pick_latest_manifest(self) -> Optional[str]:
        try:
            d = (ROOT / (self.infer_out_dir.get().strip() or "runs_infer_strat_pilot")).resolve()
            if not d.exists():
                return None
            mans = sorted(d.glob("infer_strat_*.manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if mans:
                return str(mans[0])
        except Exception:
            return None
        return None

    def on_infer_collect_clicked(self) -> None:
        if self.running_var.get():
            messagebox.showwarning("Running", "Process is running. Press Cancel to stop.")
            return
        if not EVAL.exists():
            messagebox.showerror("Missing", f"missing {EVAL}")
            return

        self._save_settings()

        out_dir = self.infer_out_dir.get().strip() or "runs_infer_strat_pilot"

        args = [
            self._python_exe(), str(EVAL),
            "--matches", str(int(self.infer_collect_matches.get())),
            "--target", "150",
            "--seed", str(int(self.infer_collect_seed.get())),
            "--me", "rust",
            "--level", "quick",
            "--det", str(int(self.infer_collect_det.get())),
            "--think_ms", str(int(self.infer_collect_think_ms.get())),
            "--opp", str(self.infer_collect_opp.get()),
            "--jobs", "1",
            "--assert_every", "10",
            "--progress_every", str(int(self.infer_collect_progress_every.get())),
            "--rust_gift_penalty_weight", str(float(self.infer_collect_gift_w.get())),
            "--infer_out_dir", out_dir,
        ]

        env = {"DOMINO_AVOID_BETA": "0", "DOMINO_INFER": "0"}
        self._start_command(args, desc="collect INF1 (strategic, open-loop)", env_override=env)

    def on_infer_train_clicked(self) -> None:
        if self.running_var.get():
            messagebox.showwarning("Running", "Process is running. Press Cancel to stop.")
            return
        if not INFER_TOOL.exists():
            messagebox.showerror("Missing", f"missing {INFER_TOOL}")
            return

        self._save_settings()

        mp = self.infer_manifest_path.get().strip()
        if not mp:
            picked = self._infer_pick_latest_manifest()
            if picked:
                mp = picked
                self.infer_manifest_path.set(mp)

        if not mp or not Path(mp).exists():
            messagebox.showerror("Missing", "No manifest selected/found. Run Collect first or Browse a manifest.")
            return

        out_name = self.infer_train_out.get().strip() or "inference_model_strat.json"

        train_cmd = [
            self._python_exe(), str(INFER_TOOL), "train",
            "--manifest", mp,
            "--steps", str(int(self.infer_train_steps.get())),
            "--batch", str(int(self.infer_train_batch.get())),
            "--hidden", str(int(self.infer_train_hidden.get())),
            "--lr", str(float(self.infer_train_lr.get())),
            "--l2", str(float(self.infer_train_l2.get())),
            "--val_samples", str(int(self.infer_train_val_samples.get())),
            "--out", out_name,
        ]

        seq: List[Tuple[List[str], Dict[str, str], str, bool]] = []
        seq.append((train_cmd, {}, f"train inference ({Path(mp).name})", False))

        if bool(self.infer_copy_to_default.get()):
            # copy model to inference_model.json
            copy_cmd = [
                self._python_exe(), "-c",
                "import shutil,sys; shutil.copyfile(sys.argv[1], sys.argv[2]); print('copied', sys.argv[1], '->', sys.argv[2])",
                out_name, "inference_model.json"
            ]
            seq.append((copy_cmd, {}, "copy to inference_model.json", False))

        self._infer_ab_runs.clear()
        self._infer_table_clear()
        self._infer_summary_set("Training...\n")
        self._start_sequence(seq, desc="Inference Train + Install", smart_eval_logs=False)

    def on_infer_train_pair_clicked(self) -> None:
        if self.running_var.get():
            messagebox.showwarning("Running", "Process is running. Press Cancel to stop.")
            return
        if not INFER_TOOL.exists():
            messagebox.showerror("Missing", f"missing {INFER_TOOL}")
            return

        self._save_settings()

        # --- resolve manifests ---
        self_mp = self.infer_self_manifest_path.get().strip()
        if not self_mp:
            picked = self._infer_pick_latest_selfplay_manifest()
            if picked:
                self_mp = picked
                self.infer_self_manifest_path.set(self_mp)

        strat_mp = self.infer_manifest_path.get().strip()
        if not strat_mp:
            picked = self._infer_pick_latest_manifest()
            if picked:
                strat_mp = picked
                self.infer_manifest_path.set(strat_mp)

        if not self_mp or not Path(self_mp).exists():
            messagebox.showerror("Missing", "Selfplay manifest not selected/found. Set selfplay_dir and click Pick Latest, or Browse it.")
            return
        if not strat_mp or not Path(strat_mp).exists():
            messagebox.showerror("Missing", "Strategic manifest not selected/found. Run Collect first or Browse it.")
            return

        out_self = self.infer_train_out_self.get().strip() or "inference_model_self.json"
        out_strat = self.infer_train_out_strat.get().strip() or "inference_model_strat.json"

        # --- training commands ---
        common = [
            "--steps", str(int(self.infer_train_steps.get())),
            "--batch", str(int(self.infer_train_batch.get())),
            "--hidden", str(int(self.infer_train_hidden.get())),
            "--lr", str(float(self.infer_train_lr.get())),
            "--l2", str(float(self.infer_train_l2.get())),
            "--val_samples", str(int(self.infer_train_val_samples.get())),
        ]

        train_self_cmd = [self._python_exe(), str(INFER_TOOL), "train", "--manifest", self_mp, *common, "--out", out_self]
        train_strat_cmd = [self._python_exe(), str(INFER_TOOL), "train", "--manifest", strat_mp, *common, "--out", out_strat]

        # Install step:
        # Ensure inference_model.json exists (and refresh it) for runtime.
        # This does NOT affect ensemble logic; ensemble uses *_self.json and *_strat.json when present.
        install_cmd = [
            self._python_exe(),
            "-c",
            "import shutil,sys; shutil.copyfile(sys.argv[1],'inference_model.json'); print('installed',sys.argv[1],'->','inference_model.json')",
            out_strat,
        ]

        seq: List[Tuple[List[str], Dict[str, str], str, bool]] = []
        seq.append((train_self_cmd, {}, f"train SELF ({Path(self_mp).name}) -> {out_self}", False))
        seq.append((train_strat_cmd, {}, f"train STRAT ({Path(strat_mp).name}) -> {out_strat}", False))
        seq.append((install_cmd, {}, "install/refresh inference_model.json", False))

        self._infer_ab_runs.clear()
        self._infer_table_clear()
        self._infer_summary_set("Training pair (self+strat) ...\n")
        self._start_sequence(seq, desc="Inference Train Pair + Install (IM4.3)", smart_eval_logs=False)

    def on_infer_ab_clicked(self) -> None:
        if self.running_var.get():
            messagebox.showwarning("Running", "Process is running. Press Cancel to stop.")
            return
        if not EVAL.exists():
            messagebox.showerror("Missing", f"missing {EVAL}")
            return

        self._save_settings()

        # require inference_model.json to exist for INFER=1
        if not (ROOT / "inference_model.json").exists():
            messagebox.showerror(
                "Missing",
                "inference_model.json not found.\nRun Train Pair + Install (recommended) or Train (single) with copy enabled.",
            )
            return

        matches = int(self.infer_ab_matches.get())
        det = int(self.infer_ab_det.get())
        think_ms = int(self.infer_ab_think_ms.get())
        jobs = int(self.infer_ab_jobs.get())
        level = str(self.infer_ab_level.get())
        opp = str(self.infer_ab_opp.get())
        gift_w = float(self.infer_ab_gift_w.get())
        assert_every = int(self.infer_ab_assert_every.get())
        seed1 = int(self.infer_ab_seed1.get())
        seed2 = int(self.infer_ab_seed2.get())

        def eval_cmd(seed: int) -> List[str]:
            return [
                self._python_exe(), str(EVAL),
                "--matches", str(matches),
                "--target", "150",
                "--seed", str(seed),
                "--me", "rust",
                "--level", level,
                "--det", str(det),
                "--think_ms", str(think_ms),
                "--opp", opp,
                "--jobs", str(jobs),
                "--assert_every", str(assert_every),
                "--progress_every", "200",
                "--rust_gift_penalty_weight", str(gift_w),
            ]

        self._infer_ab_runs.clear()
        self._infer_table_clear()
        self._infer_summary_set("Running A/B...\n")

        seq: List[Tuple[List[str], Dict[str, str], str, bool]] = []
        # seed1
        seq.append((eval_cmd(seed1), {"DOMINO_AVOID_BETA": "0", "DOMINO_INFER": "0"}, f"A/B seed={seed1} INFER=0", True))
        seq.append((eval_cmd(seed1), {"DOMINO_AVOID_BETA": "0", "DOMINO_INFER": "1"}, f"A/B seed={seed1} INFER=1", True))
        # seed2
        seq.append((eval_cmd(seed2), {"DOMINO_AVOID_BETA": "0", "DOMINO_INFER": "0"}, f"A/B seed={seed2} INFER=0", True))
        seq.append((eval_cmd(seed2), {"DOMINO_AVOID_BETA": "0", "DOMINO_INFER": "1"}, f"A/B seed={seed2} INFER=1", True))

        self._start_sequence(seq, desc="Inference A/B (two seeds, 4 runs)", smart_eval_logs=True)

    # ----------------------------
    # Process control
    # ----------------------------
    def on_cancel_clicked(self) -> None:
        if not self.running_var.get():
            self.status_var.set("No running process.")
            return

        self.cancel_requested = True
        self.status_var.set("Cancelling...")
        self._update_status_color("cancel")

        p = self.current_process
        if p is None:
            return

        try:
            p.terminate()
        except Exception:
            pass

        def killer() -> None:
            time.sleep(2.0)
            pp = self.current_process
            if pp is not None and pp.poll() is None:
                try:
                    pp.kill()
                except Exception:
                    pass

        threading.Thread(target=killer, daemon=True).start()

    def on_close(self) -> None:
        if self.running_var.get():
            if not messagebox.askyesno("Exit", "A process is running. Stop it and exit?"):
                return
            self.on_cancel_clicked()
            self.root.after(300, self.root.destroy)
            return
        self.root.destroy()

    # ----------------------------
    # Subprocess runners
    # ----------------------------
    def _start_command(self, args: List[str], desc: str, env_override: Optional[Dict[str, str]] = None) -> None:
        if self.running_var.get():
            messagebox.showwarning("Running", "Process is running. Press Cancel to stop.")
            return

        self._append_log(f"\n{'═'*60}\n[CMD] {' '.join(args)}\n{'═'*60}\n")
        self.status_var.set(f"Running: {desc}")
        self._update_status_color("running")
        self.progress_text_var.set("")
        self.progress_var.set(0.0)
        self.last_eval = None

        self.running_var.set(True)
        self.cancel_requested = False
        self.current_process = None

        def worker() -> None:
            try:
                env = os.environ.copy()
                env.setdefault("PYTHONUTF8", "1")
                env.setdefault("PYTHONIOENCODING", "utf-8")
                env.setdefault("PYTHONUNBUFFERED", "1")
                # Operational defaults (always)
                env.setdefault("DOMINO_INFER", "0")
                env.setdefault("DOMINO_AVOID_BETA", "0")
                if env_override:
                    for k, v in env_override.items():
                        env[str(k)] = str(v)

                p = subprocess.Popen(
                    args,
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                self.current_process = p

                assert p.stdout is not None
                for line in p.stdout:
                    s = line.rstrip("\n")
                    log_it = self._parse_line_for_progress_and_eval(s)
                    if log_it:
                        self.output_queue.put(s + "\n")

                rc = int(p.wait())
                if rc == 0:
                    self.output_queue.put(f"[✓ DONE rc={rc}]\n")
                else:
                    self.output_queue.put(f"[✗ FAILED rc={rc}]\n")
            except Exception as e:
                self.output_queue.put(f"[ERROR] {e}\n")
            finally:
                self.current_process = None
                self.running_var.set(False)
                self.status_var.set("Ready")
                self._update_status_color("ready")
                self.progress_text_var.set("")
                self.progress_var.set(0.0)
                self.cancel_requested = False

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _start_sequence(
        self,
        seq: List[Tuple[List[str], Dict[str, str], str, bool]],
        desc: str,
        smart_eval_logs: bool,
    ) -> None:
        """
        seq items: (args, env_override, step_desc, capture_eval_json)
        If smart_eval_logs=True, suppress noisy lines except progress/json/errors for eval steps.
        """
        if self.running_var.get():
            messagebox.showwarning("Running", "Process is running. Press Cancel to stop.")
            return

        self._append_log(f"\n{'═'*60}\n[SEQ] {desc}\n{'═'*60}\n")
        self.status_var.set(f"Running: {desc}")
        self._update_status_color("running")
        self.progress_text_var.set("")
        self.progress_var.set(0.0)

        self.running_var.set(True)
        self.cancel_requested = False
        self.current_process = None

        def worker() -> None:
            try:
                for (args, env_override, step_desc, capture_eval) in seq:
                    if self.cancel_requested:
                        break

                    self.output_queue.put(f"\n[SEQ-STEP] {step_desc}\n[CMD] {' '.join(args)}\n")

                    env = os.environ.copy()
                    env.setdefault("PYTHONUTF8", "1")
                    env.setdefault("PYTHONIOENCODING", "utf-8")
                    env.setdefault("PYTHONUNBUFFERED", "1")
                    # Operational defaults (always)
                    env.setdefault("DOMINO_INFER", "0")
                    env.setdefault("DOMINO_AVOID_BETA", "0")
                    for k, v in env_override.items():
                        env[str(k)] = str(v)

                    p = subprocess.Popen(
                        args,
                        cwd=str(ROOT),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    self.current_process = p

                    last_eval_json: Optional[Dict[str, Any]] = None

                    assert p.stdout is not None
                    for line in p.stdout:
                        s = line.rstrip("\n")
                        # always parse progress/json for UI
                        self._parse_line_for_progress_and_eval(s)

                        # smart logging during A/B eval
                        if smart_eval_logs and capture_eval:
                            keep = False
                            if PROGRESS_RE.match(s) or EVAL_PROGRESS_RE.match(s) or EVAL_JSON_RE.match(s) or INFER_JSON_RE.match(s):
                                keep = True
                            if ("Traceback" in s) or ("ERROR" in s) or ("[FAIL" in s) or ("[✗" in s):
                                keep = True
                            if keep:
                                self.output_queue.put(s + "\n")
                        else:
                            self.output_queue.put(s + "\n")

                        if capture_eval and EVAL_JSON_RE.match(s):
                            try:
                                j = json.loads(s)
                                if isinstance(j, dict) and isinstance(j.get("results"), dict):
                                    last_eval_json = j
                            except Exception:
                                pass

                    rc = int(p.wait())
                    self.current_process = None

                    if last_eval_json is not None:
                        self._infer_ab_runs.append(last_eval_json)
                        self._infer_update_table_and_summary()

                    if rc != 0:
                        self.output_queue.put(f"[SEQ-STEP FAILED rc={rc}] {step_desc}\n")
                        break
                    else:
                        self.output_queue.put(f"[SEQ-STEP OK] {step_desc}\n")

                if self.cancel_requested:
                    self.output_queue.put("[SEQ] cancelled\n")
                else:
                    self.output_queue.put("[SEQ] done\n")
            except Exception as e:
                self.output_queue.put(f"[SEQ ERROR] {e}\n")
            finally:
                self.current_process = None
                self.running_var.set(False)
                self.status_var.set("Ready")
                self._update_status_color("ready")
                self.progress_text_var.set("")
                self.progress_var.set(0.0)
                self.cancel_requested = False

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _poll_output_queue(self) -> None:
        try:
            while True:
                line = self.output_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_output_queue)

    # ----------------------------
    # Parsing
    # ----------------------------
    def _parse_line_for_progress_and_eval(self, line: str) -> bool:
        s = line.strip()

        m = PROGRESS_RE.match(s)
        if m:
            kv = dict(KV_RE.findall(m.group(1)))
            md = int(kv.get("matches_done", "0") or 0)
            tm = int(kv.get("total_matches", "0") or 0)
            sm = int(kv.get("samples", "0") or 0)
            if tm > 0:
                frac = max(0.0, min(1.0, float(md) / float(tm)))
                self.progress_var.set(frac * 100.0)
            self.progress_text_var.set(f"gen: {md}/{tm} matches, {sm} samples")
            return False

        mp = EVAL_PROGRESS_RE.match(s)
        if mp:
            done = int(mp.group(1))
            total = int(mp.group(2))
            mps = mp.group(3)
            if total > 0:
                frac = max(0.0, min(1.0, float(done) / float(total)))
                self.progress_var.set(frac * 100.0)
            self.progress_text_var.set(f"eval: {done}/{total} @ {mps} mps")
            return False

        if EVAL_JSON_RE.match(s):
            try:
                j = json.loads(s)
                if isinstance(j, dict) and "results" in j and "config" in j:
                    self.last_eval = j
                    self._update_eval_result_text(j)
            except Exception:
                pass
            return True

        if INFER_JSON_RE.match(s):
            try:
                j = json.loads(s)
                if isinstance(j, dict) and j.get("op") == "infer_out_done":
                    man = str(j.get("manifest") or "")
                    samples = int(j.get("infer_samples") or 0)
                    errs = int(j.get("infer_out_errors") or 0)
                    if man:
                        self.infer_manifest_path.set(man)
                    self.infer_samples.set(samples)
                    self.infer_out_errors.set(errs)
            except Exception:
                pass
            return True

        return True

    def _update_eval_result_text(self, rep: Dict[str, Any]) -> None:
        if self.eval_result_text is None:
            return
        self.eval_result_text.config(state=tk.NORMAL)
        self.eval_result_text.delete("1.0", tk.END)
        try:
            results = rep.get("results") or {}
            cfg = rep.get("config") or {}
            lines = [
                "═══════════════════════════════════════",
                "          EVALUATION RESULTS           ",
                "═══════════════════════════════════════",
                "",
                f"  Matches:    {cfg.get('matches')}",
                f"  Me:         {cfg.get('me_policy')}",
                f"  Opponent:   {cfg.get('opp_policy')}",
                "",
                "───────────────────────────────────────",
                "  PERFORMANCE",
                "───────────────────────────────────────",
                f"  Win Rate:     {results.get('win_rate')}",
                f"  Avg Margin:   {results.get('avg_margin')}",
                "",
                "───────────────────────────────────────",
                "  GIFT ANALYSIS",
                "───────────────────────────────────────",
                f"  Gift15 Rate:            {results.get('gift15_rate')}",
                f"  Gift Win Now Exists:    {results.get('gift_win_now_exists_rate')}",
                f"  Avg Best Opp Reply:     {results.get('avg_best_opp_reply_pts_after_me')}",
                "",
                "═══════════════════════════════════════",
            ]
            self.eval_result_text.insert("1.0", "\n".join(str(x) for x in lines))
        except Exception as e:
            self.eval_result_text.insert("1.0", f"Error parsing eval: {e}\n{json.dumps(rep, ensure_ascii=False, indent=2)}")
        self.eval_result_text.config(state=tk.DISABLED)

    # ----------------------------
    # Inference Table + Summary
    # ----------------------------
    def _infer_summary_set(self, txt: str) -> None:
        if self.infer_summary_text is None:
            return
        self.infer_summary_text.config(state=tk.NORMAL)
        self.infer_summary_text.delete("1.0", tk.END)
        self.infer_summary_text.insert("1.0", txt)
        self.infer_summary_text.config(state=tk.DISABLED)

    def _infer_table_clear(self) -> None:
        if self.infer_table is None:
            return
        for row in self.infer_table.get_children():
            self.infer_table.delete(row)

    def _infer_extract_key_metrics(self, rep: Dict[str, Any]) -> Tuple[int, int, float, float, float, float]:
        cfg = rep.get("config") or {}
        res = rep.get("results") or {}
        seed = int(cfg.get("base_seed") or 0)
        matches = int(cfg.get("matches") or 0)
        win = float(res.get("win_rate") or 0.0)
        opp_reply = float(res.get("avg_best_opp_reply_pts_after_me") or 0.0)
        gift15 = float(res.get("gift15_rate") or 0.0)
        wn = float(res.get("gift_win_now_exists_rate") or 0.0)
        return seed, matches, win, opp_reply, gift15, wn

    def _infer_update_table_and_summary(self) -> None:
        # Expect up to 4 runs: seed1 off/on, seed2 off/on
        if self.infer_table is None:
            return

        # Build mapping: (seed, infer) -> metrics
        # We infer infer flag from env of the step is not available; instead we detect by run order:
        # We enforce sequence order in on_infer_ab_clicked: off,on,off,on. So we can map by index.
        runs = self._infer_ab_runs[-4:]
        if len(runs) == 0:
            return

        # Determine infer flag by order within each seed:
        # We will group by seed then assign infer=0 to the earlier occurrence and infer=1 to later occurrence.
        per_seed: Dict[int, List[Tuple[Dict[str, Any], float, float, float, float]]] = {}
        for rep in runs:
            seed, _m, win, opp_reply, gift15, wn = self._infer_extract_key_metrics(rep)
            per_seed.setdefault(seed, []).append((rep, win, opp_reply, gift15, wn))

        self._infer_table_clear()

        # Row construction with deltas
        deltas_win: List[float] = []
        deltas_reply: List[float] = []
        deltas_g15: List[float] = []
        deltas_wn: List[float] = []

        for seed in sorted(per_seed.keys()):
            xs = per_seed[seed]
            if len(xs) < 2:
                # not enough, show raw
                rep, win, opp_reply, gift15, wn = xs[-1]
                self.infer_table.insert("", "end", values=(seed, "?", f"{win:.4f}", f"{opp_reply:.3f}", f"{gift15:.4f}", f"{wn:.4f}", "", "", "", ""))
                continue

            # by run time: assume first is infer=0, second infer=1 (order in our sequence)
            (rep0, win0, r0, g150, wn0) = xs[0]
            (rep1, win1, r1, g151, wn1) = xs[1]

            dwin = win1 - win0
            dreply = r1 - r0
            dg15 = g151 - g150
            dwn = wn1 - wn0

            deltas_win.append(dwin)
            deltas_reply.append(dreply)
            deltas_g15.append(dg15)
            deltas_wn.append(dwn)

            # baseline row
            self.infer_table.insert("", "end", values=(seed, "0", f"{win0:.4f}", f"{r0:.3f}", f"{g150:.4f}", f"{wn0:.4f}", "", "", "", ""))
            # infer row with deltas
            self.infer_table.insert("", "end", values=(seed, "1", f"{win1:.4f}", f"{r1:.3f}", f"{g151:.4f}", f"{wn1:.4f}",
                                                      f"{dwin:+.4f}", f"{dreply:+.3f}", f"{dg15:+.4f}", f"{dwn:+.4f}"))

        # Summary
        if deltas_win:
            avg_dwin = sum(deltas_win) / float(len(deltas_win))
            avg_dreply = sum(deltas_reply) / float(len(deltas_reply))
            avg_dg15 = sum(deltas_g15) / float(len(deltas_g15))
            avg_dwn = sum(deltas_wn) / float(len(deltas_wn))
            txt = (
                "=== A/B Mean Deltas (INFER=1 - INFER=0) ===\n"
                f"avg Δwin_rate: {avg_dwin:+.4f}\n"
                f"avg Δavg_best_opp_reply_pts_after_me: {avg_dreply:+.3f}\n"
                f"avg Δgift15_rate: {avg_dg15:+.4f}\n"
                f"avg Δgift_win_now_exists_rate: {avg_dwn:+.4f}\n"
                "\nInterpretation:\n"
                "- Δwin_rate > 0 is good\n"
                "- Δopp_reply < 0 is good\n"
                "- Δgift15 < 0 and ΔWN_exists < 0 are good\n"
            )
            self._infer_summary_set(txt)

    def on_infer_copy_summary_clicked(self) -> None:
        if self.infer_summary_text is None:
            return
        try:
            txt = self.infer_summary_text.get("1.0", tk.END).strip()
        except Exception:
            txt = ""
        if not txt:
            messagebox.showinfo("Copy", "No summary text to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self.status_var.set("A/B summary copied")

    def on_infer_copy_csv_clicked(self) -> None:
        if self.infer_table is None:
            return
        rows = []
        cols = self.infer_table["columns"]
        rows.append(",".join(cols))
        for iid in self.infer_table.get_children():
            vals = self.infer_table.item(iid, "values")
            # ensure CSV-safe minimal
            rows.append(",".join(str(x) for x in vals))
        txt = "\n".join(rows).strip()
        if not txt:
            messagebox.showinfo("Copy", "No table rows to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self.status_var.set("A/B table CSV copied")

    # ----------------------------
    # Log helpers
    # ----------------------------
    def _append_log(self, text: str) -> None:
        if self.log_text is None:
            return
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _copy_log(self) -> None:
        if self.log_text is None:
            return
        txt = self.log_text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self.status_var.set("Log copied")

    def _save_log(self) -> None:
        if self.log_text is None:
            return
        txt = self.log_text.get("1.0", tk.END).strip()
        if not txt:
            messagebox.showinfo("Log", "No log to save.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt"), ("All", "*.*")], initialdir=str(ROOT))
        if not p:
            return
        Path(p).write_text(txt, encoding="utf-8")
        self.status_var.set(f"Saved: {p}")

    def _clear_log(self) -> None:
        if self.log_text is None:
            return
        self.log_text.delete("1.0", tk.END)
        self.status_var.set("Log cleared")

    # ----------------------------
    # Run
    # ----------------------------
    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    root = tk.Tk()
    app = PanelApp(root)
    app.run()


if __name__ == "__main__":
    main()