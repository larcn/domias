"""
Domino Audit Probe (GUI) — Noise-safe ripgrep runner (robust).

Fixes:
- Auto-detect project root even if you accidentally select .../tools
  (walk up until we find Cargo.toml or engine.py).
- Exclude globs work with absolute paths (use !**/target/** not !target/**).
- Exclude audit outputs so it doesn't grep its own generated files.

Features:
- Add Files... / Add Folder... / Validate / Clear
- Optional Drag&Drop if tkinterdnd2 installed: pip install tkinterdnd2
- Writes:
  - <project_root>/audit/out/*.txt
  - <project_root>/audit/summary.md
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Optional drag & drop support
DND_AVAILABLE = False
DND_FILES = None
TkBase = tk.Tk

try:
    # pip install tkinterdnd2
    from tkinterdnd2 import DND_FILES as _DND_FILES  # type: ignore
    from tkinterdnd2 import TkinterDnD  # type: ignore

    DND_AVAILABLE = True
    DND_FILES = _DND_FILES
    TkBase = TkinterDnD.Tk  # type: ignore[attr-defined]
except Exception:
    DND_AVAILABLE = False
    DND_FILES = None
    TkBase = tk.Tk


# -----------------------------
# Noise control (fixed)
# NOTE: use **/ so it works with absolute input paths
# -----------------------------
EXCLUDE_GLOBS: List[str] = [
    "!**/target/**",
    "!**/.venv/**",
    "!**/__pycache__/**",
    "!**/.pytest_cache/**",
    "!**/runs_*/**",
    "!**/.smartpatch_backup/**",
    "!**/.smartpatch_sessions/**",
    "!**/.git/**",
    "!**/audit/**",  # IMPORTANT: don't grep our own reports
]

HEAD_LINES = 80
TAIL_LINES = 80


@dataclass(frozen=True)
class AuditCmd:
    key: str
    title: str
    argv: List[str]
    cwd_rel: str = "."


@dataclass
class CmdResult:
    key: str
    title: str
    argv: List[str]
    exit_code: int
    total_lines: int
    preview_text: str
    out_path: Path
    started_at: str
    finished_at: str


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _argv_to_str(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def _safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def detect_project_root(start: Path) -> Path:
    """
    Walk up from 'start' until we find a folder that looks like the project root.
    Criteria: contains Cargo.toml OR engine.py.
    """
    cur = start.resolve()
    for _ in range(12):
        if (cur / "Cargo.toml").exists() or (cur / "engine.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def _rg_base_args() -> List[str]:
    args = ["rg", "-n", "-S"]
    for g in EXCLUDE_GLOBS:
        args += ["--glob", g]
    return args


def _normalize_user_path_line(raw: str) -> str:
    s = raw.strip()

    for prefix in ("- ", "* ", "• ", "— ", "– "):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()

    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    if s.startswith(".\\"):
        s = "./" + s[2:].replace("\\", "/")

    return s


def parse_paths_text(txt: str) -> List[str]:
    out: List[str] = []
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = _normalize_user_path_line(line)
        if p:
            out.append(p)
    return out


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _relativize_if_under_root(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p.resolve())


def build_audit_cmds(include_paths: Sequence[str]) -> List[AuditCmd]:
    paths = list(include_paths) if include_paths else ["."]
    base = _rg_base_args()

    a_pat = r"^(from|import)\s+(engine|evaluate|panel|ai|factory_min|factory_panel)\b"
    b_pat = r"if __name__ == ['\"]__main__['\"]:"
    c_pat = r"suggest_move_ismcts|domino_rs|ctypes|pyo3|maturin"
    d_pat = r"DOMINO_(INFER|AVOID_BETA|MODEL|SEED|ME|THINK|DET|TEMP)|gift_penalty_weight"
    e_pat = r"suggest_move_ismcts|pub fn suggest_move|determinize|rollout|solver|DOMINO_INFER|DOMINO_AVOID_BETA"
    f_pat = r"DSH2|INF1|inf_shards|shards|generate_and_save|features_from_state_dict"

    return [
        AuditCmd("A_imports", "Python imports (engine/evaluate/panel/ai/factory_*)", base + [a_pat] + paths),
        AuditCmd("B_entrypoints", "Python entrypoints (__main__)", base + [b_pat] + paths),
        AuditCmd("C_bridge", "Rust bridge usage (domino_rs / suggest_move_ismcts / pyo3 / maturin)", base + [c_pat] + paths),
        AuditCmd("D_flags", "Runtime flags/knobs (DOMINO_* / gift_penalty_weight)", base + [d_pat] + paths),
        AuditCmd("E_rust_hotpath", "Rust hot-path signals (src/...)", base + [e_pat, "src"]),
        AuditCmd("F_datasets", "Dataset pipeline signals (DSH2/INF1/shards/features_from_state_dict)", base + [f_pat, "src", "tools", "."]),
    ]


def _run_stream_to_file(argv: List[str], cwd: Path, out_path: Path) -> Tuple[int, int, str]:
    _ensure_dir(out_path.parent)

    head: List[str] = []
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    total_lines = 0

    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        universal_newlines=True,
    )

    assert proc.stdout is not None
    with out_path.open("w", encoding="utf-8", errors="replace", newline="") as f:
        for line in proc.stdout:
            f.write(line)
            total_lines += 1
            if len(head) < HEAD_LINES:
                head.append(line)
            else:
                tail.append(line)

    exit_code = proc.wait()

    truncated = total_lines > (HEAD_LINES + TAIL_LINES)
    if not truncated:
        preview = "".join(head)
    else:
        preview = (
            "".join(head)
            + "\n"
            + f"[... TRUNCATED: total_lines={total_lines}, showing first {HEAD_LINES} + last {TAIL_LINES} ...]\n"
            + "".join(tail)
        )
    return exit_code, total_lines, preview


def generate_summary_md(root: Path, results: Sequence[CmdResult], out_dir: Path, used_paths: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("# Domino Audit Summary\n\n")
    lines.append(f"- root: `{root}`\n")
    lines.append(f"- generated_at: `{_now_iso()}`\n")
    lines.append(f"- outputs: `{_safe_relpath(out_dir, root)}`\n")
    lines.append(f"- rg: `{shutil.which('rg') or 'NOT FOUND'}`\n")

    lines.append("\n## Paths scope (A..D)\n\n")
    if used_paths:
        lines.append("```text\n")
        for p in used_paths:
            lines.append(p + "\n")
        lines.append("```\n")
    else:
        lines.append("- (empty) => scanning whole repo with noise-safe exclusions\n")

    lines.append("\n---\n\n")
    for r in results:
        lines.append(f"## {r.key} — {r.title}\n\n")
        lines.append(f"- exit_code: `{r.exit_code}`\n")
        lines.append(f"- lines: `{r.total_lines}`\n")
        lines.append(f"- out_file: `{_safe_relpath(r.out_path, root)}`\n")
        lines.append(f"- cmd: `{_argv_to_str(r.argv)}`\n\n")
        lines.append("### Preview\n\n```text\n")
        lines.append(r.preview_text.rstrip("\n"))
        lines.append("\n```\n\n---\n\n")
    return "".join(lines)


class AuditGUI(TkBase):
    def __init__(self) -> None:
        super().__init__()
        self.title("Domino Audit Probe (Noise-Safe)")
        self.geometry("1150x860")

        # Default root = parent of tools/ if script is inside tools/
        default_root = detect_project_root(Path(__file__).resolve().parent)
        self.root_var = tk.StringVar(value=str(default_root))

        self.status_var = tk.StringVar(value="Idle.")
        self._last_summary_md: Optional[str] = None
        self._last_out_dir: Optional[Path] = None

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(top, text="Project root:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.root_var, width=95).grid(row=0, column=1, sticky="we", padx=(8, 8))

        def browse_root() -> None:
            d = filedialog.askdirectory(title="Select project root directory")
            if d:
                self.root_var.set(str(detect_project_root(Path(d))))

        ttk.Button(top, text="Browse...", command=browse_root).grid(row=0, column=2, sticky="e")
        top.columnconfigure(1, weight=1)

        paths_frame = ttk.LabelFrame(
            self,
            text="Paths scope for A..D (one per line). Use Add buttons or Drag&Drop. Leave empty => scan whole repo (noise-safe).",
        )
        paths_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

        controls = ttk.Frame(paths_frame)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 6))

        ttk.Button(controls, text="Add Files...", command=self._add_files).pack(side=tk.LEFT)
        ttk.Button(controls, text="Add Folder...", command=self._add_folder).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Clear", command=self._clear_paths).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Validate", command=self._validate_paths).pack(side=tk.LEFT, padx=(8, 0))

        if DND_AVAILABLE:
            ttk.Label(controls, text="Drag&Drop: ON", foreground="green").pack(side=tk.RIGHT)
        else:
            ttk.Label(controls, text="Drag&Drop: OFF (pip install tkinterdnd2)", foreground="gray").pack(side=tk.RIGHT)

        self.paths_text = scrolledtext.ScrolledText(paths_frame, height=9, wrap=tk.WORD)
        self.paths_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.paths_text.insert(tk.END, "# Examples:\n# engine.py\n# evaluate.py\n# panel.py\n# tools/\n# src/\n")

        if DND_AVAILABLE and DND_FILES is not None:
            try:
                # type: ignore[attr-defined]
                self.paths_text.drop_target_register(DND_FILES)  # pyright: ignore
                # type: ignore[attr-defined]
                self.paths_text.dnd_bind("<<Drop>>", self._on_drop_paths)  # pyright: ignore
            except Exception:
                pass

        btns = ttk.Frame(self)
        btns.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        self.run_btn = ttk.Button(btns, text="Run Audit", command=self._run_audit)
        self.run_btn.pack(side=tk.LEFT)

        self.open_out_btn = ttk.Button(btns, text="Open audit/out folder", command=self._open_out_folder, state=tk.DISABLED)
        self.open_out_btn.pack(side=tk.LEFT, padx=(10, 0))

        self.copy_summary_btn = ttk.Button(
            btns, text="Copy summary.md to clipboard", command=self._copy_summary, state=tk.DISABLED
        )
        self.copy_summary_btn.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(btns, textvariable=self.status_var).pack(side=tk.RIGHT)

        self.nb = ttk.Notebook(self)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tabs = {}
        for key in ["summary", "A_imports", "B_entrypoints", "C_bridge", "D_flags", "E_rust_hotpath", "F_datasets"]:
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=key)
            txt = scrolledtext.ScrolledText(frame, wrap=tk.NONE)
            txt.pack(fill=tk.BOTH, expand=True)
            txt.configure(font=("Consolas", 10))
            self.tabs[key] = txt

    def _project_root(self) -> Path:
        return detect_project_root(Path(self.root_var.get()).expanduser())

    def _set_tab_text(self, key: str, text: str) -> None:
        w = self.tabs[key]
        w.configure(state=tk.NORMAL)
        w.delete("1.0", tk.END)
        w.insert(tk.END, text)
        w.configure(state=tk.NORMAL)

    def _append_paths(self, paths: Sequence[str]) -> None:
        if not paths:
            return

        root = self._project_root()
        existing = parse_paths_text(self.paths_text.get("1.0", tk.END))
        existing_set = set(existing)

        normalized_new: List[str] = []
        for p in paths:
            if not p:
                continue
            pp = Path(str(p)).expanduser()
            rel = _relativize_if_under_root(pp, root)
            normalized_new.append(rel.replace("\\", "/"))

        normalized_new = _dedupe_preserve_order(normalized_new)
        normalized_new = [p for p in normalized_new if p not in existing_set]

        if not normalized_new:
            self.status_var.set("No new paths to add.")
            return

        txt = self.paths_text.get("1.0", tk.END)
        if txt and not txt.endswith("\n"):
            self.paths_text.insert(tk.END, "\n")
        for p in normalized_new:
            self.paths_text.insert(tk.END, p + "\n")

        self.status_var.set(f"Added {len(normalized_new)} path(s).")

    def _add_files(self) -> None:
        root = self._project_root()
        fns = filedialog.askopenfilenames(title="Select files", initialdir=str(root))
        self._append_paths(list(fns))

    def _add_folder(self) -> None:
        root = self._project_root()
        d = filedialog.askdirectory(title="Select folder", initialdir=str(root))
        if d:
            self._append_paths([d])

    def _clear_paths(self) -> None:
        self.paths_text.delete("1.0", tk.END)
        self.paths_text.insert(tk.END, "# Paste paths here (one per line) OR use Add buttons.\n")
        self.status_var.set("Cleared paths.")

    def _validate_paths(self) -> None:
        root = self._project_root()
        paths = parse_paths_text(self.paths_text.get("1.0", tk.END))
        if not paths:
            messagebox.showinfo("Validate", "No paths specified. OK (will scan whole repo with exclusions).")
            return

        missing: List[str] = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = root / pp
            if not pp.exists():
                missing.append(p)

        if missing:
            msg = "These paths do NOT exist under the detected project root:\n\n" + "\n".join(missing[:80])
            if len(missing) > 80:
                msg += f"\n... (+{len(missing)-80} more)"
            messagebox.showwarning("Validate - Missing paths", msg)
            self.status_var.set(f"Validation failed: {len(missing)} missing.")
        else:
            messagebox.showinfo("Validate", f"All good. {len(paths)} path(s) exist.")
            self.status_var.set("Validation OK.")

    def _on_drop_paths(self, event) -> None:
        if not DND_AVAILABLE:
            self.status_var.set("Drag&Drop not available.")
            return

        try:
            items = list(self.tk.splitlist(event.data))
        except Exception:
            items = [str(getattr(event, "data", "")).strip()]

        cleaned: List[str] = []
        for it in items:
            s = _normalize_user_path_line(str(it))
            if s:
                cleaned.append(s)
        self._append_paths(cleaned)

    def _open_out_folder(self) -> None:
        if not self._last_out_dir:
            return
        p = self._last_out_dir.resolve()
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.check_call(["open", str(p)])
            else:
                subprocess.check_call(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showerror("Open folder failed", str(e))

    def _copy_summary(self) -> None:
        if not self._last_summary_md:
            return
        self.clipboard_clear()
        self.clipboard_append(self._last_summary_md)
        self.status_var.set("Copied summary.md to clipboard.")

    def _run_audit(self) -> None:
        if not shutil.which("rg"):
            messagebox.showerror("Missing dependency", "ripgrep (rg) not found in PATH.\nInstall it first.")
            return

        root = self._project_root()
        self.root_var.set(str(root))  # show corrected root in UI

        if not root.exists():
            messagebox.showerror("Invalid root", f"Detected project root does not exist:\n{root}")
            return

        include_paths = _dedupe_preserve_order(parse_paths_text(self.paths_text.get("1.0", tk.END)))

        out_dir = root / "audit" / "out"
        _ensure_dir(out_dir)

        for k in self.tabs:
            self._set_tab_text(k, "")

        self.run_btn.configure(state=tk.DISABLED)
        self.open_out_btn.configure(state=tk.DISABLED)
        self.copy_summary_btn.configure(state=tk.DISABLED)
        self.status_var.set(f"Running audit in: {root}")
        self.update_idletasks()

        cmds = build_audit_cmds(include_paths)
        results: List[CmdResult] = []

        for i, cmd in enumerate(cmds, start=1):
            self.status_var.set(f"Running {i}/{len(cmds)}: {cmd.key} ...")
            self.update_idletasks()

            started = _now_iso()
            out_path = out_dir / f"{cmd.key}.txt"
            cwd = root / cmd.cwd_rel

            try:
                exit_code, total_lines, preview = _run_stream_to_file(cmd.argv, cwd=cwd, out_path=out_path)
            except Exception as e:
                exit_code, total_lines, preview = (1, 0, f"[ERROR] {e}\ncmd={_argv_to_str(cmd.argv)}\n")

            finished = _now_iso()
            results.append(
                CmdResult(
                    key=cmd.key,
                    title=cmd.title,
                    argv=cmd.argv,
                    exit_code=exit_code,
                    total_lines=total_lines,
                    preview_text=preview,
                    out_path=out_path,
                    started_at=started,
                    finished_at=finished,
                )
            )

            tab_text = (
                f"{cmd.title}\n"
                f"exit_code={exit_code} lines={total_lines}\n"
                f"out_file={_safe_relpath(out_path, root)}\n"
                f"cmd={_argv_to_str(cmd.argv)}\n\n"
                f"{preview}"
            )
            self._set_tab_text(cmd.key, tab_text)

        summary_md = generate_summary_md(root, results, out_dir, include_paths)
        summary_path = root / "audit" / "summary.md"
        _ensure_dir(summary_path.parent)
        summary_path.write_text(summary_md, encoding="utf-8", errors="replace")

        self._set_tab_text("summary", summary_md)
        self._last_summary_md = summary_md
        self._last_out_dir = out_dir

        self.run_btn.configure(state=tk.NORMAL)
        self.open_out_btn.configure(state=tk.NORMAL)
        self.copy_summary_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Done. Wrote { _safe_relpath(summary_path, root) }")


def main() -> None:
    app = AuditGUI()
    app.mainloop()


if __name__ == "__main__":
    main()