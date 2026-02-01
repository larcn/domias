# FILE: factory_min.py | version: 2026-01-14.p4b3_factory_min_full
#
# هدف هذا الملف:
# - سكربت CLI واحد واضح لتشغيل الدورة الرسمية:
#     Rust self-play -> shards -> check_shards -> train_from_shards -> evaluate.py
# - دعم Presets
# - دعم endgame_mine (record-only mining) عبر تمرير mode/knobs إلى Rust factory
# - لوج واضح + إخراج JSON سطر واحد مناسب للـPanel
# - دعم Cancel عملي:
#     - Ctrl+C يوقف الدورة ويُنهي أي subprocess جارٍ (evaluate)
#     - زر Cancel في panel.py يُنهي عملية factory_min.py نفسها (هذا يكفي لإيقاف Rust generation)

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Literal

# Project root = directory containing this file
ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
EVAL_PY = ROOT / "evaluate.py"

# Import tool helpers (same environment as panel)
sys.path.insert(0, str(ROOT))
from tools import domino_tool  # type: ignore


Mode = Literal["selfplay", "endgame_mine"]


@dataclass(frozen=True)
class Preset:
    name: str
    out_dir: str
    matches: int
    det: int
    think_ms: int
    temp: float
    mode: Mode
    end_close: int
    end_bone: int
    end_hand: int


PRESETS: Dict[str, Preset] = {
    "small": Preset("small", "runs_small", 3000, 10, 800, 0.95, "selfplay", 20, 2, 6),
    "full": Preset("full", "runs_full", 15000, 12, 1200, 0.85, "selfplay", 20, 2, 6),
    "scale": Preset("scale", "runs_scale", 30000, 14, 1400, 0.80, "selfplay", 20, 2, 6),
    # Dedicated endgame mining preset (record-only filter, gameplay unchanged)
    "endmine": Preset("endmine", "runs_endmine", 15000, 12, 1200, 0.85, "endgame_mine", 20, 2, 6),
}


# -----------------------------------------------------------------------------
# Cancel handling
# -----------------------------------------------------------------------------

_CURRENT_SUBPROC: Optional[subprocess.Popen] = None
_CANCELLED = False


def _set_cancelled() -> None:
    global _CANCELLED
    _CANCELLED = True


def _sig_handler(_sig: int, _frame: Any) -> None:
    # Graceful cancel: terminate child subprocess if any, then raise KeyboardInterrupt.
    _set_cancelled()
    if _CURRENT_SUBPROC is not None:
        try:
            _CURRENT_SUBPROC.terminate()
        except Exception:
            pass
    raise KeyboardInterrupt()


def _install_signal_handlers() -> None:
    # Windows: SIGINT works; SIGTERM exists in Python but may behave differently.
    try:
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _sig_handler)  # type: ignore[attr-defined]
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def jprint(obj: Dict[str, Any]) -> None:
    # Panel-friendly one-line JSON
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def ensure_model_json_exists(model_path: Path, hidden: int = 384, seed: int = 12345) -> None:
    """
    Rust self-play requires a model file. If missing, bootstrap a random-init model.json.
    """
    if model_path.exists():
        return
    import train  # local import to avoid import cost unless needed
    m = train.MLP.init(hidden_size=int(hidden), seed=int(seed))
    m.save(str(model_path))
    print(f"[factory_min] bootstrapped missing model: {model_path}", flush=True)


def apply_preset(args: argparse.Namespace) -> None:
    if not args.preset:
        return
    key = str(args.preset).strip().lower()
    if key not in PRESETS:
        raise RuntimeError(f"Unknown preset: {args.preset}. Available: {', '.join(sorted(PRESETS.keys()))}")
    p = PRESETS[key]
    # only fill fields if user didn't explicitly pass overrides
    args.out = args.out or p.out_dir
    if args.matches is None:
        args.matches = p.matches
    if args.det is None:
        args.det = p.det
    if args.think_ms is None:
        args.think_ms = p.think_ms
    if args.temp is None:
        args.temp = p.temp
    if args.mode is None:
        args.mode = p.mode
    if args.end_close is None:
        args.end_close = p.end_close
    if args.end_bone is None:
        args.end_bone = p.end_bone
    if args.end_hand is None:
        args.end_hand = p.end_hand


def run_cmd_capture_json(args: List[str], cwd: Path) -> Dict[str, Any]:
    """
    Run a subprocess and capture the LAST line that parses as JSON.
    We still stream stdout to console for progress visibility.
    """
    global _CURRENT_SUBPROC
    _CURRENT_SUBPROC = None

    last_json: Optional[Dict[str, Any]] = None

    p = subprocess.Popen(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    _CURRENT_SUBPROC = p

    assert p.stdout is not None
    for line in p.stdout:
        s = line.rstrip("\n")
        print(s, flush=True)
        # try parse JSON lines
        if s.startswith("{") and s.endswith("}"):
            try:
                j = json.loads(s)
                if isinstance(j, dict):
                    last_json = j
            except Exception:
                pass

    rc = int(p.wait())
    _CURRENT_SUBPROC = None
    if rc != 0:
        raise RuntimeError(f"subprocess failed rc={rc}: {' '.join(args)}")
    return last_json or {"ok": False, "error": "no_json_output"}


# -----------------------------------------------------------------------------
# Core ops
# -----------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> Path:
    """
    Generate shards via Rust factory: domino_rs.generate_and_save(cfg.json, out_dir) -> manifest path.
    """
    import domino_rs  # type: ignore

    out_dir = (ROOT / str(args.out)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(str(args.model_path)).resolve()
    ensure_model_json_exists(model_path, hidden=int(args.model_hidden), seed=int(args.model_seed))

    mode: Mode = str(args.mode or "selfplay").strip().lower()  # type: ignore[assignment]
    if mode not in ("selfplay", "endgame_mine"):
        mode = "selfplay"

    # Endgame mining defaults (only used when mode=endgame_mine)
    def _end_i(name: str, v: Any, default: int) -> int:
        try:
            if v is None:
                return int(default)
            return int(v)
        except Exception:
            return int(default)

    cfg: Dict[str, Any] = {
        "matches": int(args.matches),
        "mode": str(mode),
        "det": int(args.det),
        "think_ms": int(args.think_ms),
        "temperature": float(args.temp),
        "max_moves_per_round": int(args.max_moves),
        "max_rounds": int(args.max_rounds),
        "match_target": int(args.target),
        "seed": int(args.seed),
        "model_path": str(model_path),
        "codec": str(args.codec),
        "zstd_level": int(args.zstd_level),
        "shard_max_samples": int(args.shard_max_samples),
        "threads": int(args.threads) if args.threads is not None else None,
        "leaf_value_weight": float(args.leaf_value_weight),

        # Strategic knobs (kept but default to 0 in your pipeline)
        "opp_mix_greedy": float(args.opp_mix_greedy),
        "me_mix_greedy": float(args.me_mix_greedy),
        "gift_penalty_weight": float(args.gift_penalty_weight),
        "pessimism_alpha_max": float(args.pessimism_alpha_max),
    }

    # Only attach endgame_mine knobs when actually mining.
    if mode == "endgame_mine":
        cfg["endgame_close_threshold"] = _end_i("end_close", getattr(args, "end_close", None), 20)
        cfg["endgame_boneyard_max"] = _end_i("end_bone", getattr(args, "end_bone", None), 2)
        cfg["endgame_hand_max"] = _end_i("end_hand", getattr(args, "end_hand", None), 6)

    # Remove None entries for cleanliness
    cfg = {k: v for (k, v) in cfg.items() if v is not None}

    cfg_path = out_dir / "cfg.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    manifest_path_str = domino_rs.generate_and_save(str(cfg_path), str(out_dir))  # prints PROGRESS itself
    dt = time.perf_counter() - t0

    mp = Path(str(manifest_path_str)).resolve()
    jprint({
        "time": now_ts(),
        "op": "generate",
        "out_dir": str(out_dir),
        "manifest": str(mp),
        "elapsed_sec": round(float(dt), 3),
        "mode": str(mode),
    })
    return mp


def cmd_generate_rc(args: argparse.Namespace) -> int:
    """
    CLI-friendly wrapper:
    - cmd_generate() returns a Path (useful for internal callers like cmd_cycle)
    - but the 'generate' subcommand must return an int exit code to keep main() stable.
    """
    _ = cmd_generate(args)
    return 0


def cmd_check_shards(args: argparse.Namespace) -> int:
    mp = Path(str(args.manifest)).resolve()
    return int(domino_tool.check_shards(mp, sample_limit=int(args.sample_limit)))


def cmd_train(args: argparse.Namespace) -> int:
    mp = Path(str(args.manifest)).resolve()
    model_path = Path(str(args.model_path)).resolve()
    ensure_model_json_exists(model_path, hidden=int(args.model_hidden), seed=int(args.model_seed))

    # --- decide replay manifests first (so auto-train can use total replay samples) ---
    replay_last_n = int(getattr(args, "replay_last_n", 1) or 1)
    replay_last_n = max(1, min(32, replay_last_n))

    man_dir = mp.parent
    all_m = sorted(man_dir.glob("*.manifest.json"), key=lambda p: p.name)
    if mp not in all_m:
        all_m.append(mp)
        all_m = sorted(all_m, key=lambda p: p.name)

    if replay_last_n > 1:
        manifest_paths = all_m[-replay_last_n:]
        print(f"[replay] using last_n={replay_last_n} manifests in {man_dir.name}", flush=True)
    else:
        manifest_paths = [mp]

    # Load manifest config + compute samples_total over replay set
    def _load_manifest(p: Path) -> dict:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    mans = [_load_manifest(p) for p in manifest_paths]
    samples_total = 0
    record_mode = "selfplay"
    for j, man in enumerate(mans):
        # samples
        s = int(man.get("samples") or 0)
        if s <= 0:
            shards = man.get("shards") or []
            s = int(sum(int(x.get("samples") or 0) for x in shards))
        samples_total += s

        # mode (use first manifest as reference)
        if j == 0:
            cfg = man.get("config") or {}
            if isinstance(cfg, dict):
                record_mode = str(cfg.get("mode") or record_mode)
            record_mode = (record_mode or "selfplay").strip().lower()
            if record_mode not in ("selfplay", "endgame_mine"):
                record_mode = "selfplay"

    # resolve train_mode (spike_only flag overrides)
    train_mode = str(getattr(args, "train_mode", "both") or "both").strip().lower()
    if train_mode not in ("both", "pv_only", "spike_only"):
        train_mode = "both"

    if bool(getattr(args, "spike_only", False)):
        if train_mode != "spike_only":
            print("[WARN] --spike_only is deprecated; forcing train_mode=spike_only", flush=True)
        train_mode = "spike_only"

    # auto schedule: epochs=0 or steps=0
    epochs = int(args.epochs)
    steps_per_epoch = int(args.steps_per_epoch)
    batch = int(args.batch)

    # record_mode and samples_total now come from replay set

    auto_used = False
    if epochs <= 0:
        epochs = 1 if record_mode == "selfplay" else 6
        auto_used = True
    if steps_per_epoch <= 0:
        if record_mode == "endgame_mine":
            # one full pass per epoch over *replay set*
            steps_per_epoch = int(max(10, (samples_total + batch - 1) // batch))
        else:
            target_seen = 200_000
            steps_per_epoch = int(max(50, min(2000, (target_seen + batch - 1) // batch)))
        auto_used = True

    # repeat_dataset: keep your existing string flag semantics
    if args.repeat_dataset == "auto":
        repeat_dataset = None
    else:
        repeat_dataset = (args.repeat_dataset == "true")

    if repeat_dataset is None:
        repeat_dataset = (record_mode == "endgame_mine") or (epochs > 1)
        auto_used = True

    shuffle_each = bool(args.shuffle_shards_each_epoch)
    if repeat_dataset and not shuffle_each:
        shuffle_each = True
        auto_used = True

    if auto_used:
        print(
            f"[auto-train] record_mode={record_mode} replay_samples_total={samples_total} "
            f"train_mode={train_mode} epochs={epochs} steps_per_epoch={steps_per_epoch} "
            f"batch={batch} repeat_dataset={repeat_dataset} shuffle_shards_each_epoch={shuffle_each}",
            flush=True,
        )

    if len(manifest_paths) == 1:
        return int(domino_tool.train_from_shards(
            manifest_path=manifest_paths[0],
            model_path=model_path,
            epochs=int(epochs),
            steps_per_epoch=int(steps_per_epoch),
            batch=int(batch),
            lr=float(args.lr),
            l2=float(args.l2),
            hidden=int(args.model_hidden),
            seed=int(args.model_seed),
            repeat_dataset=repeat_dataset,
            shuffle_shards_each_epoch=bool(shuffle_each),
            shuffle_buffer=int(args.shuffle_buffer),
            train_mode=train_mode,
            spike_only=False,
            spike_pos_w=float(args.spike_pos_w),
        ))

    return int(domino_tool.train_from_manifests(
        manifest_paths=manifest_paths,
        model_path=model_path,
        epochs=int(epochs),
        steps_per_epoch=int(steps_per_epoch),
        batch=int(batch),
        lr=float(args.lr),
        l2=float(args.l2),
        hidden=int(args.model_hidden),
        seed=int(args.model_seed),
        repeat_dataset=repeat_dataset,
        shuffle_shards_each_epoch=bool(shuffle_each),
        shuffle_buffer=int(args.shuffle_buffer),
        train_mode=train_mode,
        spike_only=False,
        spike_pos_w=float(args.spike_pos_w),
    ))


def cmd_eval(args: argparse.Namespace) -> Dict[str, Any]:
    if not EVAL_PY.exists():
        raise RuntimeError(f"missing evaluate.py at {EVAL_PY}")

    cmd = [
        sys.executable,
        str(EVAL_PY),
        "--matches", str(int(args.eval_matches)),
        "--target", str(int(args.target)),
        "--seed", str(int(args.eval_seed)),
        "--me", str(args.eval_me),
        "--level", str(args.eval_level),
        "--det", str(int(args.eval_det)),
        "--think_ms", str(int(args.eval_think_ms)),
        "--opp", str(args.eval_opp),
        "--jobs", str(int(args.eval_jobs)),
        "--assert_every", str(int(args.assert_every)),
        "--progress_every", str(int(args.progress_every)),

        # Keep eval aligned with generation knobs (runtime search behavior)
        "--rust_opp_mix_greedy", str(float(args.opp_mix_greedy)),
        "--rust_me_mix_greedy", str(float(args.me_mix_greedy)),
        "--rust_leaf_value_weight", str(float(args.leaf_value_weight)),
        "--rust_gift_penalty_weight", str(float(args.gift_penalty_weight)),
        "--rust_pessimism_alpha_max", str(float(args.pessimism_alpha_max)),
    ]
    return run_cmd_capture_json(cmd, cwd=ROOT)


def cmd_cycle(args: argparse.Namespace) -> int:
    apply_preset(args)

    out_dir = str(args.out or "runs_small")
    n = int(args.n)

    # Cycle seeds: deterministic but distinct per iteration
    base_seed = int(args.seed)

    # ensure model exists (important for generate)
    model_path = Path(str(args.model_path)).resolve()
    ensure_model_json_exists(model_path, hidden=int(args.model_hidden), seed=int(args.model_seed))

    for i in range(1, n + 1):
        if _CANCELLED:
            print("[cycle] cancelled", flush=True)
            return 130

        print(f"\n=== CYCLE {i}/{n} ===", flush=True)

        # 1) Generate
        g = argparse.Namespace(**vars(args))
        g.out = out_dir
        g.seed = base_seed + (i - 1) * 1000003
        mp = cmd_generate(g)

        # 2) Check shards (+ spike density stats)
        rc = domino_tool.check_shards(mp, sample_limit=int(args.check_sample_limit))
        if rc != 0:
            raise RuntimeError(f"check_shards failed rc={rc}")

        # 3) Train
        tr = argparse.Namespace(**vars(args))
        tr.manifest = str(mp)
        tr.model_path = str(model_path)
        rc = cmd_train(tr)
        if rc != 0:
            raise RuntimeError(f"train_from_shards failed rc={rc}")

        # 4) Eval
        ev = argparse.Namespace(**vars(args))
        rep = cmd_eval(ev)
        jprint({
            "time": now_ts(),
            "op": "cycle",
            "i": i,
            "manifest": str(mp),
            "results": rep.get("results", rep),
        })

    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="factory_min.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Shared defaults
    def add_common_generate_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--preset", type=str, default=None, help=f"preset: {', '.join(sorted(PRESETS.keys()))}")
        p.add_argument("--out", type=str, default=None, help="output dir (relative to project root)")
        p.add_argument("--matches", type=int, default=None)
        p.add_argument("--det", type=int, default=None)
        p.add_argument("--think_ms", type=int, default=None)
        p.add_argument("--temp", type=float, default=None)
        p.add_argument("--seed", type=int, default=99999)

        p.add_argument("--target", type=int, default=150)
        p.add_argument("--max_moves", type=int, default=500)
        p.add_argument("--max_rounds", type=int, default=200)

        # Model bootstrap path
        p.add_argument("--model_path", type=str, default="model.json")
        p.add_argument("--model_hidden", type=int, default=384)
        p.add_argument("--model_seed", type=int, default=12345)

        # Shards config
        p.add_argument("--codec", type=str, default="zstd", choices=["raw", "zstd"])
        p.add_argument("--zstd_level", type=int, default=3)
        p.add_argument("--shard_max_samples", type=int, default=50000)
        p.add_argument("--threads", type=int, default=None)

        # ISMCTS rollout knobs (keep defaults aligned with your current setup)
        p.add_argument("--opp_mix_greedy", type=float, default=1.0)
        p.add_argument("--leaf_value_weight", type=float, default=0.0)
        p.add_argument("--me_mix_greedy", type=float, default=1.0)
        p.add_argument("--gift_penalty_weight", type=float, default=0.0)
        p.add_argument("--pessimism_alpha_max", type=float, default=0.0)

        # NEW: record mode + endgame mining knobs
        p.add_argument("--mode", type=str, default=None, choices=["selfplay", "endgame_mine"])
        p.add_argument("--end_close", type=int, default=None, help="endgame_mine: close threshold (default 20)")
        p.add_argument("--end_bone", type=int, default=None, help="endgame_mine: max boneyard count (default 2)")
        p.add_argument("--end_hand", type=int, default=None, help="endgame_mine: max hand size for both (default 6)")

    def add_common_train_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--train_mode", type=str, default="both", choices=["both", "pv_only", "spike_only"])
        p.add_argument("--spike_only", action="store_true", help="DEPRECATED: use --train_mode spike_only (kept for compatibility)")
        p.add_argument("--epochs", type=int, default=1, help="epochs (0=auto)")
        p.add_argument("--steps_per_epoch", type=int, default=30, help="steps_per_epoch (0=auto)")
        p.add_argument("--batch", type=int, default=256)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--l2", type=float, default=1e-4)
        p.add_argument("--spike_pos_w", type=float, default=20.0, help="POS_W for spike BCE (only affects spike loss).")
        p.add_argument("--repeat_dataset", type=str, default="auto", choices=["auto", "true", "false"])
        p.add_argument("--shuffle_shards_each_epoch", action="store_true")
        p.add_argument("--shuffle_buffer", type=int, default=8192)
        p.add_argument("--replay_last_n", type=int, default=1, help="Replay buffer: train on last N manifests in the same out_dir (default 1).")

    def add_common_eval_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--eval_matches", type=int, default=500)
        p.add_argument("--eval_seed", type=int, default=12345)
        p.add_argument("--eval_me", type=str, default="rust", choices=["rust", "model", "ai", "greedy"])
        p.add_argument("--eval_level", type=str, default="quick", choices=["quick", "standard", "deep", "ismcts"])
        p.add_argument("--eval_det", type=int, default=8)
        p.add_argument("--eval_think_ms", type=int, default=120)
        p.add_argument("--eval_opp", type=str, default="strategic", choices=["greedy", "strategic"])
        p.add_argument("--eval_jobs", type=int, default=6)
        p.add_argument("--assert_every", type=int, default=10)
        p.add_argument("--progress_every", type=int, default=50)

    # cycle
    p_cycle = sub.add_parser("cycle", help="Generate + check_shards + train_from_shards + evaluate (N times)")
    p_cycle.add_argument("--n", type=int, default=1)
    add_common_generate_flags(p_cycle)
    add_common_train_flags(p_cycle)
    add_common_eval_flags(p_cycle)
    p_cycle.add_argument("--check_sample_limit", type=int, default=200, help="samples checked in check_shards")
    p_cycle.set_defaults(func=cmd_cycle)

    # generate
    p_gen = sub.add_parser("generate", help="Generate shards only (Rust factory)")
    add_common_generate_flags(p_gen)
    p_gen.set_defaults(func=cmd_generate_rc)

    # check
    p_chk = sub.add_parser("check", help="Check shards manifest")
    p_chk.add_argument("--manifest", type=str, required=True)
    p_chk.add_argument("--sample_limit", type=int, default=200)
    p_chk.set_defaults(func=cmd_check_shards)

    # train
    p_tr = sub.add_parser("train", help="Train model.json from shards manifest")
    p_tr.add_argument("--manifest", type=str, required=True)
    add_common_train_flags(p_tr)
    p_tr.add_argument("--model_path", type=str, default="model.json")
    p_tr.add_argument("--model_hidden", type=int, default=384)
    p_tr.add_argument("--model_seed", type=int, default=12345)
    p_tr.set_defaults(func=cmd_train)

    # eval
    p_ev = sub.add_parser("eval", help="Run evaluate.py with current model.json")
    p_ev.add_argument("--target", type=int, default=150)
    add_common_eval_flags(p_ev)
    p_ev.set_defaults(func=lambda a: (jprint({"time": now_ts(), "op": "eval", "results": cmd_eval(a).get("results", {})}) or 0))

    # presets list
    p_p = sub.add_parser("presets", help="List presets")
    def _list_presets(_a: argparse.Namespace) -> int:
        print("Available presets:")
        for k in sorted(PRESETS.keys()):
            p = PRESETS[k]
            print(f"  {k:8} out={p.out_dir} matches={p.matches} det={p.det} think_ms={p.think_ms} temp={p.temp} mode={p.mode} end_close={p.end_close} end_bone={p.end_bone} end_hand={p.end_hand}")
        return 0
    p_p.set_defaults(func=_list_presets)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    _install_signal_handlers()
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    # Apply preset for commands that have it
    if hasattr(args, "preset"):
        apply_preset(args)

    try:
        rc = args.func(args)
        return int(rc)
    except KeyboardInterrupt:
        print("[factory_min] cancelled (KeyboardInterrupt)", flush=True)
        return 130
    finally:
        # Ensure child process is terminated if still running
        global _CURRENT_SUBPROC
        if _CURRENT_SUBPROC is not None:
            try:
                _CURRENT_SUBPROC.terminate()
            except Exception:
                pass
            _CURRENT_SUBPROC = None


if __name__ == "__main__":
    raise SystemExit(main())