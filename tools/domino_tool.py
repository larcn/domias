# FILE: tools/domino_tool.py | version: 2026-01-14.p4b3_fixed_full
# Gates + Shards tooling for Domino AI (Rust self-play pipeline)
#
# This version matches DSH2 shard format:
#   feat: 193*f32
#   pi  : 112*f32
#   spike: 112*i8        (per-action win-now-exists target)
#   z   : f32
#   mask: 112*i8
#
# Includes:
# - Conformance gate (Python engine vs Rust state_apply_script)
# - Feature gate (Python features_small/legal_mask_state vs Rust features_from_state_dict)
# - Data gate (check_shards + spike density stats)
# - Speed gate (bench generation)
# - Training smoke (train_from_shards)
#
# FIX: train_from_shards repeat_dataset path always defines rng (prevents UnboundLocalError).

from __future__ import annotations

import argparse
import json
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Make console output robust on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root import
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import zstandard as zstd

import engine
import ai
import train

# Rust extension required for conformance/feature/generation gates
import domino_rs  # type: ignore

FEAT_DIM = 193
ACTION_SIZE = 112

MAGIC = b"DSH2"
HEADER_LEN = 16

# DSH2 record layout:
# feat: 193*f32 + pi: 112*f32 + spike: 112*i8 + z: f32 + mask: 112*i8
RECORD_SIZE = (
    FEAT_DIM * 4
    + ACTION_SIZE * 4
    + ACTION_SIZE * 1
    + 4
    + ACTION_SIZE * 1
)


def _apply_script_python_obj(my_hand: List[str], match_target: int, script: List[str]) -> engine.GameState:
    st = engine.GameState()
    hand = [engine.parse_tile(x) for x in my_hand]
    st.start_new_game(hand, match_target=match_target)

    for line in script:
        parts = line.split("|")
        op = parts[0]
        if op == "play":
            _, player, tile, end = parts
            st.play_tile(player, engine.parse_tile(tile), end)
        elif op == "pass":
            _, player, cert = parts
            st.record_pass(player, certainty=cert)
        elif op == "draw_me":
            _, tile = parts
            st.record_draw_me(engine.parse_tile(tile))
        elif op == "draw":
            _, player, count, cert = parts
            st.record_draw(player, count=int(count), certainty=cert)
        elif op == "finalize_out":
            _, p = parts
            st.finalize_out_with_opponent_pips(int(p))
        elif op == "declare_locked":
            _, p = parts
            st.declare_locked(int(p))
        elif op == "set_scores":
            _, my, opp = parts
            st.my_score = int(my)
            st.opponent_score = int(opp)
        elif op == "set_turn":
            _, turn = parts
            st.current_turn = turn
        elif op == "start_new_round":
            _, tiles = parts
            hand2 = [engine.parse_tile(x) for x in tiles.split(",") if x.strip()]
            st.start_new_round(hand2)
        elif op == "set_hand_counts":
            _, hand_csv, opp_cnt, bone_cnt = parts
            hand2 = [engine.parse_tile(x.strip()) for x in hand_csv.split(",") if x.strip()]
            st.my_hand = set(hand2)
            st.opponent_tile_count = int(opp_cnt)
            st.boneyard_count = int(bone_cnt)
            if st.forced_play_tile is not None and st.forced_play_tile not in st.my_hand:
                st.forced_play_tile = None
        else:
            raise RuntimeError(f"unknown op: {op}")

    return st


def run_conformance() -> int:
    diffs: List[str] = []
    hand_best_64 = ["6-4", "6-3", "6-2", "6-1", "6-0", "5-4", "5-3"]

    scenarios: List[Tuple[str, int, List[str]]] = [
        ("target_immediate_guaranteed_first_move", 150, ["set_scores|145|0", "play|me|6-4|right"]),
        ("opening_rule_enforced_error", 150, ["play|me|6-3|right"]),
        ("out_me_pending_then_finalize_out", 150, ["set_scores|0|0", "set_hand_counts|6-4|7|20", "set_turn|me", "play|me|6-4|right", "finalize_out|23"]),
        ("locked_evidence_and_declare", 150, ["set_scores|0|0", "set_hand_counts|0-0|7|0", "set_turn|opponent", "pass|opponent|certain", "declare_locked|17"]),
    ]

    for name, target, script in scenarios:
        py_err = None
        rs_err = None
        py_state = None
        rs_state = None

        try:
            py_state = _apply_script_python_obj(hand_best_64, target, script).to_dict()
        except Exception as e:
            py_err = str(e)

        try:
            rs_state = domino_rs.state_apply_script(hand_best_64, int(target), script)  # type: ignore[attr-defined]
        except Exception as e:
            rs_err = str(e)

        if py_err or rs_err:
            if (py_err is None) != (rs_err is None):
                diffs.append(f"[{name}] error mismatch:\n  PY={py_err}\n  RS={rs_err}")
            continue

        pm = (py_state.get("meta") or {})
        rm = (rs_state.get("meta") or {})
        for k in ("my_score", "opponent_score", "round_over", "round_end_reason", "pending_out_opponent_pips"):
            if pm.get(k) != rm.get(k):
                diffs.append(f"[{name}] meta {k} mismatch PY={pm.get(k)} RS={rm.get(k)}")

    if diffs:
        print("[FAIL] Conformance FAILED")
        for d in diffs[:40]:
            print(d)
        return 2

    print("[OK] Conformance OK")
    return 0


def run_feature_gate() -> int:
    hand_best_64 = ["6-4", "6-3", "6-2", "6-1", "6-0", "5-4", "5-3"]
    scenarios: List[Tuple[str, int, List[str]]] = [
        ("feat_start", 150, []),
        ("feat_after_open", 150, ["play|me|6-4|right"]),
        ("feat_with_opp_pass", 150, ["play|me|6-4|right", "set_turn|opponent", "pass|opponent|probable"]),
        ("feat_with_opp_draw", 150, ["play|me|6-4|right", "set_turn|opponent", "draw|opponent|2|probable"]),
    ]

    bad = 0
    for name, target, script in scenarios:
        st = _apply_script_python_obj(hand_best_64, target, script)
        py_feat = ai.features_small(st).astype(np.float32)
        py_mask = ai.legal_mask_state(st).astype(np.int8)

        feat_b, mask_b = domino_rs.features_from_state_dict(st.to_dict())  # type: ignore[attr-defined]
        rs_feat = np.frombuffer(feat_b, dtype=np.float32, count=FEAT_DIM)
        rs_mask = np.frombuffer(mask_b, dtype=np.int8, count=ACTION_SIZE)

        if not np.array_equal(py_mask, rs_mask):
            print(f"[{name}] MASK mismatch")
            bad += 1

        diff = np.abs(py_feat - rs_feat)
        if float(diff.max()) > 1e-5:
            i = int(diff.argmax())
            print(f"[{name}] FEAT mismatch max={float(diff.max()):.6g} at i={i}")
            bad += 1

    if bad:
        print(f"[FAIL] Feature Gate FAILED bad={bad}")
        return 3

    print("[OK] Feature Gate OK")
    return 0


def _read_shard_bytes(path: Path, codec: str) -> bytes:
    codec = (codec or "").strip().lower()
    is_zst = (codec in ("zstd", "zst")) or (path.suffix.lower() == ".zst") or path.name.lower().endswith(".dsh.zst")
    if not is_zst:
        return path.read_bytes()

    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            return reader.read()


def check_shards(manifest_path: Path, sample_limit: int = 200) -> int:
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = man.get("shards") or []
    if not shards:
        print("[FAIL] No shards listed in manifest")
        return 2

    total = 0
    checked = 0
    bad = 0

    # spike density stats over checked subset
    spike_legal_total = 0
    spike_legal_pos = 0
    states_any_unsafe = 0

    for sh in shards:
        fn = sh.get("filename") or ""
        codec = sh.get("codec") or man.get("codec") or "raw"
        if not fn:
            bad += 1
            continue

        p = manifest_path.parent / fn
        if not p.exists():
            print("[FAIL] Missing shard:", p)
            bad += 1
            continue

        data = _read_shard_bytes(p, codec)
        if len(data) < HEADER_LEN:
            print("[FAIL] Shard too small:", p)
            bad += 1
            continue
        if data[:4] != MAGIC:
            print("[FAIL] Bad shard magic:", p, data[:4])
            bad += 1
            continue

        feat_dim = int.from_bytes(data[4:8], "little")
        act = int.from_bytes(data[8:12], "little")
        rec = int.from_bytes(data[12:16], "little")
        if feat_dim != FEAT_DIM or act != ACTION_SIZE or rec != RECORD_SIZE:
            print("[FAIL] Header mismatch:", p, feat_dim, act, rec)
            bad += 1
            continue

        payload = data[HEADER_LEN:]
        if len(payload) % RECORD_SIZE != 0:
            print("[FAIL] Payload not divisible by record size:", p, len(payload))
            bad += 1
            continue

        nrec = len(payload) // RECORD_SIZE
        total += nrec

        for i in range(nrec):
            if checked >= sample_limit:
                break
            off = i * RECORD_SIZE

            _feat = np.frombuffer(payload, dtype=np.float32, count=FEAT_DIM, offset=off)
            off += FEAT_DIM * 4
            pi = np.frombuffer(payload, dtype=np.float32, count=ACTION_SIZE, offset=off)
            off += ACTION_SIZE * 4
            spike = np.frombuffer(payload, dtype=np.int8, count=ACTION_SIZE, offset=off)
            off += ACTION_SIZE * 1
            z = float(np.frombuffer(payload, dtype=np.float32, count=1, offset=off)[0])
            off += 4
            mask = np.frombuffer(payload, dtype=np.int8, count=ACTION_SIZE, offset=off)

            if z < -1.0001 or z > 1.0001:
                bad += 1
            s = float(pi.sum())
            if abs(s - 1.0) > 1e-3:
                bad += 1
            if float(pi[mask == 0].sum()) > 1e-6:
                bad += 1
            if int(mask.sum()) <= 0:
                bad += 1

            # spike sanity: only check legal actions (mask==1) are 0/1
            if not np.all((spike[mask == 1] == 0) | (spike[mask == 1] == 1)):
                bad += 1

            # spike density stats
            legal_n = int(mask.sum())
            if legal_n > 0:
                spike_legal_total += legal_n
                pos = int((spike[mask == 1] == 1).sum())
                spike_legal_pos += pos
                if pos > 0:
                    states_any_unsafe += 1

            checked += 1

    if bad:
        print(f"[FAIL] check_shards FAILED bad={bad} total_records={total} checked={checked}")
        return 4

    denom_legal = float(max(1, spike_legal_total))
    denom_states = float(max(1, checked))
    spike_pos_rate_legal = float(spike_legal_pos) / denom_legal
    states_any_unsafe_rate = float(states_any_unsafe) / denom_states
    avg_unsafe_actions_per_state = float(spike_legal_pos) / denom_states

    print(f"[OK] check_shards OK total_records={total} checked={checked}")
    print(f"[STATS] spike_pos_rate_legal={spike_pos_rate_legal:.6f} states_with_any_unsafe_rate={states_any_unsafe_rate:.6f} avg_unsafe_actions_per_state={avg_unsafe_actions_per_state:.4f}")
    return 0


def generate(out_dir: Path, matches: int, det: int, think_ms: int, temp: float, model_path: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "matches": int(matches),
        "mode": "selfplay",
        "det": int(det),
        "think_ms": int(think_ms),
        "temperature": float(temp),
        "max_moves_per_round": 500,
        "max_rounds": 200,
        "match_target": 150,
        "seed": int(time.time() * 1000) % (2**31),
        "model_path": str(model_path),
        "codec": "zstd",
        "zstd_level": 3,
        "shard_max_samples": 50000,
    }
    cfg_path = out_dir / "_cfg.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = domino_rs.generate_and_save(str(cfg_path), str(out_dir))  # type: ignore[attr-defined]
    mp = Path(str(manifest_path))
    (out_dir / "_last_manifest.json").write_text(json.dumps({"manifest": str(mp)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return mp


def train_from_shards(
    manifest_path: Path,
    model_path: Path,
    epochs: int = 1,
    steps_per_epoch: int = 30,
    batch: int = 256,
    lr: float = 1e-3,
    l2: float = 1e-4,
    hidden: int = 384,
    seed: int = 12345,
    repeat_dataset: Optional[bool] = None,
    shuffle_shards_each_epoch: bool = True,
    shuffle_buffer: int = 8192,
    train_mode: str = "both",
    spike_only: bool = False,
    spike_pos_w: float = 20.0,
) -> int:
    if repeat_dataset is None:
        repeat_dataset = (int(epochs) > 1)

    # resolve train_mode (spike_only flag overrides)
    mode = str(train_mode or "both").strip().lower()
    if spike_only:
        mode = "spike_only"
    if mode not in ("both", "pv_only", "spike_only"):
        raise ValueError(f"bad train_mode={train_mode!r} (expected both|pv_only|spike_only)")

    try:
        m = train.MLP.load(str(model_path))
    except Exception:
        m = train.MLP.init(hidden_size=int(hidden), seed=int(seed))

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = man.get("shards") or []
    if not shards:
        print("[FAIL] No shards in manifest")
        return 2

    def record_iter(shards_order: List[dict], rng: random.Random):
        sb = max(0, int(shuffle_buffer))
        buf: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []

        for sh in shards_order:
            fn = sh.get("filename") or ""
            codec = sh.get("codec") or man.get("codec") or "raw"
            data = _read_shard_bytes(manifest_path.parent / fn, codec)
            payload = data[HEADER_LEN:]
            nrec = len(payload) // RECORD_SIZE

            for i in range(nrec):
                off = i * RECORD_SIZE
                feat = np.frombuffer(payload, dtype=np.float32, count=FEAT_DIM, offset=off).copy()
                off += FEAT_DIM * 4
                pi = np.frombuffer(payload, dtype=np.float32, count=ACTION_SIZE, offset=off).copy()
                off += ACTION_SIZE * 4
                spike = np.frombuffer(payload, dtype=np.int8, count=ACTION_SIZE, offset=off).copy()
                off += ACTION_SIZE * 1
                z = float(np.frombuffer(payload, dtype=np.float32, count=1, offset=off)[0])
                off += 4
                mask = np.frombuffer(payload, dtype=np.int8, count=ACTION_SIZE, offset=off).copy()

                if sb <= 0:
                    yield feat, pi, spike, z, mask
                    continue

                item = (feat, pi, spike, z, mask)
                if len(buf) < sb:
                    buf.append(item)
                    continue

                j = rng.randrange(0, len(buf))
                out = buf[j]
                buf[j] = item
                yield out

        if sb > 0 and buf:
            rng.shuffle(buf)
            for out in buf:
                yield out

    it = None
    if not repeat_dataset:
        rng0 = random.Random(int(seed) + 99991)
        it = record_iter(shards, rng0)

    for ep in range(int(epochs)):
        if repeat_dataset:
            order = list(shards)
            rng = random.Random(int(seed) + 1000 * ep)  # ALWAYS defined (fix)
            if shuffle_shards_each_epoch:
                rng.shuffle(order)
            it = record_iter(order, rng)

        steps_done = 0
        seen_samples = 0

        for step in range(int(steps_per_epoch)):
            X = np.zeros((batch, FEAT_DIM), np.float32)
            PI = np.zeros((batch, ACTION_SIZE), np.float32)
            SPIKE = np.zeros((batch, ACTION_SIZE), np.float32)
            Z = np.zeros((batch,), np.float32)
            M = np.zeros((batch, ACTION_SIZE), np.int8)

            got = 0
            for i in range(batch):
                try:
                    feat, pi, spike, z, mask = next(it)  # type: ignore[arg-type]
                except StopIteration:
                    break
                X[i] = feat
                PI[i] = pi
                SPIKE[i] = spike.astype(np.float32)
                Z[i] = z
                M[i] = mask
                got += 1

            if got < max(8, batch // 8):
                if not repeat_dataset:
                    print(f"[train] dataset exhausted at ep={ep+1} step={step+1}/{steps_per_epoch} (got={got})")
                break

            X = X[:got]
            PI = PI[:got]
            SPIKE = SPIKE[:got]
            Z = Z[:got]
            M = M[:got]

            losses = m.train_batch(
                X, PI, Z, M,
                SPIKE=(None if mode == "pv_only" else SPIKE),
                train_mode=mode,
                learning_rate=float(lr),
                l2_lambda=float(l2),
                spike_pos_w=float(spike_pos_w),
            )

            steps_done += 1
            seen_samples += int(got)

            if (step % 10) == 0:
                pl = float(losses.get("policy_loss", 0.0))
                vl = float(losses.get("value_loss", 0.0))
                sl = float(losses.get("spike_loss", 0.0))
                print(f"[train] ep={ep+1} step={step+1}/{steps_per_epoch} policy_loss={pl:.4f} value_loss={vl:.4f} spike_loss={sl:.4f}")

        print(f"[train] ep_done={ep+1}/{epochs} steps_done={steps_done}/{steps_per_epoch} repeat_dataset={repeat_dataset} seen_samples≈{seen_samples}")

        if not repeat_dataset and steps_done < max(1, int(steps_per_epoch) // 4):
            print("[train] stopping early: non-repeat dataset and not enough remaining samples.")
            break

    m.save(str(model_path))
    print("[OK] train_from_shards wrote:", model_path)
    return 0


def train_from_manifests(
    manifest_paths: List[Path],
    model_path: Path,
    epochs: int = 1,
    steps_per_epoch: int = 30,
    batch: int = 256,
    lr: float = 1e-3,
    l2: float = 1e-4,
    hidden: int = 384,
    seed: int = 12345,
    repeat_dataset: Optional[bool] = None,
    shuffle_shards_each_epoch: bool = True,
    shuffle_buffer: int = 8192,
    train_mode: str = "both",
    spike_only: bool = False,
    spike_pos_w: float = 20.0,
) -> int:
    # Normalize + validate
    mps = [Path(x).resolve() for x in (manifest_paths or [])]
    mps = [p for p in mps if p.exists()]
    if not mps:
        print("[FAIL] train_from_manifests: no valid manifest paths")
        return 2

    # Load model
    try:
        m = train.MLP.load(str(model_path))
    except Exception:
        m = train.MLP.init(hidden_size=int(hidden), seed=int(seed))

    mode = str(train_mode or "both").strip().lower()
    if spike_only:
        mode = "spike_only"
    if mode not in ("both", "pv_only", "spike_only"):
        raise ValueError(f"bad train_mode={train_mode!r} (expected both|pv_only|spike_only)")

    # Auto repeat_dataset default
    if repeat_dataset is None:
        repeat_dataset = (int(epochs) > 1)

    # Preload manifests metadata once
    loaded: List[Tuple[Path, Dict[str, Any], List[dict]]] = []
    total_samples = 0
    for mp in mps:
        man = json.loads(mp.read_text(encoding="utf-8"))
        shards = man.get("shards") or []
        if not shards:
            continue
        loaded.append((mp, man, shards))
        # best effort sample count
        total_samples += int(man.get("samples") or sum(int(s.get("samples") or 0) for s in shards))

    if not loaded:
        print("[FAIL] train_from_manifests: all manifests empty/bad")
        return 2

    print(f"[replay] manifests={len(loaded)} total_samples≈{total_samples} train_mode={mode} repeat_dataset={repeat_dataset}", flush=True)
    for mp, man, _sh in loaded[:12]:
        print(f"  [replay] {mp.name} samples≈{int(man.get('samples') or 0)}", flush=True)
    if len(loaded) > 12:
        print(f"  [replay] ... ({len(loaded)-12} more)", flush=True)

    def record_iter_for_manifest(manifest_path: Path, man: Dict[str, Any], shards_order: List[dict], rng: random.Random):
        sb = max(0, int(shuffle_buffer))
        buf: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []

        for sh in shards_order:
            fn = sh.get("filename") or ""
            codec = sh.get("codec") or man.get("codec") or "raw"
            data = _read_shard_bytes(manifest_path.parent / fn, codec)
            payload = data[HEADER_LEN:]
            nrec = len(payload) // RECORD_SIZE

            for i in range(nrec):
                off = i * RECORD_SIZE
                feat = np.frombuffer(payload, dtype=np.float32, count=FEAT_DIM, offset=off).copy()
                off += FEAT_DIM * 4
                pi = np.frombuffer(payload, dtype=np.float32, count=ACTION_SIZE, offset=off).copy()
                off += ACTION_SIZE * 4
                spike = np.frombuffer(payload, dtype=np.int8, count=ACTION_SIZE, offset=off).copy()
                off += ACTION_SIZE * 1
                z = float(np.frombuffer(payload, dtype=np.float32, count=1, offset=off)[0])
                off += 4
                mask = np.frombuffer(payload, dtype=np.int8, count=ACTION_SIZE, offset=off).copy()

                if sb <= 0:
                    yield feat, pi, spike, z, mask
                    continue

                item = (feat, pi, spike, z, mask)
                if len(buf) < sb:
                    buf.append(item)
                    continue

                j = rng.randrange(0, len(buf))
                out = buf[j]
                buf[j] = item
                yield out

        if sb > 0 and buf:
            rng.shuffle(buf)
            for out in buf:
                yield out

    def record_iter_all(order_loaded: List[Tuple[Path, Dict[str, Any], List[dict]]], rng: random.Random):
        for (mpath, man, shards) in order_loaded:
            shards_order = list(shards)
            if shuffle_shards_each_epoch:
                rng.shuffle(shards_order)
            yield from record_iter_for_manifest(mpath, man, shards_order, rng)

    it = None
    if not repeat_dataset:
        rng0 = random.Random(int(seed) + 99991)
        # fixed order once
        it = record_iter_all(loaded, rng0)

    for ep in range(int(epochs)):
        if repeat_dataset:
            order_loaded = list(loaded)
            rng = random.Random(int(seed) + 1000 * ep)
            if shuffle_shards_each_epoch:
                rng.shuffle(order_loaded)  # shuffle manifests order too
            it = record_iter_all(order_loaded, rng)

        steps_done = 0
        seen_samples = 0

        for step in range(int(steps_per_epoch)):
            X = np.zeros((batch, FEAT_DIM), np.float32)
            PI = np.zeros((batch, ACTION_SIZE), np.float32)
            SPIKE = np.zeros((batch, ACTION_SIZE), np.float32)
            Z = np.zeros((batch,), np.float32)
            M = np.zeros((batch, ACTION_SIZE), np.int8)

            got = 0
            for i in range(batch):
                try:
                    feat, pi, spike, z, mask = next(it)  # type: ignore[arg-type]
                except StopIteration:
                    break
                X[i] = feat
                PI[i] = pi
                SPIKE[i] = spike.astype(np.float32)
                Z[i] = z
                M[i] = mask
                got += 1

            if got < max(8, batch // 8):
                if not repeat_dataset:
                    print(f"[train] dataset exhausted at ep={ep+1} step={step+1}/{steps_per_epoch} (got={got})")
                break

            X = X[:got]
            PI = PI[:got]
            SPIKE = SPIKE[:got]
            Z = Z[:got]
            M = M[:got]

            losses = m.train_batch(
                X, PI, Z, M,
                SPIKE=(None if mode == "pv_only" else SPIKE),
                train_mode=mode,
                learning_rate=float(lr),
                l2_lambda=float(l2),
                spike_pos_w=float(spike_pos_w),
            )

            steps_done += 1
            seen_samples += int(got)

            if (step % 10) == 0:
                pl = float(losses.get("policy_loss", 0.0))
                vl = float(losses.get("value_loss", 0.0))
                sl = float(losses.get("spike_loss", 0.0))
                print(f"[train] ep={ep+1} step={step+1}/{steps_per_epoch} mode={mode} policy_loss={pl:.4f} value_loss={vl:.4f} spike_loss={sl:.4f}")

        print(f"[train] ep_done={ep+1}/{epochs} steps_done={steps_done}/{steps_per_epoch} repeat_dataset={repeat_dataset} seen_samples≈{seen_samples}")

        if not repeat_dataset and steps_done < max(1, int(steps_per_epoch) // 4):
            print("[train] stopping early: non-repeat dataset and not enough remaining samples.")
            break

    m.save(str(model_path))
    print("[OK] train_from_manifests wrote:", model_path)
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    rc = run_conformance()
    if rc != 0:
        return rc

    rc = run_feature_gate()
    if rc != 0:
        return rc

    out = ROOT / "runs_smoke"
    mp = generate(out, matches=10, det=8, think_ms=250, temp=0.95, model_path="model.json")
    rc = check_shards(mp, sample_limit=200)
    if rc != 0:
        return rc

    t0 = time.perf_counter()
    _mp2 = generate(ROOT / "runs_bench", matches=50, det=8, think_ms=250, temp=0.95, model_path="model.json")
    dt = time.perf_counter() - t0
    mps = 50.0 / max(1e-9, dt)
    print(f"[OK] bench: matches=50 elapsed={dt:.3f}s throughput={mps:.2f} matches/s")

    sm = ROOT / "_smoke_model.json"
    if (ROOT / "model.json").exists():
        sm.write_text((ROOT / "model.json").read_text(encoding="utf-8"), encoding="utf-8")

    rc = train_from_shards(mp, sm, epochs=1, steps_per_epoch=20, batch=128, lr=0.001, l2=1e-4, hidden=384, seed=12345)
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="domino_tool.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="Run gates: conformance + feature + data + speed + train-smoke")
    p_check.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())