# FILE: tools/infer_tool.py | version: 2026-01-17.im1_inf1_tool
# Sidecar tooling for INF1 inference shards:
# - check: validate shard headers + basic label sanity
# - train: train a small inference MLP (numpy) to predict opp_hand_mask28
# - pilot: check + train (single command)

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]

INF_MAGIC = b"INF1"
HEADER_LEN = 16

INF_FEAT_DIM = 235
LABEL_SIZE = 28
RECORD_SIZE = INF_FEAT_DIM * 4 + LABEL_SIZE * 1 + 2


def _read_bytes(path: Path, codec: str) -> bytes:
    codec = (codec or "").strip().lower()
    is_zst = (codec in ("zstd", "zst")) or path.suffix.lower() == ".zst" or path.name.lower().endswith(".zst")
    if not is_zst:
        return path.read_bytes()
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            return reader.read()


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))

def _infer_samples_from_manifest(manifest_path: Path) -> int:
    man = _load_manifest(manifest_path)
    s = man.get("infer_samples", None)
    try:
        if s is not None:
            return int(s)
    except Exception:
        pass
    # best-effort fallback from shard list
    shards = man.get("infer_shards") or []
    try:
        return int(sum(int(x.get("samples") or 0) for x in shards))
    except Exception:
        return 0


def _iter_records_from_manifest(manifest_path: Path, sample_cap: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray, int, int]]:
    man = _load_manifest(manifest_path)
    shards = man.get("infer_shards") or []
    if not shards:
        raise RuntimeError("manifest has no infer_shards (INF1 not generated)")

    codec = man.get("infer_codec") or man.get("codec") or "raw"
    got = 0

    for sh in shards:
        fn = sh.get("filename") or ""
        sh_codec = sh.get("codec") or codec
        if not fn:
            continue
        p = manifest_path.parent / fn
        data = _read_bytes(p, sh_codec)
        if len(data) < HEADER_LEN:
            raise RuntimeError(f"INF shard too small: {p}")
        if data[:4] != INF_MAGIC:
            raise RuntimeError(f"bad INF magic: {p} got={data[:4]!r}")

        feat_dim = int.from_bytes(data[4:8], "little")
        lab = int.from_bytes(data[8:12], "little")
        rec = int.from_bytes(data[12:16], "little")
        if feat_dim != INF_FEAT_DIM or lab != LABEL_SIZE or rec != RECORD_SIZE:
            raise RuntimeError(f"INF header mismatch: {p} feat={feat_dim} lab={lab} rec={rec}")

        payload = data[HEADER_LEN:]
        if len(payload) % RECORD_SIZE != 0:
            raise RuntimeError(f"INF payload not divisible: {p} bytes={len(payload)} rec={RECORD_SIZE}")
        nrec = len(payload) // RECORD_SIZE

        for i in range(nrec):
            if sample_cap is not None and got >= sample_cap:
                return
            off = i * RECORD_SIZE
            feat = np.frombuffer(payload, dtype=np.float32, count=INF_FEAT_DIM, offset=off).copy()
            off += INF_FEAT_DIM * 4
            label = np.frombuffer(payload, dtype=np.int8, count=LABEL_SIZE, offset=off).copy()
            off += LABEL_SIZE * 1
            opp_cnt = int(payload[off])
            bone_cnt = int(payload[off + 1])
            got += 1
            yield feat, label, opp_cnt, bone_cnt


def cmd_check(args: argparse.Namespace) -> int:
    mp = Path(args.manifest).resolve()
    n = int(args.sample_limit)
    ok = 0
    bad = 0

    sum_ones = 0
    sum_opp_cnt = 0

    for feat, label, opp_cnt, _bone_cnt in _iter_records_from_manifest(mp, sample_cap=n):
        # label should be 0/1
        if not np.all((label == 0) | (label == 1)):
            bad += 1
            continue

        ones = int(label.sum())
        sum_ones += ones
        sum_opp_cnt += int(opp_cnt)

        # visible tiles from base feat: my_hand(0..27) + played_set(28..55)
        visible = (feat[0:28] + feat[28:56]) > 0.5
        if int(label[visible].sum()) != 0:
            bad += 1
            continue

        ok += 1

    denom = max(1, ok + bad)
    avg_label = float(sum_ones) / float(max(1, ok))
    avg_opp = float(sum_opp_cnt) / float(max(1, ok))

    print(json.dumps({
        "ok": True,
        "checked": int(ok + bad),
        "passed": int(ok),
        "failed": int(bad),
        "avg_label_ones": round(avg_label, 3),
        "avg_opp_cnt": round(avg_opp, 3),
        "fail_rate": round(float(bad) / float(denom), 4),
    }, ensure_ascii=False))

    return 0 if bad == 0 else 2


class InferenceMLP:
    def __init__(self, feat_dim: int, hidden: int, seed: int = 12345) -> None:
        rng = np.random.RandomState(int(seed))

        def xavier(out_dim: int, in_dim: int) -> np.ndarray:
            limit = math.sqrt(6.0 / float(in_dim + out_dim))
            return rng.uniform(-limit, limit, size=(out_dim, in_dim)).astype(np.float32)

        self.feat_dim = int(feat_dim)
        self.hidden = int(hidden)

        self.W1 = xavier(self.hidden, self.feat_dim)
        self.b1 = np.zeros((self.hidden,), np.float32)
        self.W2 = xavier(LABEL_SIZE, self.hidden)
        self.b2 = np.zeros((LABEL_SIZE,), np.float32)

        self.t = 0
        self.adam: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def save(self, path: Path) -> None:
        j = {
            "type": "infer_mlp_v1",
            "feat_dim": int(self.feat_dim),
            "label_size": int(LABEL_SIZE),
            "hidden": int(self.hidden),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }
        path.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1.T + self.b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ self.W2.T + self.b2
        return z1, h1, logits

    def _adam_update(self, name: str, param: np.ndarray, grad: np.ndarray, lr: float, b1: float = 0.9, b2: float = 0.999) -> np.ndarray:
        m, v = self.adam.get(name, (np.zeros_like(param), np.zeros_like(param)))
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * (grad * grad)
        self.adam[name] = (m, v)
        t = max(1, int(self.t))
        mh = m / (1.0 - (b1 ** t))
        vh = v / (1.0 - (b2 ** t))
        return param - lr * mh / (np.sqrt(vh) + 1e-8)

    def train_step(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        visible_mask: np.ndarray,
        lr: float,
        l2: float,
        pos_w: float,
    ) -> Dict[str, float]:
        self.t += 1
        bs = int(X.shape[0])
        z1, h1, logits = self.forward(X)

        # mask out visible tiles from loss/grad (they are known)
        train_mask = (~visible_mask).astype(np.float32)  # [B,28]

        # BCEWithLogits (weighted positives)
        # loss = - (pos_w*y*log(sig) + (1-y)*log(1-sig))
        sig = self._sigmoid(logits)
        eps = 1e-6
        loss_mat = -(pos_w * Y * np.log(sig + eps) + (1.0 - Y) * np.log(1.0 - sig + eps))
        denom = float(np.maximum(1.0, np.sum(train_mask)))
        loss = float(np.sum(loss_mat * train_mask) / denom)

        # dL/dlogits for sigmoid+bce:
        # for positives: pos_w*(sig - 1), for negatives: sig
        dlog = np.where(Y >= 0.5, pos_w * (sig - 1.0), sig) * train_mask
        dlog = dlog / denom

        gW2 = dlog.T @ h1
        gb2 = np.sum(dlog, axis=0)

        dh1 = dlog @ self.W2
        dz1 = dh1 * (z1 > 0)

        gW1 = dz1.T @ X
        gb1 = np.sum(dz1, axis=0)

        # L2
        gW2 += l2 * self.W2
        gW1 += l2 * self.W1

        self.W2 = self._adam_update("W2", self.W2, gW2.astype(np.float32), lr)
        self.b2 = self._adam_update("b2", self.b2, gb2.astype(np.float32), lr)
        self.W1 = self._adam_update("W1", self.W1, gW1.astype(np.float32), lr)
        self.b1 = self._adam_update("b1", self.b1, gb1.astype(np.float32), lr)

        return {"loss": loss}


def _topk_accuracy(probs: np.ndarray, y: np.ndarray, opp_cnt: int, visible: np.ndarray) -> float:
    # exclude visible tiles from ranking
    p = probs.copy()
    p[visible] = -1.0
    k = max(1, int(opp_cnt))
    idx = np.argsort(-p)[:k]
    hits = float(np.sum(y[idx] >= 0.5))
    return hits / float(k)

def _baseline_topk_expected(opp_cnt: int, visible: np.ndarray) -> float:
    # Random ranking among hidden tiles: expected hit fraction = k / n_hidden = opp_cnt / (28 - visible_count)
    n_hidden = int(28 - int(np.sum(visible)))
    k = int(max(1, opp_cnt))
    if n_hidden <= 0:
        return 0.0
    return float(k) / float(n_hidden)

def _reservoir_sample_val(manifest_path: Path, n: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
    # Reservoir sampling over the full stream (single pass). Deterministic by seed.
    rng = random.Random(int(seed) ^ 0xA5A51234)
    out: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    i = 0
    for rec in _iter_records_from_manifest(manifest_path, sample_cap=None):
        i += 1
        if len(out) < n:
            out.append(rec)
            continue
        j = rng.randrange(0, i)
        if j < n:
            out[j] = rec
    return out


def cmd_train(args: argparse.Namespace) -> int:
    mp = Path(args.manifest).resolve()
    out_path = Path(args.out).resolve()

    hidden = int(args.hidden)
    lr = float(args.lr)
    l2 = float(args.l2)
    pos_w = float(args.pos_w)
    steps = int(args.steps)
    batch = int(args.batch)
    shuffle_buffer = int(args.shuffle_buffer)
    seed = int(args.seed)
    repeat_mode = str(args.repeat_dataset or "auto").strip().lower()

    model = InferenceMLP(INF_FEAT_DIM, hidden=hidden, seed=seed)

    # Decide whether to repeat dataset
    infer_samples = _infer_samples_from_manifest(mp)
    steps_per_pass = int(max(1, infer_samples // max(1, batch)))
    if repeat_mode == "true":
        repeat_dataset = True
    elif repeat_mode == "false":
        repeat_dataset = False
    else:
        repeat_dataset = (steps > steps_per_pass)

    # validation set (reservoir sample to avoid "first records" bias)
    val_n = int(args.val_samples)
    val = _reservoir_sample_val(mp, n=val_n, seed=seed + 777)

    rng = random.Random(seed + 99991)
    losses: List[float] = []

    def next_record(stream_it: Iterator[Tuple[np.ndarray, np.ndarray, int, int]]) -> Tuple[np.ndarray, np.ndarray, int, int]:
        return next(stream_it)

    stream: Iterator[Tuple[np.ndarray, np.ndarray, int, int]] = _iter_records_from_manifest(mp, sample_cap=None)
    buf: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

    trained_steps = 0
    while trained_steps < steps:
        # Fill batch with a shuffle buffer (lightweight)
        while len(buf) < batch:
            try:
                feat, label_i8, opp_cnt, _bone_cnt = next_record(stream)
            except StopIteration:
                if repeat_dataset:
                    stream = _iter_records_from_manifest(mp, sample_cap=None)
                    continue
                else:
                    break

            y = label_i8.astype(np.float32)
            visible = ((feat[0:28] + feat[28:56]) > 0.5).astype(bool)
            item = (feat, y, visible, int(opp_cnt))

            if shuffle_buffer <= 0:
                buf.append(item)
                continue
            if len(buf) < shuffle_buffer:
                buf.append(item)
                continue
            j = rng.randrange(0, len(buf))
            buf[j], item = item, buf[j]
            buf.append(item)

        if len(buf) < max(8, batch // 8):
            break

        take = buf[:batch]
        buf = buf[batch:]

        X = np.stack([t[0] for t in take], axis=0).astype(np.float32)
        Y = np.stack([t[1] for t in take], axis=0).astype(np.float32)
        V = np.stack([t[2] for t in take], axis=0).astype(bool)

        lossd = model.train_step(X, Y, V, lr=lr, l2=l2, pos_w=pos_w)
        losses.append(float(lossd["loss"]))
        trained_steps += 1
        if trained_steps % 50 == 0:
            print(json.dumps({"step": trained_steps, "loss": round(float(np.mean(losses[-50:])), 5)}, ensure_ascii=False))

    # validation metrics
    accs: List[float] = []
    bases: List[float] = []
    for feat, label_i8, opp_cnt, _bone_cnt in val:
        y = label_i8.astype(np.float32)
        visible = ((feat[0:28] + feat[28:56]) > 0.5)
        _z1, _h1, logits = model.forward(feat.reshape(1, -1).astype(np.float32))
        probs = (1.0 / (1.0 + np.exp(-logits.reshape(-1)))).astype(np.float32)
        accs.append(_topk_accuracy(probs, y, opp_cnt=opp_cnt, visible=visible))
        bases.append(_baseline_topk_expected(opp_cnt=opp_cnt, visible=visible))

    model.save(out_path)
    topk = float(np.mean(accs or [0.0]))
    base = float(np.mean(bases or [0.0]))
    print(json.dumps({
        "ok": True,
        "trained_steps": int(trained_steps),
        "final_loss": round(float(np.mean(losses[-min(50, len(losses)):] or [0.0])), 6),
        "val_topk_acc": round(topk, 4),
        "val_topk_baseline": round(base, 4),
        "val_topk_delta": round(topk - base, 4),
        "repeat_dataset": bool(repeat_dataset),
        "steps_per_pass_est": int(steps_per_pass),
        "model_out": str(out_path),
    }, ensure_ascii=False))
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    # check then train with the same args
    rc = cmd_check(args)
    if rc != 0:
        return rc
    return cmd_train(args)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="infer_tool.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_chk = sub.add_parser("check", help="validate INF1 shards")
    p_chk.add_argument("--manifest", type=str, required=True)
    p_chk.add_argument("--sample_limit", type=int, default=500)
    p_chk.set_defaults(func=cmd_check)

    p_tr = sub.add_parser("train", help="train inference MLP from INF1")
    p_tr.add_argument("--manifest", type=str, required=True)
    p_tr.add_argument("--hidden", type=int, default=256)
    p_tr.add_argument("--steps", type=int, default=400)
    p_tr.add_argument("--batch", type=int, default=256)
    p_tr.add_argument("--lr", type=float, default=1e-3)
    p_tr.add_argument("--l2", type=float, default=1e-4)
    p_tr.add_argument("--pos_w", type=float, default=6.0)
    p_tr.add_argument("--repeat_dataset", type=str, default="auto", choices=["auto", "true", "false"])
    p_tr.add_argument("--shuffle_buffer", type=int, default=4096)
    p_tr.add_argument("--seed", type=int, default=12345)
    p_tr.add_argument("--val_samples", type=int, default=2000)
    p_tr.add_argument("--out", type=str, default="inference_model.json")
    p_tr.set_defaults(func=cmd_train)

    p_pi = sub.add_parser("pilot", help="check + train (single command)")
    for a in ("manifest", "hidden", "steps", "batch", "lr", "l2", "pos_w", "shuffle_buffer", "seed", "val_samples", "out", "sample_limit"):
        # reuse args by adding manually
        pass
    p_pi.add_argument("--manifest", type=str, required=True)
    p_pi.add_argument("--sample_limit", type=int, default=500)
    p_pi.add_argument("--hidden", type=int, default=256)
    p_pi.add_argument("--steps", type=int, default=400)
    p_pi.add_argument("--batch", type=int, default=256)
    p_pi.add_argument("--lr", type=float, default=1e-3)
    p_pi.add_argument("--l2", type=float, default=1e-4)
    p_pi.add_argument("--pos_w", type=float, default=6.0)
    p_pi.add_argument("--repeat_dataset", type=str, default="auto", choices=["auto", "true", "false"])
    p_pi.add_argument("--shuffle_buffer", type=int, default=4096)
    p_pi.add_argument("--seed", type=int, default=12345)
    p_pi.add_argument("--val_samples", type=int, default=2000)
    p_pi.add_argument("--out", type=str, default="inference_model.json")
    p_pi.set_defaults(func=cmd_pilot)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())