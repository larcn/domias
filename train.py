# FILE: train.py | version: 2026-01-14.p4a2 (+Spike head)
# Clean AlphaZero-style rebuild (within original project structure):
#
# Data semantics:
# - PI: ISMCTS root visit distribution (masked), with apply_guard=False (pure).
# - Z : final MATCH outcome only (+1/-1), no tanh(score), no sigmoid(diff).
#
# Generation:
# - Self-play is symmetric using two perspective GameStates:
#     stA: player A is "me"
#     stB: player B is "me"
#   Truth: we keep actual hands/boneyard and apply actions to both states.
#
# Training:
# - Same base MLP format as before: model.json type=mlp_pv_v1
# - FEAT_DIM=193, ACTION_SIZE=112 must match ai.py
#
# P2 (2026-xx-xx): Quantile head (Python-only)
# -------------------------------------------
# - Extend MLP with an auxiliary quantile head (Wq, bq) of size K=11.
# - forward(X, MASK) now returns: policy, value, quantiles, cache.
# - train_batch(...) optionally takes Q (target quantiles) and applies
#   pinball loss on quantiles in addition to policy/value losses.
# - JSON model:
#     * Still "type": "mlp_pv_v1"
#     * Adds optional keys "Wq" and "bq" (ignored by old Rust loader).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set, Literal
import json
import math
import os
import random
import tempfile
import time

import numpy as np

import storage
import ai
from engine import (
    GameState, Board, Tile, EndName, ALL_TILES,
    tile_is_double, tile_pip_count
)

TeacherKind = Literal["2ply", "deep", "ismcts"]  # kept for backward-compat only (ignored)

ENDS: List[EndName] = ["right", "left", "up", "down"]
TILES_SORTED: List[Tile] = sorted(ALL_TILES, key=lambda t: (t[0], t[1]))
TILE_TO_IDX: Dict[Tile, int] = {t: i for i, t in enumerate(TILES_SORTED)}

ACTION_SIZE = 112
FEAT_DIM = 193
QUANTILES_K = 11
TAUS = np.linspace(0.05, 0.95, QUANTILES_K, dtype=np.float32)


# -----------------------------
# Action encoding / masks
# -----------------------------

def encode_action(tile: Tile, end: EndName) -> int:
    return TILE_TO_IDX[tile] * 4 + ENDS.index(end)


def legal_mask_from_state(state: GameState) -> np.ndarray:
    mask = np.zeros((ACTION_SIZE,), dtype=np.int8)
    for (tile, end) in state.legal_moves_me():
        mask[encode_action(tile, end)] = 1
    return mask


# -----------------------------
# Features (must match ai.features_small)
# -----------------------------

def features_small(state: GameState) -> np.ndarray:
    return ai.features_small(state)


# -----------------------------
# DB schema guard
# -----------------------------

def count_samples(db_path: os.PathLike | None = None) -> int:
    return storage.count_samples_pv(db_path=db_path or storage.DB_PATH)


def _detect_existing_feat_dim(db_path: os.PathLike | None = None) -> Optional[int]:
    try:
        if count_samples(db_path) <= 0:
            return None
        batch = storage.sample_batch_pv(1, db_path=db_path or storage.DB_PATH)
        if not batch:
            return None
        feat_blob = batch[0][0]
        if not isinstance(feat_blob, (bytes, bytearray)):
            return None
        return int(len(feat_blob) // 4)
    except Exception:
        return None


def _require_clean_db_for_feat_dim(expected_dim: int, db_path: os.PathLike | None = None) -> None:
    got = _detect_existing_feat_dim(db_path)
    if got is None:
        return
    if int(got) != int(expected_dim):
        raise RuntimeError(
            f"train.db already contains features with dim={got}, but current FEAT_DIM={expected_dim}. "
            f"Delete train.db (or use --clean) to avoid mixed feature schemas."
        )


# -----------------------------
# MLP (policy + scalar value + quantiles head)

@dataclass
class MLP:
    W1: np.ndarray   # [hidden, FEAT_DIM]
    b1: np.ndarray   # [hidden]
    Wp: np.ndarray   # [ACTION_SIZE, hidden]
    bp: np.ndarray   # [ACTION_SIZE]
    Wv: np.ndarray   # [1, hidden]
    bv: np.ndarray   # [1]
    Wq: np.ndarray   # [QUANTILES_K, hidden]
    bq: np.ndarray   # [QUANTILES_K]
    # NEW: spike head (per action)
    Ws: np.ndarray   # [ACTION_SIZE, hidden]
    bs: np.ndarray   # [ACTION_SIZE]
    adam: Dict[str, Tuple[np.ndarray, np.ndarray]]
    t: int

    @classmethod
    def init(cls, hidden_size: int = 384, seed: Optional[int] = None) -> "MLP":
        if seed is not None:
            np.random.seed(int(seed))

        def xavier(shape: Tuple[int, int]) -> np.ndarray:
            fan_in, fan_out = shape[1], shape[0]
            limit = math.sqrt(6.0 / float(fan_in + fan_out))
            return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

        W1 = xavier((hidden_size, FEAT_DIM))
        b1 = np.zeros((hidden_size,), np.float32)

        Wp = xavier((ACTION_SIZE, hidden_size))
        bp = np.zeros((ACTION_SIZE,), np.float32)

        Wv = xavier((1, hidden_size))
        bv = np.zeros((1,), np.float32)

        # NEW: quantile head
        Wq = xavier((QUANTILES_K, hidden_size))
        bq = np.zeros((QUANTILES_K,), np.float32)

        # NEW: spike head
        Ws = xavier((ACTION_SIZE, hidden_size))
        bs = np.zeros((ACTION_SIZE,), np.float32)

        return cls(W1, b1, Wp, bp, Wv, bv, Wq, bq, Ws, bs, adam={}, t=0)

    def save(self, path: str = "model.json") -> None:
        model_data: Dict[str, Any] = {
            "type": "mlp_pv_v1",
            "feat_dim": int(FEAT_DIM),
            "action_size": int(ACTION_SIZE),
            "hidden": int(self.W1.shape[0]),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "Wp": self.Wp.tolist(),
            "bp": self.bp.tolist(),
            "Wv": self.Wv.tolist(),
            "bv": self.bv.tolist(),
            # NEW: optional quantile head (ignored by old loaders)
            "Wq": self.Wq.tolist(),
            "bq": self.bq.tolist(),
            # NEW: spike head
            "Ws": self.Ws.tolist(),
            "bs": self.bs.tolist(),
        }

        dir_path = os.path.dirname(path) or "."
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=dir_path, suffix=".tmp", delete=False, encoding="utf-8") as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
                tmp_path = f.name
            os.replace(tmp_path, path)
            tmp_path = None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @classmethod
    def load(cls, path: str = "model.json") -> "MLP":
        with open(path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        if model_data.get("type") != "mlp_pv_v1":
            raise ValueError("Incompatible model type")
        if int(model_data.get("action_size", -1)) != ACTION_SIZE:
            raise ValueError("Model action_size mismatch")
        if int(model_data.get("feat_dim", -1)) != FEAT_DIM:
            raise ValueError(
                f"Model feat_dim mismatch: model has {model_data.get('feat_dim')} but code expects {FEAT_DIM}. "
                f"Delete old model.json and retrain with clean DB."
            )

        hidden = int(model_data["hidden"])

        W1 = np.array(model_data["W1"], np.float32)
        b1 = np.array(model_data["b1"], np.float32)
        Wp = np.array(model_data["Wp"], np.float32)
        bp = np.array(model_data["bp"], np.float32)
        Wv = np.array(model_data["Wv"], np.float32)
        bv = np.array(model_data["bv"], np.float32)

        # NEW: load optional quantile head if present; else init zeros
        if "Wq" in model_data and "bq" in model_data:
            Wq = np.array(model_data["Wq"], np.float32)
            bq = np.array(model_data["bq"], np.float32)
        else:
            Wq = np.zeros((QUANTILES_K, hidden), np.float32)
            bq = np.zeros((QUANTILES_K,), np.float32)

        # NEW: load optional spike head if present; else init zeros
        if "Ws" in model_data and "bs" in model_data:
            Ws = np.array(model_data["Ws"], np.float32)
            bs = np.array(model_data["bs"], np.float32)
        else:
            Ws = np.zeros((ACTION_SIZE, hidden), np.float32)
            bs = np.zeros((ACTION_SIZE,), np.float32)

        return cls(
            W1=W1,
            b1=b1,
            Wp=Wp,
            bp=bp,
            Wv=Wv,
            bv=bv,
            Wq=Wq,
            bq=bq,
            Ws=Ws,
            bs=bs,
            adam={},
            t=0
        )

    def forward(
        self,
        X: np.ndarray,
        MASK: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns:
          policy   : [B, ACTION_SIZE]
          value    : [B]
          quantiles: [B, QUANTILES_K]
          spike    : [B, ACTION_SIZE] in [0,1]
        """
        # hidden
        z1 = X @ self.W1.T + self.b1
        h1 = np.maximum(z1, 0.0)

        # policy logits
        logits = h1 @ self.Wp.T + self.bp
        logits = logits.copy()
        logits[MASK == 0] = -1e9

        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        policy = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12)

        # scalar value head
        value = np.tanh((h1 @ self.Wv.T + self.bv)).reshape(-1)

        # NEW: quantile head (no tanh; raw forecast)
        quantiles = (h1 @ self.Wq.T + self.bq).astype(np.float32)  # [B, K]

        # NEW: spike head (sigmoid)
        spike_logits = (h1 @ self.Ws.T + self.bs).astype(np.float32)  # [B, A]
        spike = (1.0 / (1.0 + np.exp(-spike_logits))).astype(np.float32)

        cache = {
            "X": X,
            "z1": z1,
            "h1": h1,
            "policy": policy,
            "value": value,
            "quantiles": quantiles,
            "spike": spike,
            "spike_logits": spike_logits,
        }
        return policy.astype(np.float32), value.astype(np.float32), quantiles, spike, cache

    def _adam_update(
        self,
        name: str,
        param: np.ndarray,
        grad: np.ndarray,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> np.ndarray:
        m1, m2 = self.adam.get(name, (np.zeros_like(param), np.zeros_like(param)))
        m1 = beta1 * m1 + (1 - beta1) * grad
        m2 = beta2 * m2 + (1 - beta2) * (grad * grad)
        self.adam[name] = (m1, m2)

        t = max(1, int(self.t))
        m1h = m1 / (1 - beta1 ** t)
        m2h = m2 / (1 - beta2 ** t)
        return param - learning_rate * m1h / (np.sqrt(m2h) + epsilon)

    def train_batch(
        self,
        X: np.ndarray,
        PI: np.ndarray,
        Z: np.ndarray,
        MASK: np.ndarray,
        Q: Optional[np.ndarray] = None,
        SPIKE: Optional[np.ndarray] = None,
        train_mode: str = "both",          # NEW: both|pv_only|spike_only
        spike_only: bool = False,          # legacy alias; if True forces spike_only mode
        spike_pos_w: float = 20.0,
        learning_rate: float = 1e-3,
        l2_lambda: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Train on a batch:
          X   : [B, FEAT_DIM]
          PI  : [B, ACTION_SIZE]
          Z   : [B]
          MASK: [B, ACTION_SIZE]
          Q   : [B, QUANTILES_K] or None
          SPIKE: [B, ACTION_SIZE] or None
        Returns:
          dict with policy_loss, value_loss, quantile_loss
        """
        self.t += 1
        bs = X.shape[0]

        # determine train_mode
        mode = str(train_mode or "both").strip().lower()
        if spike_only:
            mode = "spike_only"
        if mode not in ("both", "pv_only", "spike_only"):
            raise ValueError(f"bad train_mode={train_mode!r} (expected both|pv_only|spike_only)")

        P, V, Q_hat, S, cache = self.forward(X, MASK)

        # Policy loss: cross-entropy with PI
        policy_loss = float(np.mean(-np.sum(PI * np.log(P + 1e-12), axis=1)))
        # Value loss: MSE
        value_loss = float(np.mean((V - Z) ** 2))

        # d(logits) for policy
        dlogits = (P - PI) / float(bs)
        dlogits[MASK == 0] = 0.0

        h1 = cache["h1"]
        z1 = cache["z1"]

        # Gradients for policy head
        gWp = dlogits.T @ h1
        gbp = np.sum(dlogits, axis=0)

        # Scalar value head
        dV = (2.0 * (V - Z) / float(bs))
        dzv = (dV * (1.0 - V * V)).reshape(bs, 1)
        gWv = dzv.T @ h1
        gbv = np.sum(dzv, axis=0)

        # Quantile head (pinball loss)
        if Q is not None:
            assert Q.shape == (bs, QUANTILES_K), f"Q shape mismatch {Q.shape} vs {(bs, QUANTILES_K)}"
            U = Q - Q_hat  # [B,K], target - pred
            taus = TAUS.reshape(1, QUANTILES_K)  # [1,K]
            # derivative wrt Q_hat: dL/dQ_hat = - (dL/dU)
            grad_q = np.where(
                U >= 0.0,
                -taus,          # u>=0 => ∂ρ/∂u = τ
                -(taus - 1.0),  # u<0  => ∂ρ/∂u = τ-1
            ) / float(bs)
            # pinball loss value (for logging)
            loss_q_sample = np.maximum(taus * U, (taus - 1.0) * U)
            quantile_loss = float(np.mean(loss_q_sample))
        else:
            grad_q = np.zeros_like(Q_hat)
            quantile_loss = 0.0

        # Gradients for Wq, bq
        gWq = grad_q.T @ h1           # [K,hidden]
        gbq = np.sum(grad_q, axis=0)  # [K]

        # Spike head (Weighted BCE on legal actions only)
        # pv_only => ignore spike completely (no loss, no grads, no update)
        if mode == "pv_only":
            spike_loss = 0.0
            dspike_logits = np.zeros_like(S)
        elif SPIKE is not None:
            assert SPIKE.shape == (bs, ACTION_SIZE), f"SPIKE shape mismatch {SPIKE.shape} vs {(bs, ACTION_SIZE)}"
            legal = (MASK == 1)
            denom_sp = float(max(1, int(np.sum(legal))))

            POS_W = float(spike_pos_w)
            eps = 1e-6
            bce = -(POS_W * SPIKE * np.log(S + eps) + (1.0 - SPIKE) * np.log(1.0 - S + eps))
            spike_loss = float(np.sum(bce[legal]) / denom_sp)

            dspike_logits = np.where(SPIKE >= 0.5, POS_W * (S - 1.0), S) / denom_sp
            dspike_logits[~legal] = 0.0
        else:
            spike_loss = 0.0
            dspike_logits = np.zeros_like(S)

        gWs = dspike_logits.T @ h1               # [A, hidden]
        gbs = np.sum(dspike_logits, axis=0)      # [A]

        # Backprop through hidden
        dh1 = dlogits @ self.Wp + dzv @ self.Wv + grad_q @ self.Wq + dspike_logits @ self.Ws
        dz1 = dh1 * (z1 > 0)
        gW1 = dz1.T @ cache["X"]
        gb1 = np.sum(dz1, axis=0)

        # L2 regularization + Adam updates based on mode
        if mode == "spike_only":
            # Freeze trunk + policy/value/quantiles heads. Only update spike head.
            gWs += l2_lambda * self.Ws

            self.Ws = self._adam_update("Ws", self.Ws, gWs.astype(np.float32), learning_rate)
            self.bs = self._adam_update("bs", self.bs, gbs.astype(np.float32), learning_rate)

        elif mode == "pv_only":
            # Train PV(+quantiles) only. Freeze spike head entirely.
            gW1 += l2_lambda * self.W1
            gWp += l2_lambda * self.Wp
            gWv += l2_lambda * self.Wv
            gWq += l2_lambda * self.Wq

            self.W1 = self._adam_update("W1", self.W1, gW1.astype(np.float32), learning_rate)
            self.b1 = self._adam_update("b1", self.b1, gb1.astype(np.float32), learning_rate)
            self.Wp = self._adam_update("Wp", self.Wp, gWp.astype(np.float32), learning_rate)
            self.bp = self._adam_update("bp", self.bp, gbp.astype(np.float32), learning_rate)
            self.Wv = self._adam_update("Wv", self.Wv, gWv.astype(np.float32), learning_rate)
            self.bv = self._adam_update("bv", self.bv, gbv.astype(np.float32), learning_rate)
            self.Wq = self._adam_update("Wq", self.Wq, gWq.astype(np.float32), learning_rate)
            self.bq = self._adam_update("bq", self.bq, gbq.astype(np.float32), learning_rate)

        else:
            # both: update everything
            gW1 += l2_lambda * self.W1
            gWp += l2_lambda * self.Wp
            gWv += l2_lambda * self.Wv
            gWq += l2_lambda * self.Wq
            gWs += l2_lambda * self.Ws

            self.W1 = self._adam_update("W1", self.W1, gW1.astype(np.float32), learning_rate)
            self.b1 = self._adam_update("b1", self.b1, gb1.astype(np.float32), learning_rate)
            self.Wp = self._adam_update("Wp", self.Wp, gWp.astype(np.float32), learning_rate)
            self.bp = self._adam_update("bp", self.bp, gbp.astype(np.float32), learning_rate)
            self.Wv = self._adam_update("Wv", self.Wv, gWv.astype(np.float32), learning_rate)
            self.bv = self._adam_update("bv", self.bv, gbv.astype(np.float32), learning_rate)
            self.Wq = self._adam_update("Wq", self.Wq, gWq.astype(np.float32), learning_rate)
            self.bq = self._adam_update("bq", self.bq, gbq.astype(np.float32), learning_rate)
            self.Ws = self._adam_update("Ws", self.Ws, gWs.astype(np.float32), learning_rate)
            self.bs = self._adam_update("bs", self.bs, gbs.astype(np.float32), learning_rate)

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "quantile_loss": quantile_loss,
            "spike_loss": spike_loss,
        }


# -----------------------------
# PI sampling from ISMCTS visits
# -----------------------------

def _pi_from_ismcts_visits(
    state: GameState,
    det: int,
    think_ms: int,
    temperature: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: (PI, MASK)
    - PI is derived from ISMCTS root visit counts, masked and normalized.
    - apply_guard=False (training purity).
    """
    st = GameState.from_dict(state.to_dict())
    st.current_turn = "me"
    belief = ai.build_belief_from_state(st)

    mask = legal_mask_from_state(st)
    if int(mask.sum()) == 0:
        return np.zeros((ACTION_SIZE,), np.float32), mask

    base_seed = int(rng.randrange(1, 2**31 - 1))

    # Prefer calling internal ISMCTS (fast + pure visits), fallback to suggest_moves if needed.
    try:
        sugs, _meta = ai._ismcts_root_search(  # type: ignore[attr-defined]
            state=st,
            belief=belief,
            base_seed=base_seed,
            det=int(det),
            think_ms=int(think_ms),
            max_depth=6,
            c_puct=1.6,
            redet_opp=True,
            apply_guard=False,
        )
    except Exception as e:
        raise RuntimeError(
            "ISMCTS entrypoint not available or incompatible. "
            "Ensure ai.py is updated (supports _ismcts_root_search(..., apply_guard=False))."
        ) from e

    visits = np.zeros((ACTION_SIZE,), dtype=np.float64)
    for s in sugs:
        try:
            t = ai.parse_tile(s["tile"])
            e = s["end"]
        except Exception:
            continue
        if e not in ENDS:
            continue
        aidx = encode_action(t, e)
        visits[aidx] = max(0.0, float(s.get("visits", 0.0)))

    visits[mask == 0] = 0.0
    s_vis = float(visits.sum())
    if s_vis <= 1e-12:
        PI = mask.astype(np.float32) / float(max(1, int(mask.sum())))
        return PI, mask

    temp = max(1e-6, float(temperature))
    if abs(temp - 1.0) > 1e-6:
        visits = np.power(visits, 1.0 / temp)
        visits[mask == 0] = 0.0
        s_vis = float(visits.sum())

    PI = (visits / (s_vis + 1e-12)).astype(np.float32)
    return PI, mask


def pick_move_from_pi(state: GameState, PI: np.ndarray, rng: random.Random, p_argmax: float = 0.05) -> Tuple[Tile, EndName]:
    legal = state.legal_moves_me()
    if not legal:
        raise RuntimeError("No legal moves")

    acts = [encode_action(t, e) for (t, e) in legal]
    probs = np.array([float(PI[a]) for a in acts], dtype=np.float64)
    s = float(probs.sum())
    if s <= 1e-12:
        return legal[int(rng.randrange(0, len(legal)))]

    probs /= s
    if rng.random() < float(p_argmax):
        idx = int(np.argmax(probs))
        return legal[idx]

    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += float(p)
        if acc >= r:
            return legal[i]
    return legal[-1]


# -----------------------------
# Self-play helpers (Python path, not Rust)

def _best_opening_tile(hand: Set[Tile]) -> Tile:
    doubles = [t for t in hand if tile_is_double(t)]
    if doubles:
        return max(doubles, key=lambda t: t[0])
    return max(hand, key=lambda t: (t[0] + t[1], t[0], t[1]))


def _key_open(t: Tile) -> Tuple[int, int, int]:
    return (1 if tile_is_double(t) else 0, t[0] + t[1], t[0])


def _deal(rng: random.Random) -> Tuple[List[Tile], List[Tile], List[Tile]]:
    tiles = list(ALL_TILES)
    rng.shuffle(tiles)
    return tiles[:7], tiles[7:14], tiles[14:]


def _sum_pips(hand: Set[Tile]) -> int:
    return int(sum(tile_pip_count(t) for t in hand))


def _sync_counts(stA: GameState, stB: GameState, handA: Set[Tile], handB: Set[Tile], boneyard: List[Tile]) -> None:
    stA.opponent_tile_count = int(len(handB))
    stA.boneyard_count = int(len(boneyard))
    stB.opponent_tile_count = int(len(handA))
    stB.boneyard_count = int(len(boneyard))

    total = int(len(handA) + len(handB) + len(boneyard) + len(stA.board.played_set))
    if total != 28:
        raise RuntimeError(f"[selfplay] invariant broken: {total} != 28")


def _finalize_out_me_pending(st: GameState, opp_hand_truth: Set[Tile]) -> None:
    if st.round_over and st.round_end_reason == "out_me_pending":
        st.finalize_out_with_opponent_pips(_sum_pips(opp_hand_truth))


def _hand_has_any_move(board: Board, hand: Set[Tile]) -> bool:
    for t in hand:
        if board.legal_ends_for_tile(t):
            return True
    return False


def _maybe_declare_locked_both(stA: GameState, stB: GameState, handA: Set[Tile], handB: Set[Tile], boneyard: List[Tile]) -> bool:
    if boneyard:
        return False

    board = stA.board
    if _hand_has_any_move(board, handA) or _hand_has_any_move(board, handB):
        return False

    # Evidence + declare on both perspectives (keeps scores aligned)
    stA.current_turn = "opponent"
    stA.record_pass("opponent", certainty="certain")  # flips to me
    stA.declare_locked(opponent_pips=_sum_pips(handB))

    stB.current_turn = "opponent"
    stB.record_pass("opponent", certainty="certain")  # flips to me
    stB.declare_locked(opponent_pips=_sum_pips(handA))

    return True


def _apply_forced_best_opening(stA: GameState, stB: GameState, handA: Set[Tile], handB: Set[Tile]) -> None:
    """
    If opening_mode is forced_best and board is empty:
    choose opener by comparing best opening tile in each hand, then play it immediately.
    This enforces engine opening rule and avoids illegal first moves.
    """
    if not stA.board.is_empty():
        return
    if getattr(stA, "opening_mode", "forced_best") != "forced_best":
        return

    a_req = _best_opening_tile(set(handA))
    b_req = _best_opening_tile(set(handB))

    # opener is the side with stronger opening key
    if _key_open(b_req) > _key_open(a_req):
        # B opens
        stB.current_turn = "me"
        stA.current_turn = "opponent"
        if b_req not in handB:
            raise RuntimeError("truth mismatch: B opener tile not in handB")
        handB.remove(b_req)
        stB.play_tile("me", b_req, "right")
        stA.play_tile("opponent", b_req, "right")
    else:
        # A opens
        stA.current_turn = "me"
        stB.current_turn = "opponent"
        if a_req not in handA:
            raise RuntimeError("truth mismatch: A opener tile not in handA")
        handA.remove(a_req)
        stA.play_tile("me", a_req, "right")
        stB.play_tile("opponent", a_req, "right")


# -----------------------------
# Public: selfplay_one_match_rows (used by MP)

def selfplay_one_match_rows(
    rng: random.Random,
    determinizations: int,
    think_ms: int,
    temperature: float,
    max_moves_per_round: int,
    match_target: int = 150,
    max_rounds: int = 200,
) -> List[Tuple[bytes, bytes, float, bytes]]:
    """
    One match -> rows (feat, pi, z, mask) for BOTH players' perspectives.
    Z is always +1/-1 (no zeros).
    """
    # Round 1 deal
    handA_list, handB_list, boneyard = _deal(rng)
    handA: Set[Tile] = set(handA_list)
    handB: Set[Tile] = set(handB_list)

    stA = GameState(); stB = GameState()
    stA.start_new_game(handA_list, match_target=int(match_target))
    stB.start_new_game(handB_list, match_target=int(match_target))

    pendingA: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    pendingB: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # Apply forced opening rule if needed
    _apply_forced_best_opening(stA, stB, handA, handB)
    _sync_counts(stA, stB, handA, handB, boneyard)

    rounds = 1
    while (int(stA.my_score) < int(stA.match_target)) and (int(stA.opponent_score) < int(stA.match_target)) and (rounds <= int(max_rounds)):
        # one round loop
        for _ply in range(int(max_moves_per_round)):
            _sync_counts(stA, stB, handA, handB, boneyard)

            if stA.round_over or stB.round_over:
                break

            if _maybe_declare_locked_both(stA, stB, handA, handB, boneyard):
                break

            if stA.round_over or stB.round_over:
                break

            # ---------- A to act ----------
            if stA.current_turn == "me":
                # draw-until-play A (A knows tile; B only sees count)
                while stA.must_draw_me() and boneyard:
                    drawn = boneyard.pop()
                    handA.add(drawn)
                    stA.record_draw_me(drawn)

                    # B observes opponent draw (unknown tile)
                    # record_draw requires opponent turn on stB
                    stB.current_turn = "opponent"
                    stB.record_draw("opponent", count=1, certainty="certain")

                _sync_counts(stA, stB, handA, handB, boneyard)

                if stA.round_over:
                    _finalize_out_me_pending(stA, handB)
                    break

                if stA.must_pass_me():
                    stA.record_pass("me", certainty="certain")
                    stB.current_turn = "opponent"
                    stB.record_pass("opponent", certainty="certain")
                    continue

                PI, MASK = _pi_from_ismcts_visits(stA, int(determinizations), int(think_ms), float(temperature), rng)
                pendingA.append((features_small(stA), PI, MASK))

                t, e = pick_move_from_pi(stA, PI, rng, p_argmax=0.05)
                if t not in handA:
                    raise RuntimeError("truth mismatch: A played tile not in handA")
                handA.remove(t)

                stA.play_tile("me", t, e)
                stB.play_tile("opponent", t, e)

                _finalize_out_me_pending(stA, handB)
                _finalize_out_me_pending(stB, handA)
                continue

            # ---------- B to act ----------
            stB.current_turn = "me"

            while stB.must_draw_me() and boneyard:
                drawn = boneyard.pop()
                handB.add(drawn)
                stB.record_draw_me(drawn)

                stA.current_turn = "opponent"
                stA.record_draw("opponent", count=1, certainty="certain")

            _sync_counts(stA, stB, handA, handB, boneyard)

            if stB.round_over:
                _finalize_out_me_pending(stB, handA)
                break

            if stB.must_pass_me():
                stB.record_pass("me", certainty="certain")
                stA.current_turn = "opponent"
                stA.record_pass("opponent", certainty="certain")
                continue

            PI, MASK = _pi_from_ismcts_visits(stB, int(determinizations), int(think_ms), float(temperature), rng)
            pendingB.append((features_small(stB), PI, MASK))

            t, e = pick_move_from_pi(stB, PI, rng, p_argmax=0.05)
            if t not in handB:
                raise RuntimeError("truth mismatch: B played tile not in handB")
            handB.remove(t)

            stB.play_tile("me", t, e)
            stA.play_tile("opponent", t, e)

            _finalize_out_me_pending(stA, handB)
            _finalize_out_me_pending(stB, handA)

        # end-of-round finalization
        _finalize_out_me_pending(stA, handB)
        _finalize_out_me_pending(stB, handA)

        if (int(stA.my_score) >= int(stA.match_target)) or (int(stA.opponent_score) >= int(stA.match_target)):
            break

        if rounds >= int(max_rounds):
            break

        if not stA.round_over:
            # safety
            break

        prev_reason_A = stA.round_end_reason
        rounds += 1

        # new deal
        handA_list, handB_list, boneyard = _deal(rng)
        handA = set(handA_list)
        handB = set(handB_list)

        stA.start_new_round(handA_list)
        stB.start_new_round(handB_list)
        _sync_counts(stA, stB, handA, handB, boneyard)

        # opener:
        if stA.opening_mode == "free":
            if prev_reason_A in ("out_me", "target_me"):
                stA.current_turn = "me"; stB.current_turn = "opponent"
            elif prev_reason_A in ("out_opponent", "target_opponent"):
                stA.current_turn = "opponent"; stB.current_turn = "me"
            else:
                stA.current_turn = "me"; stB.current_turn = "opponent"
        else:
            # forced_best mode: apply forced opening rule
            _apply_forced_best_opening(stA, stB, handA, handB)
            _sync_counts(stA, stB, handA, handB, boneyard)

    # Determine match winner from A perspective
    target = int(stA.match_target)
    my = int(stA.my_score)
    opp = int(stA.opponent_score)
    reason = str(stA.round_end_reason or "")

    if reason == "target_me":
        zA = 1.0
    elif reason == "target_opponent":
        zA = -1.0
    else:
        a_win = (my >= target) and (opp < target)
        b_win = (opp >= target) and (my < target)
        if a_win and not b_win:
            zA = 1.0
        elif b_win and not a_win:
            zA = -1.0
        else:
            # deterministic fallback (avoid 0)
            zA = 1.0 if my >= opp else -1.0

    zB = -zA

    rows: List[Tuple[bytes, bytes, float, bytes]] = []
    for feat, PI, MASK in pendingA:
        rows.append((feat.astype(np.float32).tobytes(), PI.astype(np.float32).tobytes(), float(zA), MASK.astype(np.int8).tobytes()))
    for feat, PI, MASK in pendingB:
        rows.append((feat.astype(np.float32).tobytes(), PI.astype(np.float32).tobytes(), float(zB), MASK.astype(np.int8).tobytes()))
    return rows


# -----------------------------
# Public: generate_samples (single-process)

def generate_samples(
    games: int = 500,
    determinizations: int = 12,
    temperature: float = 0.85,
    max_moves: int = 500,
    seed: Optional[int] = None,
    teacher: TeacherKind = "ismcts",  # ignored
    think_ms: int = 1200,
) -> int:
    _require_clean_db_for_feat_dim(FEAT_DIM, db_path=storage.DB_PATH)

    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    rng = random.Random(int(seed))

    buffer_rows: List[Tuple[bytes, bytes, float, bytes]] = []
    total_added = 0
    FLUSH_EVERY = 4000

    def flush() -> None:
        nonlocal total_added, buffer_rows
        if not buffer_rows:
            return
        total_added += storage.add_samples_pv(buffer_rows)
        buffer_rows = []
    for _ in range(int(games)):
        buffer_rows.extend(selfplay_one_match_rows(
            rng=rng,
            determinizations=int(determinizations),
            think_ms=int(think_ms),
            temperature=float(temperature),
            max_moves_per_round=int(max_moves),
            match_target=150,
            max_rounds=200,
        ))
        if len(buffer_rows) >= FLUSH_EVERY:
            flush()

    flush()
    return int(total_added)


# -----------------------------
# Train model from DB (train.db)

def train_model(
    epochs: int = 3,
    batch_size: int = 1024,
    learning_rate: float = 0.0010,
    l2_lambda: float = 1e-4,
    hidden_size: int = 384,
    seed: Optional[int] = None,
    model_path: str = "model.json"
) -> Dict[str, Any]:
    try:
        model = MLP.load(model_path)
        init_mode = "loaded"
    except Exception:
        model = MLP.init(hidden_size=int(hidden_size), seed=seed)
        init_mode = "fresh"

    sample_count = storage.count_samples_pv()
    if sample_count < 800:
        return {"ok": False, "error": f"عينات غير كافية ({sample_count}) - يحتاج 800 على الأقل", "init_mode": init_mode}

    history: List[Dict[str, float]] = []
    steps_per_epoch = max(1, int(sample_count // int(batch_size)))

    for epoch in range(int(epochs)):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_quantile_loss = 0.0
        effective_steps = 0

        for _step in range(steps_per_epoch):
            batch = storage.sample_batch_pv(int(batch_size))
            if not batch:
                continue

            X_bytes = b"".join([feat for (feat, _, _, _) in batch])
            PI_bytes = b"".join([pi for (_, pi, _, _) in batch])
            Z_values = [z for (_, _, z, _) in batch]
            M_bytes = b"".join([mask for (_, _, _, mask) in batch])

            if len(X_bytes) % (4 * FEAT_DIM) != 0:
                return {
                    "ok": False,
                    "error": "Mixed feature schema detected in DB. Clean train.db and regenerate samples.",
                    "init_mode": init_mode
                }

            X = np.frombuffer(X_bytes, np.float32).reshape((-1, FEAT_DIM))
            PI = np.frombuffer(PI_bytes, np.float32).reshape((-1, ACTION_SIZE))
            Z = np.array(Z_values, dtype=np.float32)
            M = np.frombuffer(M_bytes, np.int8).reshape((-1, ACTION_SIZE))

            losses = model.train_batch(
                X, PI, Z, M,
                Q=None,  # DB path: لا توجد quantiles هنا
                learning_rate=float(learning_rate),
                l2_lambda=float(l2_lambda)
            )

            epoch_policy_loss += float(losses["policy_loss"])
            epoch_value_loss += float(losses["value_loss"])
            epoch_quantile_loss += float(losses.get("quantile_loss", 0.0))
            effective_steps += 1

        denom = float(max(1, effective_steps))
        history.append({
            "epoch": float(epoch + 1),
            "policy_loss": float(epoch_policy_loss / denom),
            "value_loss": float(epoch_value_loss / denom),
            "quantile_loss": float(epoch_quantile_loss / denom),
        })

    model.save(model_path)
    return {"ok": True, "init_mode": init_mode, "samples": int(sample_count), "history": history, "model_path": model_path}