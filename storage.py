from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import re
import sqlite3
import random
from datetime import datetime
from engine import GameState

SAVE_DIR = Path("saves")
DB_PATH = Path("train.db")
ACTION_SIZE = 112


def _ensure_dir() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    """
    توليد اسم آمن (لـ base name) بدون مسارات أو رموز خطرة.
    يسمح بالأحرف/الأرقام/_- . وبالعربية والمسافة (ثم تُحوّل إلى _).
    """
    name = (name or "").strip()
    name = re.sub(r"[^a-zA-Z0-9_\-\.\u0600-\u06FF ]+", "", name)
    name = name.replace(" ", "_").strip("._-")
    return name or "game"


def _normalize_save_filename(filename: str) -> str:
    """
    - يأخذ basename فقط (يحذف أي مسارات ضمنية)
    - يضيف .json إن لم تكن موجودة
    """
    fn = (filename or "").strip()
    fn = Path(fn).name  # يمنع ../ و subdirs
    if not fn:
        raise ValueError("Empty filename")
    if not fn.endswith(".json"):
        fn += ".json"
    return fn


def _resolve_save_path(filename: str) -> Path:
    """
    يحول اسم ملف الحفظ إلى مسار آمن داخل SAVE_DIR فقط.
    يرفض أي شيء يحاول الخروج من SAVE_DIR.
    """
    _ensure_dir()
    fn = _normalize_save_filename(filename)

    base_dir = SAVE_DIR.resolve()
    path = (SAVE_DIR / fn).resolve()

    try:
        path.relative_to(base_dir)
    except ValueError:
        raise ValueError("Unauthorized path access")

    return path


def save_game(state: GameState, name: Optional[str] = None) -> str:
    _ensure_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    base = safe_name(name) if name else "game"
    # تقليل احتمال التصادم داخل نفس الثانية بإضافة لاحقة صغيرة
    suffix = f"{random.randint(0, 9999):04d}"
    fn = f"{base}_{ts}_{suffix}.json"

    with open(SAVE_DIR / fn, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

    return fn


def save_game_as(state: GameState, filename: str, overwrite: bool = True) -> str:
    """
    Save state to a specific filename inside saves/ (optionally overwrite).
    filename can be 'x.json' or 'x' (will normalize to .json).
    """
    path = _resolve_save_path(filename)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Save already exists: {path.name}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

    return path.name


def load_game(filename: str) -> GameState:
    """
    تحميل حالة لعبة من saves/.
    آمن ضد path traversal ويقبل:
      - "match1_....json"
      - "match1_...." (سيضيف .json)
    """
    path = _resolve_save_path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Save not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return GameState.from_dict(data)


def list_saves() -> List[str]:
    _ensure_dir()
    return sorted([p.name for p in SAVE_DIR.glob("*.json")], reverse=True)


def delete_save(filename: str) -> str:
    path = _resolve_save_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Save not found: {path.name}")
    path.unlink()
    return path.name


def delete_saves(filenames: List[str]) -> Dict[str, Any]:
    deleted: List[str] = []
    missing: List[str] = []

    for name in filenames:
        try:
            deleted.append(delete_save(str(name)))
        except FileNotFoundError:
            try:
                missing.append(_normalize_save_filename(str(name)))
            except Exception:
                missing.append(str(name))

    return {"deleted": deleted, "missing": missing}


def export_log_text(state: GameState) -> str:
    d = state.to_dict()
    meta = d.get("meta", {})
    board = d.get("board", {})
    lines: List[str] = []
    lines.append("=== Domino AI Log ===")
    lines.append(f"turn: {meta.get('current_turn')} | started_from_beginning: {meta.get('started_from_beginning')}")
    lines.append(f"score: me={meta.get('my_score')} opp={meta.get('opponent_score')}")
    lines.append(f"opp_tiles={meta.get('opponent_tile_count')} boneyard={meta.get('boneyard_count')}")
    lines.append(f"ends: {board.get('ends')}")
    lines.append("")
    lines.append("=== Events ===")
    for i, ev in enumerate(d.get("events", []), start=1):
        lines.append(f"{i:02d}. {ev}")
    lines.append("")
    lines.append("=== Snapshot (JSON) ===")
    lines.append(json.dumps(d, ensure_ascii=False, indent=2))
    return "\n".join(lines)


# =============================================================================
# SQLite helpers (training DB)
# =============================================================================

def _connect(db_path: Path) -> sqlite3.Connection:
    """
    اتصال SQLite مع إعدادات تحسن الاستقرار/الأداء.
    - timeout لتخفيف أخطاء 'database is locked'
    - WAL يسمح بقراءة أثناء كتابة غالبًا
    """
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def init_train_db(db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS samples_pv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feat BLOB NOT NULL,
                pi   BLOB NOT NULL,
                z    REAL NOT NULL,
                mask BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_pv_id ON samples_pv(id)")


def add_samples_pv(rows: List[Tuple[bytes, bytes, float, bytes]], db_path: Path = DB_PATH) -> int:
    init_train_db(db_path)
    if not rows:
        return 0
    with _connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO samples_pv (feat, pi, z, mask) VALUES (?, ?, ?, ?)",
            rows
        )
    return len(rows)


def count_samples_pv(db_path: Path = DB_PATH) -> int:
    init_train_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM samples_pv").fetchone()
    return int(row[0] if row else 0)


def sample_batch_pv(batch_size: int, db_path: Path = DB_PATH) -> List[Tuple[bytes, bytes, float, bytes]]:
    init_train_db(db_path)
    bs = int(batch_size)
    if bs <= 0:
        return []

    with _connect(db_path) as conn:
        row = conn.execute("SELECT MAX(id) FROM samples_pv").fetchone()
        max_id = int(row[0] if row and row[0] is not None else 0)
        if max_id <= 0:
            return []

        pick_n = min(max_id, max(bs * 2, bs))
        ids = random.sample(range(1, max_id + 1), pick_n)

        placeholders = ",".join(["?"] * len(ids))
        rows = conn.execute(
            f"SELECT feat, pi, z, mask FROM samples_pv WHERE id IN ({placeholders})",
            ids
        ).fetchall()

    out = [(r[0], r[1], float(r[2]), r[3]) for r in rows]
    random.shuffle(out)
    return out[:bs]