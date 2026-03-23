"""
api/auth.py
────────────
Authentication layer: SQLite user database, bcrypt password hashing, JWT tokens.

Database: data/auth/users.db  (created automatically on first startup)

Tables:
  users(id, username, email, hashed_password, created_at, last_login, is_active)

Token format: HS256 JWT signed with AUTH_SECRET_KEY (.env or auto-generated)
Token lifetime: AUTH_TOKEN_EXPIRE_MINUTES (.env, default 1440 = 24 hours)
"""
from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import bcrypt as _bcrypt_lib
from jose import JWTError, jwt
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_DB_DIR  = Path("./data/auth")
_DB_PATH = _DB_DIR / "users.db"
_DB_DIR.mkdir(parents=True, exist_ok=True)

_SECRET_KEY  = os.getenv("AUTH_SECRET_KEY") or secrets.token_hex(32)
_ALGORITHM   = "HS256"
_EXPIRE_MIN  = int(os.getenv("AUTH_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 h

class _PwdCtx:
    @staticmethod
    def hash(password: str) -> str:
        return _bcrypt_lib.hashpw(password.encode(), _bcrypt_lib.gensalt(rounds=12)).decode()
    @staticmethod
    def verify(password: str, hashed: str) -> bool:
        try:
            return _bcrypt_lib.checkpw(password.encode(), hashed.encode())
        except Exception:
            return False

_pwd_ctx = _PwdCtx()

# ── Data models ───────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    email:    str
    password: str


class UserOut(BaseModel):
    id:         int
    username:   str
    email:      str
    created_at: float
    last_login: Optional[float]
    is_active:  bool


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str
    user_id:      int


# ── Database helpers ──────────────────────────────────────────────────────────

@contextmanager
def _db():
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the users table if it does not exist."""
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                email           TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                hashed_password TEXT    NOT NULL,
                created_at      REAL    NOT NULL,
                last_login      REAL,
                is_active       INTEGER NOT NULL DEFAULT 1
            )
        """)
        # Seed a default admin account if the table is empty
        row = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if row == 0:
            _create_user_internal(
                conn,
                username="admin",
                email="admin@localhost",
                password="admin123",
            )
            logger.info(
                "Auth DB seeded with default admin account "
                "(username=admin, password=admin123). "
                "Change this immediately in production."
            )
    logger.info("Auth DB ready at %s", _DB_PATH)


def _create_user_internal(
    conn: sqlite3.Connection,
    username: str,
    email:    str,
    password: str,
) -> int:
    hashed = _pwd_ctx.hash(password)
    cur    = conn.execute(
        "INSERT INTO users (username, email, hashed_password, created_at) "
        "VALUES (?, ?, ?, ?)",
        (username.strip(), email.strip().lower(), hashed, time.time()),
    )
    return cur.lastrowid


# ── Public API ────────────────────────────────────────────────────────────────

def create_user(data: UserCreate) -> UserOut:
    """Create a new user. Raises ValueError on duplicate username/email."""
    if len(data.password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    with _db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username=? OR email=?",
            (data.username.strip(), data.email.strip().lower()),
        ).fetchone()
        if existing:
            raise ValueError("Username or email already registered.")
        uid = _create_user_internal(conn, data.username, data.email, data.password)
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return _row_to_user(row)


def authenticate_user(username: str, password: str) -> Optional[UserOut]:
    """
    Verify credentials. Returns UserOut on success, None on failure.
    Updates last_login timestamp on success.
    """
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1",
            (username.strip(),),
        ).fetchone()
        if not row:
            return None
        if not _pwd_ctx.verify(password, row["hashed_password"]):
            return None
        conn.execute(
            "UPDATE users SET last_login=? WHERE id=?",
            (time.time(), row["id"]),
        )
        return _row_to_user(row)


def get_user_by_id(user_id: int) -> Optional[UserOut]:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id=? AND is_active=1", (user_id,)
        ).fetchone()
    return _row_to_user(row) if row else None


def create_access_token(user: UserOut) -> str:
    """Mint a signed JWT for the given user."""
    payload = {
        "sub":      str(user.id),
        "username": user.username,
        "iat":      int(time.time()),
        "exp":      int(time.time()) + _EXPIRE_MIN * 60,
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT.
    Returns the payload dict on success, None on any error.
    """
    try:
        return jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
    except JWTError:
        return None


def _row_to_user(row: sqlite3.Row) -> UserOut:
    return UserOut(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        created_at=row["created_at"],
        last_login=row["last_login"],
        is_active=bool(row["is_active"]),
    )
