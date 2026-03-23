"""
api/auth.py
────────────
Multi-user authentication layer.

Database : data/auth/users.db  (SQLite, auto-created on first run)

Schema
------
users
  id               INTEGER PK AUTOINCREMENT
  username         TEXT    UNIQUE NOT NULL  (case-insensitive)
  email            TEXT    UNIQUE NOT NULL  (case-insensitive)
  display_name     TEXT    NOT NULL DEFAULT ''
  hashed_password  TEXT    NOT NULL
  role             TEXT    NOT NULL DEFAULT 'user'   -- 'user' | 'admin'
  avatar_color     TEXT    NOT NULL DEFAULT '#6c63ff'
  is_active        INTEGER NOT NULL DEFAULT 1
  created_at       REAL    NOT NULL
  last_login       REAL

login_attempts
  id          INTEGER PK AUTOINCREMENT
  username    TEXT    NOT NULL
  ip_address  TEXT    NOT NULL DEFAULT ''
  attempted_at REAL   NOT NULL
  success     INTEGER NOT NULL DEFAULT 0

Token format : HS256 JWT signed with AUTH_SECRET_KEY
               payload: sub (user id), username, role, iat, exp
Token lifetime: AUTH_TOKEN_EXPIRE_MINUTES (default 1440 min = 24 h)

Environment variables
---------------------
AUTH_SECRET_KEY              Override the auto-generated signing key (set
                             in .env for stable tokens across restarts)
AUTH_TOKEN_EXPIRE_MINUTES    JWT lifetime in minutes (default 1440)
AUTH_MAX_LOGIN_ATTEMPTS      Failed logins before lockout (default 10)
AUTH_LOCKOUT_MINUTES         Lockout duration in minutes (default 15)
AUTH_OPEN_REGISTRATION       'true' allows anyone to register (default true)
                             Set 'false' to require admin approval
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import bcrypt as _bcrypt_lib
from jose import JWTError, jwt
from pydantic import BaseModel, field_validator, EmailStr

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_DB_DIR   = Path("./data/auth")
_DB_PATH  = _DB_DIR / "users.db"
_DB_DIR.mkdir(parents=True, exist_ok=True)

_SECRET_KEY        = os.getenv("AUTH_SECRET_KEY") or secrets.token_hex(32)
_ALGORITHM         = "HS256"
_EXPIRE_MIN        = int(os.getenv("AUTH_TOKEN_EXPIRE_MINUTES", "1440"))
_MAX_ATTEMPTS      = int(os.getenv("AUTH_MAX_LOGIN_ATTEMPTS", "10"))
_LOCKOUT_MIN       = int(os.getenv("AUTH_LOCKOUT_MINUTES", "15"))
_OPEN_REGISTRATION = os.getenv("AUTH_OPEN_REGISTRATION", "true").lower() != "false"

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\-\.]{3,32}$")

# Palette of avatar colours — assigned round-robin on registration
_AVATAR_COLORS = [
    "#6c63ff", "#22c55e", "#f59e0b", "#38bdf8",
    "#ef4444", "#a78bfa", "#fb923c", "#34d399",
    "#f472b6", "#60a5fa",
]


# ── Pydantic models ───────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username:     str
    email:        str
    password:     str
    display_name: str = ""

    @field_validator("username")
    @classmethod
    def _val_username(cls, v: str) -> str:
        v = v.strip()
        if not _USERNAME_RE.match(v):
            raise ValueError(
                "Username must be 3–32 characters and contain only "
                "letters, numbers, underscores, hyphens, or dots."
            )
        return v

    @field_validator("email")
    @classmethod
    def _val_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email address.")
        return v

    @field_validator("password")
    @classmethod
    def _val_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number.")
        return v


class UserUpdate(BaseModel):
    """Fields a user may update on their own profile."""
    display_name: Optional[str] = None
    email:        Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password:     str

    @field_validator("new_password")
    @classmethod
    def _val_new(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("New password must be at least 8 characters.")
        if not any(c.isdigit() for c in v):
            raise ValueError("New password must contain at least one number.")
        return v


class AdminPasswordReset(BaseModel):
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _val(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class UserOut(BaseModel):
    id:           int
    username:     str
    email:        str
    display_name: str
    role:         str
    avatar_color: str
    is_active:    bool
    created_at:   float
    last_login:   Optional[float]

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str
    display_name: str
    user_id:      int
    role:         str
    avatar_color: str


# ── Database ──────────────────────────────────────────────────────────────────

@contextmanager
def _db():
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables and seed the default admin account if needed."""
    with _db() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                username         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                email            TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                display_name     TEXT    NOT NULL DEFAULT '',
                hashed_password  TEXT    NOT NULL,
                role             TEXT    NOT NULL DEFAULT 'user'
                                         CHECK(role IN ('user','admin')),
                avatar_color     TEXT    NOT NULL DEFAULT '#6c63ff',
                is_active        INTEGER NOT NULL DEFAULT 1,
                created_at       REAL    NOT NULL,
                last_login       REAL
            )
        """)

        # ── Schema migration: add columns added after initial release ─────────
        # SQLite does not support IF NOT EXISTS for ALTER TABLE ADD COLUMN,
        # so we check the existing columns first.
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        migrations = [
            ("display_name",  "TEXT NOT NULL DEFAULT ''"),
            ("role",          "TEXT NOT NULL DEFAULT 'user'"),
            ("avatar_color",  "TEXT NOT NULL DEFAULT '#6c63ff'"),
        ]
        for col, definition in migrations:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
                logger.info("Schema migration: added column 'users.%s'", col)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT    NOT NULL,
                ip_address   TEXT    NOT NULL DEFAULT '',
                attempted_at REAL    NOT NULL,
                success      INTEGER NOT NULL DEFAULT 0
            )
        """)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_login_attempts_username "
            "ON login_attempts(username, attempted_at)"
        )

        # Seed default admin if no users exist
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            _insert_user(
                conn,
                username="admin",
                email="admin@localhost",
                display_name="Administrator",
                password="Admin1234",
                role="admin",
                avatar_color="#6c63ff",
            )
            logger.warning(
                "Auth DB initialised with default admin "
                "(username=admin  password=Admin1234). "
                "Change this immediately via Settings → Change Password."
            )

    logger.info("Auth DB ready: %s", _DB_PATH)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return _bcrypt_lib.hashpw(password.encode(), _bcrypt_lib.gensalt(rounds=12)).decode()


def _verify(password: str, hashed: str) -> bool:
    try:
        return _bcrypt_lib.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def _insert_user(
    conn: sqlite3.Connection,
    *,
    username:     str,
    email:        str,
    display_name: str,
    password:     str,
    role:         str = "user",
    avatar_color: str = "#6c63ff",
) -> int:
    cur = conn.execute(
        "INSERT INTO users "
        "(username, email, display_name, hashed_password, role, avatar_color, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            username.strip(),
            email.strip().lower(),
            display_name.strip() or username.strip(),
            _hash(password),
            role,
            avatar_color,
            time.time(),
        ),
    )
    return cur.lastrowid


def _row_to_user(row: sqlite3.Row) -> UserOut:
    return UserOut(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        display_name=row["display_name"] or row["username"],
        role=row["role"],
        avatar_color=row["avatar_color"],
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        last_login=row["last_login"],
    )


def _next_avatar_color(conn: sqlite3.Connection) -> str:
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    return _AVATAR_COLORS[count % len(_AVATAR_COLORS)]


def _record_attempt(username: str, success: bool, ip: str = "") -> None:
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO login_attempts (username, ip_address, attempted_at, success) "
                "VALUES (?,?,?,?)",
                (username.lower(), ip, time.time(), int(success)),
            )
            # Prune records older than 24 hours to keep the table tidy
            conn.execute(
                "DELETE FROM login_attempts WHERE attempted_at < ?",
                (time.time() - 86400,),
            )
    except Exception as exc:
        logger.debug("Could not record login attempt: %s", exc)


def _is_locked_out(username: str) -> bool:
    """Return True when the user has too many recent failed logins."""
    window = time.time() - _LOCKOUT_MIN * 60
    with _db() as conn:
        failed = conn.execute(
            "SELECT COUNT(*) FROM login_attempts "
            "WHERE username=? AND success=0 AND attempted_at > ?",
            (username.lower(), window),
        ).fetchone()[0]
    return failed >= _MAX_ATTEMPTS


# ── Public API ────────────────────────────────────────────────────────────────

def registration_open() -> bool:
    """True when self-service registration is enabled (default)."""
    return _OPEN_REGISTRATION


def create_user(data: UserCreate) -> UserOut:
    """
    Register a new user account.

    Raises ValueError on:
      - duplicate username or email
      - registration closed (AUTH_OPEN_REGISTRATION=false)

    Validation is handled by UserCreate Pydantic validators.
    """
    if not _OPEN_REGISTRATION:
        raise ValueError(
            "Self-service registration is currently disabled. "
            "Contact an administrator to create your account."
        )
    with _db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username=? OR email=?",
            (data.username.strip(), data.email.strip().lower()),
        ).fetchone()
        if existing:
            raise ValueError("Username or email is already registered.")
        color = _next_avatar_color(conn)
        uid   = _insert_user(
            conn,
            username=data.username,
            email=data.email,
            display_name=data.display_name or data.username,
            password=data.password,
            role="user",
            avatar_color=color,
        )
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    logger.info("New user registered: '%s' (%s)", data.username, data.email)
    return _row_to_user(row)


def admin_create_user(
    data:     UserCreate,
    role:     str = "user",
    is_active: bool = True,
) -> UserOut:
    """Admin-only user creation — bypasses the open-registration flag."""
    with _db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username=? OR email=?",
            (data.username.strip(), data.email.strip().lower()),
        ).fetchone()
        if existing:
            raise ValueError("Username or email is already registered.")
        color = _next_avatar_color(conn)
        uid   = _insert_user(
            conn,
            username=data.username,
            email=data.email,
            display_name=data.display_name or data.username,
            password=data.password,
            role=role,
            avatar_color=color,
        )
        if not is_active:
            conn.execute("UPDATE users SET is_active=0 WHERE id=?", (uid,))
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    logger.info("Admin created user: '%s' (role=%s)", data.username, role)
    return _row_to_user(row)


def authenticate_user(username: str, password: str, ip: str = "") -> Optional[UserOut]:
    """
    Verify credentials.

    Returns UserOut on success, None on failure.
    Records the attempt and applies rate-limiting lockout.
    """
    username = username.strip()

    if _is_locked_out(username):
        logger.warning("Login blocked (rate-limit): '%s' from %s", username, ip)
        return None

    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (username,),
        ).fetchone()

    if not row:
        _record_attempt(username, success=False, ip=ip)
        return None

    if not _verify(password, row["hashed_password"]):
        _record_attempt(username, success=False, ip=ip)
        logger.info("Failed login attempt for '%s'", username)
        return None

    if not row["is_active"]:
        logger.info("Inactive account login attempt: '%s'", username)
        return None

    _record_attempt(username, success=True, ip=ip)
    with _db() as conn:
        conn.execute(
            "UPDATE users SET last_login=? WHERE id=?",
            (time.time(), row["id"]),
        )
    return _row_to_user(row)


def get_user_by_id(user_id: int) -> Optional[UserOut]:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id=?", (user_id,)
        ).fetchone()
    return _row_to_user(row) if row else None


def list_users(
    include_inactive: bool = False,
    search:           str  = "",
    limit:            int  = 100,
    offset:           int  = 0,
) -> list[UserOut]:
    """Return all users (admin use). Optionally filter by search string."""
    conditions = []
    params: list = []

    if not include_inactive:
        conditions.append("is_active=1")

    if search.strip():
        conditions.append("(username LIKE ? OR email LIKE ? OR display_name LIKE ?)")
        s = f"%{search.strip()}%"
        params += [s, s, s]

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params += [limit, offset]

    with _db() as conn:
        rows = conn.execute(
            f"SELECT * FROM users {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
    return [_row_to_user(r) for r in rows]


def update_user_profile(user_id: int, data: UserUpdate) -> UserOut:
    """User updates their own display_name or email."""
    updates: dict = {}
    if data.display_name is not None:
        updates["display_name"] = data.display_name.strip()
    if data.email is not None:
        email = data.email.strip().lower()
        if "@" not in email:
            raise ValueError("Invalid email address.")
        updates["email"] = email

    if not updates:
        user = get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found.")
        return user

    set_clause = ", ".join(f"{k}=?" for k in updates)
    with _db() as conn:
        # Check email uniqueness if being changed
        if "email" in updates:
            clash = conn.execute(
                "SELECT id FROM users WHERE email=? AND id!=?",
                (updates["email"], user_id),
            ).fetchone()
            if clash:
                raise ValueError("Email is already in use.")
        conn.execute(
            f"UPDATE users SET {set_clause} WHERE id=?",
            list(updates.values()) + [user_id],
        )
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _row_to_user(row)


def change_password(user_id: int, data: PasswordChange) -> bool:
    """
    Change a user's password after verifying the current one.
    Returns True on success, False when current_password is wrong.
    """
    with _db() as conn:
        row = conn.execute(
            "SELECT hashed_password FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if not row:
            return False
        if not _verify(data.current_password, row["hashed_password"]):
            return False
        conn.execute(
            "UPDATE users SET hashed_password=? WHERE id=?",
            (_hash(data.new_password), user_id),
        )
    logger.info("Password changed for user id=%d", user_id)
    return True


def admin_reset_password(user_id: int, data: AdminPasswordReset) -> bool:
    """Admin resets any user's password without requiring the old one."""
    with _db() as conn:
        result = conn.execute(
            "UPDATE users SET hashed_password=? WHERE id=?",
            (_hash(data.new_password), user_id),
        )
    return result.rowcount > 0


def set_user_active(user_id: int, active: bool) -> bool:
    """Admin activates or deactivates a user account."""
    with _db() as conn:
        result = conn.execute(
            "UPDATE users SET is_active=? WHERE id=?",
            (int(active), user_id),
        )
    return result.rowcount > 0


def set_user_role(user_id: int, role: str) -> bool:
    """Admin promotes or demotes a user (role: 'user' | 'admin')."""
    if role not in ("user", "admin"):
        raise ValueError("Role must be 'user' or 'admin'.")
    with _db() as conn:
        result = conn.execute(
            "UPDATE users SET role=? WHERE id=?",
            (role, user_id),
        )
    return result.rowcount > 0


def delete_user(user_id: int) -> bool:
    """
    Permanently delete a user account.
    Returns False if the user was not found.
    """
    with _db() as conn:
        result = conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    return result.rowcount > 0


def get_login_stats(user_id: int, days: int = 7) -> dict:
    """Return recent login statistics for a user (for the admin panel)."""
    window = time.time() - days * 86400
    with _db() as conn:
        row = conn.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
        if not row:
            return {}
        username = row["username"]
        total = conn.execute(
            "SELECT COUNT(*) FROM login_attempts WHERE username=? AND attempted_at>?",
            (username.lower(), window),
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM login_attempts "
            "WHERE username=? AND success=0 AND attempted_at>?",
            (username.lower(), window),
        ).fetchone()[0]
    return {"total_attempts": total, "failed_attempts": failed, "days": days}


def user_count() -> dict:
    """Return total / active / admin user counts (for dashboard)."""
    with _db() as conn:
        total  = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
        admins = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
    return {"total": total, "active": active, "admins": admins}


# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(user: UserOut) -> str:
    """Mint a signed HS256 JWT for the given user."""
    now = int(time.time())
    payload = {
        "sub":      str(user.id),
        "username": user.username,
        "role":     user.role,
        "iat":      now,
        "exp":      now + _EXPIRE_MIN * 60,
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT. Returns payload dict or None on failure."""
    try:
        return jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
    except JWTError:
        return None
