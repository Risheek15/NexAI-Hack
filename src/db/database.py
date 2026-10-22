"""
Database Layer for Maintenance Copilot.
Manages SQLite storage for:
1. Canonical approved logbook records (seeded from CSV records).
2. Pending maintenance fix requests submitted by technicians awaiting admin approval.
3. Real-time audit logs of all queries executed by technicians and administrators.
4. User accounts (Admin and Technician credentials, role-based access, salted PBKDF2 hashing).
5. Cryptographic session token generation and verification.
"""

import os
import time
import json
import base64
import hmac
import sqlite3
import hashlib
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from src.normalizer import normalize_text


DEFAULT_DB_PATH = "data/copilot.db"
SESSION_SECRET = os.environ.get("SESSION_SECRET", "maintenance-copilot-secure-session-key-2026")


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """
    Generates a secure PBKDF2-HMAC-SHA256 password hash with a random 16-byte salt
    and 100,000 iterations.
    """
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.strip().encode("utf-8"), salt, 100_000)
    return f"pbkdf2:sha256:100000:{salt.hex()}:{key.hex()}"


def _verify_password(stored_hash: str, password: str) -> Tuple[bool, bool]:
    """
    Verifies password against stored hash.
    Supports modern PBKDF2-HMAC-SHA256 and legacy unsalted SHA-256 hashes.
    Returns (is_valid, needs_upgrade).
    """
    if not stored_hash or not password:
        return False, False

    # Check PBKDF2 format: pbkdf2:sha256:iterations:salt_hex:hash_hex
    if stored_hash.startswith("pbkdf2:sha256:"):
        parts = stored_hash.split(":")
        if len(parts) == 5:
            try:
                iters = int(parts[2])
                salt = bytes.fromhex(parts[3])
                expected_key = bytes.fromhex(parts[4])
                key = hashlib.pbkdf2_hmac("sha256", password.strip().encode("utf-8"), salt, iters)
                return hmac.compare_digest(key, expected_key), False
            except Exception:
                return False, False

    # Legacy unsalted SHA-256 (64 hex characters)
    legacy_hash = hashlib.sha256(password.strip().encode("utf-8")).hexdigest()
    if hmac.compare_digest(legacy_hash, stored_hash):
        return True, True  # Valid credentials, but needs upgrade to salted PBKDF2

    return False, False


def create_session_token(user_id: int, username: str, role: str) -> str:
    """Creates a cryptographically signed HMAC-SHA256 session token."""
    payload = json.dumps({"uid": user_id, "u": username, "r": role, "ts": int(time.time())})
    b64_payload = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    sig = hmac.new(SESSION_SECRET.encode("utf-8"), b64_payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{b64_payload}.{sig}"


def verify_session_token(token: str, max_age_seconds: int = 86400) -> Optional[Dict[str, Any]]:
    """Verifies HMAC signature and expiration on a session token."""
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        b64_payload, sig = parts
        expected_sig = hmac.new(SESSION_SECRET.encode("utf-8"), b64_payload.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        raw = base64.urlsafe_b64decode(b64_payload.encode("ascii")).decode("utf-8")
        data = json.loads(raw)
        if time.time() - data.get("ts", 0) > max_age_seconds:
            return None
        return {"id": data["uid"], "username": data["u"], "role": data["r"]}
    except Exception:
        return None


class DatabaseManager:
    """
    Thread-safe SQLite database manager for Maintenance Copilot records,
    pending fix requests, technician query audit logging, and user authentication.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initializes database tables and default accounts if they do not exist."""
        with self._get_connection() as conn:
            # 1. Canonical Approved Logbook Records Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    record_id TEXT PRIMARY KEY,
                    problem_raw TEXT NOT NULL,
                    action_raw TEXT NOT NULL,
                    problem_normalized TEXT NOT NULL,
                    action_normalized TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'approved',
                    submitted_by TEXT NOT NULL DEFAULT 'SYSTEM',
                    submitted_at TEXT NOT NULL,
                    approved_by TEXT NOT NULL DEFAULT 'SYSTEM',
                    approved_at TEXT NOT NULL
                );
            """)

            # 2. Pending Fix Submissions Queue Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    temp_id TEXT UNIQUE NOT NULL,
                    problem_raw TEXT NOT NULL,
                    action_raw TEXT NOT NULL,
                    problem_normalized TEXT NOT NULL,
                    action_normalized TEXT NOT NULL,
                    submitted_by TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    admin_notes TEXT DEFAULT '',
                    approved_record_id TEXT DEFAULT NULL
                );
            """)

            # 3. Technician Query Audit Logs Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    query TEXT NOT NULL,
                    evidence_status TEXT NOT NULL,
                    evidence_reason TEXT NOT NULL,
                    cited_record_ids TEXT NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)

            # 4. User Accounts Table (Admin & Technicians)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_active TEXT DEFAULT NULL
                );
            """)
            conn.commit()

            # Ensure last_active column exists if migrated
            try:
                conn.execute("ALTER TABLE users ADD COLUMN last_active TEXT DEFAULT NULL;")
                conn.commit()
            except Exception:
                pass

            # Seed default Admin and 3 Technician accounts if empty using salted PBKDF2
            user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if user_count == 0:
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                default_users = [
                    ("admin", _hash_password("admin123"), "Airport Administrator", "admin", now),
                    ("technician1", _hash_password("password1"), "Technician 1", "technician", now),
                    ("technician2", _hash_password("password2"), "Technician 2", "technician", now),
                    ("technician3", _hash_password("password3"), "Technician 3", "technician", now),
                ]
                conn.executemany("""
                    INSERT OR IGNORE INTO users (username, password_hash, full_name, role, created_at)
                    VALUES (?, ?, ?, ?, ?);
                """, default_users)
                conn.commit()

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Verifies username and password against the SQLite users table.
        Transparently migrates legacy unsalted SHA-256 hashes to salted PBKDF2 upon successful verification.
        """
        clean_user = username.strip().lower()

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, full_name, role, created_at FROM users WHERE lower(username) = ?",
                (clean_user,)
            ).fetchone()
            if not row:
                return None

            is_valid, needs_upgrade = _verify_password(row["password_hash"], password)
            if not is_valid:
                return None

            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if needs_upgrade:
                # Transparently upgrade legacy hash to salted PBKDF2
                upgraded_hash = _hash_password(password)
                try:
                    conn.execute(
                        "UPDATE users SET password_hash = ?, last_active = ? WHERE id = ?",
                        (upgraded_hash, now, row["id"])
                    )
                    conn.commit()
                except Exception:
                    pass
            else:
                try:
                    conn.execute("UPDATE users SET last_active = ? WHERE id = ?", (now, row["id"]))
                    conn.commit()
                except Exception:
                    pass

            user_dict = {
                "id": row["id"],
                "username": row["username"],
                "full_name": row["full_name"],
                "role": row["role"],
                "created_at": row["created_at"],
                "last_active": now
            }
            return user_dict

    def create_user(
        self,
        username: str,
        password: str,
        full_name: str,
        role: str = "technician"
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Airport Administrator creates a new technician or admin account.
        Stores credentials with salted PBKDF2 hashing.
        """
        clean_user = username.strip().lower()
        if not clean_user or len(clean_user) < 3:
            return False, "Username must be at least 3 characters.", None
        if not password or len(password.strip()) < 4:
            return False, "Password must be at least 4 characters.", None
        if not full_name.strip():
            return False, "Full name is required.", None

        pass_hash = _hash_password(password)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self._get_connection() as conn:
            try:
                cursor = conn.execute("""
                    INSERT INTO users (username, password_hash, full_name, role, created_at)
                    VALUES (?, ?, ?, ?, ?);
                """, (clean_user, pass_hash, full_name.strip(), role.lower(), now))
                conn.commit()
                new_id = cursor.lastrowid
                user_info = {
                    "id": new_id,
                    "username": clean_user,
                    "full_name": full_name.strip(),
                    "role": role.lower(),
                    "created_at": now,
                    "last_active": None
                }
                return True, "User created successfully.", user_info
            except sqlite3.IntegrityError:
                return False, f"Username '{clean_user}' already exists.", None
            except Exception as e:
                return False, f"Failed to create user: {e}", None

    def get_users(self, role_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all registered users (excluding sensitive password hashes)."""
        with self._get_connection() as conn:
            if role_filter:
                rows = conn.execute(
                    "SELECT id, username, full_name, role, created_at, last_active FROM users WHERE role = ? ORDER BY id ASC",
                    (role_filter,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, username, full_name, role, created_at, last_active FROM users ORDER BY id ASC"
                ).fetchall()
            return [dict(r) for r in rows]

    def delete_user(self, user_id: int) -> Tuple[bool, str]:
        """Airport Administrator deletes a technician account."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            if cursor.rowcount > 0:
                return True, "User deleted successfully."
            return False, f"User with ID {user_id} not found."

    def get_stats(self) -> Dict[str, int]:
        """Calculates global administrative and airport system stats."""
        with self._get_connection() as conn:
            total_records = conn.execute("SELECT COUNT(*) FROM records WHERE status = 'approved'").fetchone()[0]
            pending_fixes = conn.execute("SELECT COUNT(*) FROM pending_records WHERE status = 'pending'").fetchone()[0]
            total_technicians = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'technician'").fetchone()[0]
            total_queries = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]

            return {
                "total_records": total_records,
                "pending_fixes": pending_fixes,
                "total_technicians": total_technicians,
                "total_queries": total_queries
            }

    def seed_from_jsonl(self, jsonl_path: str = "data/processed/records.jsonl") -> int:
        """Seeds canonical initial records from records.jsonl into the SQLite `records` table."""
        path = Path(jsonl_path)
        if not path.is_file():
            return 0

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            existing_count = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            if existing_count > 0:
                return existing_count

            rows_to_insert = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    rows_to_insert.append((
                        item["record_id"],
                        item["problem_raw"],
                        item["action_raw"],
                        item.get("problem_normalized", item["problem_raw"]),
                        item.get("action_normalized", item["action_raw"]),
                        "approved",
                        "INITIAL_CSV",
                        now,
                        "SYSTEM",
                        now
                    ))

            conn.executemany("""
                INSERT OR IGNORE INTO records (
                    record_id, problem_raw, action_raw, problem_normalized, action_normalized,
                    status, submitted_by, submitted_at, approved_by, approved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, rows_to_insert)
            conn.commit()
            return len(rows_to_insert)

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an approved record by its primary key record_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM records WHERE record_id = ? AND status = 'approved'",
                (record_id,)
            ).fetchone()
            if row:
                return dict(row)
            return None

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Returns all approved records."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM records WHERE status = 'approved' ORDER BY record_id ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def submit_pending_fix(
        self,
        problem_raw: str,
        action_raw: str,
        submitted_by: str = "Technician"
    ) -> Dict[str, Any]:
        """
        Technician submits a new observed discrepancy and corrective action fix.
        Normalized automatically, queued in pending_records for Admin approval.
        """
        p_raw = problem_raw.strip()
        a_raw = action_raw.strip()
        p_norm = normalize_text(p_raw)
        a_norm = normalize_text(a_raw)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp_id = f"REQ-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(2).hex().upper()}"

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO pending_records (
                    temp_id, problem_raw, action_raw, problem_normalized, action_normalized,
                    submitted_by, submitted_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending');
            """, (temp_id, p_raw, a_raw, p_norm, a_norm, submitted_by, now))
            conn.commit()

        return {
            "temp_id": temp_id,
            "problem_raw": p_raw,
            "action_raw": a_raw,
            "problem_normalized": p_norm,
            "action_normalized": a_norm,
            "submitted_by": submitted_by,
            "submitted_at": now,
            "status": "pending"
        }

    def get_pending_fixes(self, status_filter: str = "pending") -> List[Dict[str, Any]]:
        """Fetches pending fix requests for Admin inspection."""
        with self._get_connection() as conn:
            if status_filter == "all":
                rows = conn.execute(
                    "SELECT * FROM pending_records ORDER BY submitted_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM pending_records WHERE status = ? ORDER BY submitted_at DESC",
                    (status_filter,)
                ).fetchall()
            return [dict(r) for r in rows]

    def approve_pending_fix(
        self,
        temp_id: str,
        admin_name: str = "Airport Administrator"
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Airport Administrator approves a technician's pending fix.
        Assigns the next sequential 6-digit ID (e.g. 106170), moves it into the
        official `records` table, and returns the approved record data.
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            # 1. Fetch pending entry
            pending = conn.execute(
                "SELECT * FROM pending_records WHERE temp_id = ? AND status = 'pending'",
                (temp_id,)
            ).fetchone()

            if not pending:
                return False, None, None

            # 2. Determine next record_id
            max_row = conn.execute("""
                SELECT MAX(CAST(record_id AS INTEGER)) FROM records
                WHERE record_id GLOB '[0-9][0-9][0-9][0-9][0-9][0-9]'
            """).fetchone()

            max_id = max_row[0] if max_row and max_row[0] else 106169
            new_id = str(max_id + 1)

            # 3. Insert into official records table
            conn.execute("""
                INSERT INTO records (
                    record_id, problem_raw, action_raw, problem_normalized, action_normalized,
                    status, submitted_by, submitted_at, approved_by, approved_at
                ) VALUES (?, ?, ?, ?, ?, 'approved', ?, ?, ?, ?);
            """, (
                new_id,
                pending["problem_raw"],
                pending["action_raw"],
                pending["problem_normalized"],
                pending["action_normalized"],
                pending["submitted_by"],
                pending["submitted_at"],
                admin_name,
                now
            ))

            # 4. Update pending_records status
            conn.execute("""
                UPDATE pending_records
                SET status = 'approved', approved_record_id = ?, admin_notes = 'Approved by admin'
                WHERE temp_id = ?;
            """, (new_id, temp_id))

            conn.commit()

            approved_record = {
                "record_id": new_id,
                "problem_raw": pending["problem_raw"],
                "action_raw": pending["action_raw"],
                "problem_normalized": pending["problem_normalized"],
                "action_normalized": pending["action_normalized"],
                "submitted_by": pending["submitted_by"],
                "approved_by": admin_name,
                "approved_at": now
            }
            return True, new_id, approved_record

    def reject_pending_fix(
        self,
        temp_id: str,
        admin_name: str = "Airport Administrator",
        reason: str = "Insufficient documentation or duplicate"
    ) -> bool:
        """Admin rejects a pending fix request."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE pending_records
                SET status = 'rejected', admin_notes = ?
                WHERE temp_id = ? AND status = 'pending';
            """, (f"Rejected by {admin_name}: {reason}", temp_id))
            conn.commit()
            return cursor.rowcount > 0

    def log_query(
        self,
        user_name: str,
        user_role: str,
        query: str,
        evidence_status: str,
        evidence_reason: str,
        cited_record_ids: List[str],
        execution_time_ms: float
    ) -> None:
        """Logs technician or admin maintenance query for compliance and auditing."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cited_str = json.dumps(cited_record_ids)
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO query_logs (
                    user_name, user_role, query, evidence_status, evidence_reason,
                    cited_record_ids, execution_time_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (user_name, user_role, query, evidence_status, evidence_reason, cited_str, execution_time_ms, now))
            try:
                conn.execute(
                    "UPDATE users SET last_active = ? WHERE lower(username) = lower(?) OR lower(full_name) = lower(?)",
                    (now, user_name, user_name)
                )
            except Exception:
                pass
            conn.commit()

    def get_query_logs(self, limit: int = 50, user_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Admin views audit history of all technician and admin queries, or technician views their own logs."""
        with self._get_connection() as conn:
            if user_name:
                clean_target = user_name.strip().lower()
                clean_no_spaces = clean_target.replace(" ", "")
                rows = conn.execute(
                    """SELECT * FROM query_logs 
                       WHERE lower(user_name) = ? 
                          OR lower(replace(user_name, ' ', '')) = ?
                          OR lower(user_name) LIKE ?
                       ORDER BY id DESC LIMIT ?""",
                    (clean_target, clean_no_spaces, f"%{clean_no_spaces}%", limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["cited_record_ids"] = json.loads(d["cited_record_ids"])
                except Exception:
                    d["cited_record_ids"] = []
                result.append(d)
            return result
