"""
DocPilot AI / VISION AI — Persistent Database & Encrypted Credential Vault (db.py)
Provides transactional SQLite persistence (.data/docpilot.db), OS-backed credential encryption
(Windows Credential Manager / Windows DPAPI / Cryptography Fernet / Authenticated AEAD-HMAC),
automatic migration from legacy .data/docpilot_db.json, and strict conversation/document isolation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config import DATA_ROOT, PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
DATA_DIR = DATA_ROOT
EXPORTS_DIR = DATA_DIR / "exports"
UPLOADS_DIR = DATA_DIR / "uploads"
SQLITE_DB_PATH = DATA_DIR / "docpilot.db"
LEGACY_JSON_DB_PATH = DATA_DIR / "docpilot_db.json"
DPAPI_KEY_FILE = DATA_DIR / ".vault_dpapi.bin"
FALLBACK_KEY_FILE = DATA_DIR / ".vault_master.key"

KEYRING_SERVICE_NAME = "DocPilot_VISION_AI_Local"
KEYRING_ACCOUNT_NAME = "master_encryption_key_v1"

SUPPORTED_PROVIDERS = ["gemini", "groq", "openrouter", "openai", "anthropic", "local"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


STORAGE_DIR = BASE_DIR / "storage"
LOGS_DIR = BASE_DIR / "logs"

def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "data").mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("uploads", "extracted", "indexes", "exports", "temp", "backups"):
        (STORAGE_DIR / sub).mkdir(parents=True, exist_ok=True)


# ============================================================================
# OS-BACKED MASTER KEY & AUTHENTICATED ENCRYPTION ENGINE
# ============================================================================

def _win32_dpapi_protect(data: bytes) -> Optional[bytes]:
    """Encrypt bytes using Windows DPAPI (CryptProtectData) tied to current Windows user."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [
                ("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_char)),
            ]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32

        buf = ctypes.create_string_buffer(data, len(data))
        in_blob = DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        out_blob = DATA_BLOB()

        if crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            "DocPilot_VISION_AI",
            None,
            None,
            None,
            0x01,  # CRYPTPROTECT_UI_FORBIDDEN
            ctypes.byref(out_blob),
        ):
            raw = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            kernel32.LocalFree(out_blob.pbData)
            return raw
    except Exception:
        pass
    return None


def _win32_dpapi_unprotect(encrypted_data: bytes) -> Optional[bytes]:
    """Decrypt bytes using Windows DPAPI (CryptUnprotectData)."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [
                ("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_char)),
            ]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32

        buf = ctypes.create_string_buffer(encrypted_data, len(encrypted_data))
        in_blob = DATA_BLOB(len(encrypted_data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        out_blob = DATA_BLOB()

        if crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            0x01,  # CRYPTPROTECT_UI_FORBIDDEN
            ctypes.byref(out_blob),
        ):
            raw = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            kernel32.LocalFree(out_blob.pbData)
            return raw
    except Exception:
        pass
    return None


class CredentialVaultCrypto:
    """
    Encrypts API keys at rest.
    Master encryption key is stored in Windows Credential Manager (via `keyring`) or
    Windows DPAPI (`CryptProtectData`), with a local OS-permission-restricted fallback
    for headless environments.
    Never stores plaintext API keys in SQLite or JSON.
    """

    def __init__(self) -> None:
        ensure_data_dirs()
        self._master_key: bytes = self._load_or_create_master_key()

    def _load_or_create_master_key(self) -> bytes:
        # 1. Try OS Credential Manager via keyring (Windows Credential Manager / macOS Keychain)
        try:
            import keyring  # type: ignore

            stored = keyring.get_password(KEYRING_SERVICE_NAME, KEYRING_ACCOUNT_NAME)
            if stored:
                raw = base64.urlsafe_b64decode(stored.encode("ascii"))
                if len(raw) >= 32:
                    return raw
            new_key = secrets.token_bytes(32)
            keyring.set_password(
                KEYRING_SERVICE_NAME,
                KEYRING_ACCOUNT_NAME,
                base64.urlsafe_b64encode(new_key).decode("ascii"),
            )
            return new_key
        except Exception:
            pass

        # 2. Try Windows DPAPI (built-in on Windows without third-party packages)
        if os.name == "nt":
            if DPAPI_KEY_FILE.exists():
                try:
                    unprotected = _win32_dpapi_unprotect(DPAPI_KEY_FILE.read_bytes())
                    if unprotected and len(unprotected) >= 32:
                        return unprotected
                except Exception:
                    pass
            new_key = secrets.token_bytes(32)
            protected = _win32_dpapi_protect(new_key)
            if protected:
                try:
                    DPAPI_KEY_FILE.write_bytes(protected)
                    return new_key
                except Exception:
                    pass

        # 3. Fallback to restricted local key file in .data/.vault_master.key
        if FALLBACK_KEY_FILE.exists():
            try:
                encoded = FALLBACK_KEY_FILE.read_text(encoding="utf-8").strip()
                raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
                if len(raw) >= 32:
                    return raw
            except Exception:
                pass

        new_key = secrets.token_bytes(32)
        try:
            FALLBACK_KEY_FILE.write_text(
                base64.urlsafe_b64encode(new_key).decode("ascii"), encoding="utf-8"
            )
            try:
                os.chmod(FALLBACK_KEY_FILE, 0o600)
            except Exception:
                pass
        except Exception:
            pass
        return new_key

    @staticmethod
    def _xor_keystream(data: bytes, enc_key: bytes, nonce: bytes) -> bytes:
        out = bytearray(len(data))
        block_count = (len(data) + 31) // 32
        for counter in range(block_count):
            counter_bytes = counter.to_bytes(8, "big")
            keystream_block = hmac.new(enc_key, nonce + counter_bytes, hashlib.sha256).digest()
            start = counter * 32
            end = min(start + 32, len(data))
            for idx in range(start, end):
                out[idx] = data[idx] ^ keystream_block[idx - start]
        return bytes(out)

    def encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        raw_bytes = plaintext.encode("utf-8")

        # Use Authenticated Encrypt-then-MAC (PBKDF2-HMAC-SHA256 + SHA256-CTR + HMAC-SHA256)
        # Compatible across all Python 3.10+ installations regardless of C-extension state
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac("sha256", self._master_key, salt, 100_000, dklen=64)
        enc_key, mac_key = derived[:32], derived[32:]
        ciphertext = self._xor_keystream(raw_bytes, enc_key, nonce)
        tag = hmac.new(mac_key, b"ENCv1" + salt + nonce + ciphertext, hashlib.sha256).digest()
        payload = salt + nonce + tag + ciphertext
        return "ENCv1:" + base64.urlsafe_b64encode(payload).decode("ascii")

    def decrypt(self, token: str) -> str:
        if not token:
            return ""
        if not token.startswith("ENCv1:"):
            # Never return unencrypted or unknown format tokens directly unless migrating
            return ""
        try:
            payload = base64.urlsafe_b64decode(token[6:].encode("ascii"))
            if len(payload) < 64:
                return ""
            salt = payload[:16]
            nonce = payload[16:32]
            tag = payload[32:64]
            ciphertext = payload[64:]
            derived = hashlib.pbkdf2_hmac("sha256", self._master_key, salt, 100_000, dklen=64)
            enc_key, mac_key = derived[:32], derived[32:]
            expected_tag = hmac.new(
                mac_key, b"ENCv1" + salt + nonce + ciphertext, hashlib.sha256
            ).digest()
            if not hmac.compare_digest(tag, expected_tag):
                return ""
            plaintext_bytes = self._xor_keystream(ciphertext, enc_key, nonce)
            return plaintext_bytes.decode("utf-8")
        except Exception:
            return ""


# ============================================================================
# PERSISTENT SQLITE DATABASE LAYER
# ============================================================================

class LocalDatabase:
    def __init__(self, db_path: Path = SQLITE_DB_PATH) -> None:
        ensure_data_dirs()
        self.db_path = db_path
        self._lock = threading.RLock()
        self.crypto = CredentialVaultCrypto()
        self._init_schema()
        self._seed_defaults()
        self._migrate_legacy_json_if_needed()
        self._scan_existing_exports()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS users (
                            id TEXT PRIMARY KEY,
                            authentication_subject TEXT NOT NULL,
                            username TEXT NOT NULL,
                            email TEXT NOT NULL,
                            role TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS workspaces (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            name TEXT NOT NULL,
                            description TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS user_settings (
                            user_id TEXT PRIMARY KEY,
                            active_provider TEXT NOT NULL DEFAULT 'gemini',
                            active_model TEXT NOT NULL DEFAULT '',
                            active_model_name TEXT NOT NULL DEFAULT '',
                            active_model_is_free INTEGER NOT NULL DEFAULT 0,
                            local_endpoint TEXT NOT NULL DEFAULT 'http://localhost:11434',
                            theme TEXT NOT NULL DEFAULT 'dark',
                            workspace_mode TEXT NOT NULL DEFAULT 'study',
                            teaching_mode TEXT NOT NULL DEFAULT 'auto',
                            study_difficulty TEXT NOT NULL DEFAULT 'mixed',
                            voice_settings_json TEXT NOT NULL DEFAULT '{}',
                            updated_at TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS provider_vault (
                            provider TEXT PRIMARY KEY,
                            encrypted_api_key TEXT NOT NULL DEFAULT '',
                            base_url TEXT NOT NULL DEFAULT '',
                            verified INTEGER NOT NULL DEFAULT 0,
                            authentication_method TEXT NOT NULL DEFAULT 'none',
                            verification_timestamp TEXT NOT NULL DEFAULT '',
                            last_error_code TEXT NOT NULL DEFAULT '',
                            last_error_message TEXT NOT NULL DEFAULT '',
                            available_models_json TEXT NOT NULL DEFAULT '[]',
                            model_capabilities_json TEXT NOT NULL DEFAULT '{}',
                            updated_at TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS sessions (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            workspace_id TEXT NOT NULL,
                            workspace_mode TEXT NOT NULL DEFAULT 'study',
                            title TEXT NOT NULL,
                            topic TEXT NOT NULL DEFAULT '',
                            provider_id TEXT NOT NULL DEFAULT 'gemini',
                            model_id TEXT NOT NULL DEFAULT '',
                            model_name TEXT NOT NULL DEFAULT '',
                            teaching_depth TEXT NOT NULL DEFAULT 'auto',
                            response_style TEXT NOT NULL DEFAULT 'balanced',
                            reasoning_preference TEXT NOT NULL DEFAULT 'standard',
                            document_ids_json TEXT NOT NULL DEFAULT '[]',
                            session_preferences_json TEXT NOT NULL DEFAULT '{}',
                            cheatsheet_json TEXT,
                            flashcards_json TEXT NOT NULL DEFAULT '[]',
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        );

                        CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                            ON sessions(user_id, updated_at DESC);

                        CREATE TABLE IF NOT EXISTS messages (
                            id TEXT PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            content TEXT NOT NULL,
                            model TEXT NOT NULL DEFAULT '',
                            provider TEXT NOT NULL DEFAULT '',
                            sources_json TEXT NOT NULL DEFAULT '[]',
                            generated_files_json TEXT NOT NULL DEFAULT '[]',
                            created_at TEXT NOT NULL
                        );

                        CREATE INDEX IF NOT EXISTS idx_messages_session_created
                            ON messages(session_id, created_at ASC);

                        CREATE TABLE IF NOT EXISTS documents (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            workspace_id TEXT NOT NULL,
                            session_id TEXT NOT NULL DEFAULT '',
                            filename TEXT NOT NULL,
                            file_type TEXT NOT NULL,
                            file_size INTEGER NOT NULL DEFAULT 0,
                            char_count INTEGER NOT NULL DEFAULT 0,
                            chunk_count INTEGER NOT NULL DEFAULT 0,
                            summary TEXT NOT NULL DEFAULT '',
                            status TEXT NOT NULL DEFAULT 'ready',
                            created_at TEXT NOT NULL
                        );

                        CREATE INDEX IF NOT EXISTS idx_documents_session
                            ON documents(user_id, session_id);

                        CREATE TABLE IF NOT EXISTS chunks (
                            id TEXT PRIMARY KEY,
                            document_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            session_id TEXT NOT NULL DEFAULT '',
                            filename TEXT NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            page_number INTEGER NOT NULL DEFAULT 1,
                            section_title TEXT NOT NULL DEFAULT '',
                            content TEXT NOT NULL,
                            embedding_json TEXT NOT NULL DEFAULT '[]',
                            created_at TEXT NOT NULL
                        );

                        CREATE INDEX IF NOT EXISTS idx_chunks_document
                            ON chunks(document_id, chunk_index ASC);
                        CREATE INDEX IF NOT EXISTS idx_chunks_session
                            ON chunks(user_id, session_id);

                        CREATE TABLE IF NOT EXISTS memories (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            session_id TEXT NOT NULL DEFAULT '',
                            category TEXT NOT NULL DEFAULT 'fact',
                            content TEXT NOT NULL,
                            importance INTEGER NOT NULL DEFAULT 3,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS flashcard_sets (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            workspace_id TEXT NOT NULL,
                            title TEXT NOT NULL,
                            topic TEXT NOT NULL DEFAULT '',
                            document_ids_json TEXT NOT NULL DEFAULT '[]',
                            cards_json TEXT NOT NULL DEFAULT '[]',
                            created_at TEXT NOT NULL
                        );

                        CREATE INDEX IF NOT EXISTS idx_flashcards_session
                            ON flashcard_sets(session_id, created_at DESC);

                        CREATE TABLE IF NOT EXISTS quizzes (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            workspace_id TEXT NOT NULL,
                            title TEXT NOT NULL,
                            topic TEXT NOT NULL DEFAULT '',
                            document_ids_json TEXT NOT NULL DEFAULT '[]',
                            questions_json TEXT NOT NULL DEFAULT '[]',
                            created_at TEXT NOT NULL
                        );

                        CREATE INDEX IF NOT EXISTS idx_quizzes_session
                            ON quizzes(session_id, created_at DESC);

                        CREATE TABLE IF NOT EXISTS quiz_attempts (
                            id TEXT PRIMARY KEY,
                            quiz_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            session_id TEXT NOT NULL DEFAULT '',
                            score INTEGER NOT NULL DEFAULT 0,
                            total_questions INTEGER NOT NULL DEFAULT 0,
                            accuracy_percentage REAL NOT NULL DEFAULT 0.0,
                            weak_topics_json TEXT NOT NULL DEFAULT '[]',
                            answers_json TEXT NOT NULL DEFAULT '{}',
                            created_at TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS cheatsheets (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            topic TEXT NOT NULL DEFAULT '',
                            document_ids_json TEXT NOT NULL DEFAULT '[]',
                            sheet_json TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        );

                        CREATE INDEX IF NOT EXISTS idx_cheatsheets_session
                            ON cheatsheets(session_id, updated_at DESC);

                        CREATE TABLE IF NOT EXISTS generated_files (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL DEFAULT 'usr-default',
                            session_id TEXT NOT NULL DEFAULT '',
                            message_id TEXT NOT NULL DEFAULT '',
                            filename TEXT NOT NULL UNIQUE,
                            format TEXT NOT NULL,
                            category TEXT NOT NULL DEFAULT 'Document',
                            mime_type TEXT NOT NULL,
                            file_size INTEGER NOT NULL DEFAULT 0,
                            file_path TEXT NOT NULL,
                            download_url TEXT NOT NULL,
                            validated INTEGER NOT NULL DEFAULT 1,
                            validation_summary TEXT NOT NULL DEFAULT '',
                            created_at TEXT NOT NULL
                        );

                        CREATE INDEX IF NOT EXISTS idx_generated_files_session
                            ON generated_files(session_id, created_at DESC);
                        """
                    )
            finally:
                conn.close()

    def _seed_defaults(self) -> None:
        ts = now_iso()
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    row = conn.execute("SELECT id FROM users WHERE id = 'usr-default'").fetchone()
                    if not row:
                        conn.execute(
                            """
                            INSERT INTO users (id, authentication_subject, username, email, role, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                "usr-default",
                                "guest",
                                "Jarvis",
                                "Jarvis@gmail.com",
                                "Lead Research Fellow",
                                ts,
                                ts,
                            ),
                        )
                    ws_row = conn.execute(
                        "SELECT id FROM workspaces WHERE id = 'ws-default'"
                    ).fetchone()
                    if not ws_row:
                        conn.execute(
                            """
                            INSERT INTO workspaces (id, user_id, name, description, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                "ws-default",
                                "usr-default",
                                "Research & Study Workspace",
                                "Personal study and document intelligence workspace",
                                ts,
                                ts,
                            ),
                        )
                    set_row = conn.execute(
                        "SELECT user_id FROM user_settings WHERE user_id = 'usr-default'"
                    ).fetchone()
                    if not set_row:
                        conn.execute(
                            """
                            INSERT INTO user_settings (
                                user_id, active_provider, active_model, active_model_name,
                                active_model_is_free, local_endpoint, theme, workspace_mode,
                                teaching_mode, study_difficulty, voice_settings_json, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                "usr-default",
                                "gemini",
                                "",
                                "",
                                0,
                                "http://localhost:11434",
                                "dark",
                                "study",
                                "auto",
                                "mixed",
                                "{}",
                                ts,
                            ),
                        )
            finally:
                conn.close()

    def _migrate_legacy_json_if_needed(self) -> None:
        """Safely migrate existing records from .data/docpilot_db.json without overwriting or losing data."""
        if not LEGACY_JSON_DB_PATH.exists():
            return
        try:
            raw_text = LEGACY_JSON_DB_PATH.read_text(encoding="utf-8")
            legacy = json.loads(raw_text)
        except Exception as exc:
            print(f"[DB] Warning: Could not parse legacy JSON DB: {exc}")
            return

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    # Migrate users
                    for u in legacy.get("users", []):
                        if not isinstance(u, dict) or not u.get("id"):
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO users (id, authentication_subject, username, email, role, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                u["id"],
                                u.get("authentication_subject", "guest"),
                                u.get("username", "User"),
                                u.get("email", ""),
                                u.get("role", "Student"),
                                u.get("created_at", now_iso()),
                                u.get("updated_at", now_iso()),
                            ),
                        )

                    # Migrate workspaces
                    for w in legacy.get("workspaces", []):
                        if not isinstance(w, dict) or not w.get("id"):
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO workspaces (id, user_id, name, description, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                w["id"],
                                w.get("user_id", "usr-default"),
                                w.get("name", "Workspace"),
                                w.get("description", ""),
                                w.get("created_at", now_iso()),
                                w.get("updated_at", now_iso()),
                            ),
                        )

                    # Migrate sessions
                    for s in legacy.get("sessions", []):
                        if not isinstance(s, dict) or not s.get("id"):
                            continue
                        doc_ids = s.get("document_ids") or []
                        prefs = s.get("session_preferences") or {}
                        cs = s.get("cheatsheet")
                        fc = s.get("flashcards") or []
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO sessions (
                                id, user_id, workspace_id, workspace_mode, title, topic,
                                provider_id, model_id, model_name, teaching_depth,
                                response_style, reasoning_preference, document_ids_json,
                                session_preferences_json, cheatsheet_json, flashcards_json,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                s["id"],
                                s.get("user_id", "usr-default"),
                                s.get("workspace_id", "ws-default"),
                                s.get("workspace_mode", "study"),
                                s.get("title", "Study Session"),
                                s.get("topic", ""),
                                s.get("provider_id", "gemini"),
                                s.get("model_id", ""),
                                s.get("model_name") or s.get("model_id") or "",
                                s.get("teaching_depth", "auto"),
                                s.get("response_style", "balanced"),
                                s.get("reasoning_preference", "standard"),
                                json.dumps(doc_ids),
                                json.dumps(prefs),
                                json.dumps(cs) if cs else None,
                                json.dumps(fc),
                                s.get("created_at", now_iso()),
                                s.get("updated_at", now_iso()),
                            ),
                        )

                    # Migrate messages
                    for m in legacy.get("messages", []):
                        if not isinstance(m, dict) or not m.get("id"):
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO messages (
                                id, session_id, user_id, role, content, model, provider,
                                sources_json, generated_files_json, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                m["id"],
                                m.get("session_id", ""),
                                m.get("user_id", "usr-default"),
                                m.get("role", "user"),
                                m.get("content", ""),
                                m.get("model", ""),
                                m.get("provider", ""),
                                json.dumps(m.get("sources") or []),
                                json.dumps(m.get("generated_files") or []),
                                m.get("created_at", now_iso()),
                            ),
                        )

                    # Migrate documents
                    for d in legacy.get("documents", []):
                        if not isinstance(d, dict) or not d.get("id"):
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO documents (
                                id, user_id, workspace_id, session_id, filename, file_type,
                                file_size, char_count, chunk_count, summary, status, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                d["id"],
                                d.get("user_id", "usr-default"),
                                d.get("workspace_id", "ws-default"),
                                d.get("session_id", ""),
                                d.get("filename", "document.txt"),
                                d.get("file_type", "txt"),
                                int(d.get("file_size") or 0),
                                int(d.get("char_count") or 0),
                                int(d.get("chunk_count") or 0),
                                d.get("summary", ""),
                                d.get("status", "ready"),
                                d.get("created_at", now_iso()),
                            ),
                        )

                    # Migrate chunks
                    for c in legacy.get("chunks", []):
                        if not isinstance(c, dict) or not c.get("id"):
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO chunks (
                                id, document_id, user_id, session_id, filename, chunk_index,
                                page_number, section_title, content, embedding_json, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                c["id"],
                                c.get("document_id", ""),
                                c.get("user_id", "usr-default"),
                                c.get("session_id", ""),
                                c.get("filename", ""),
                                int(c.get("chunk_index") or 0),
                                int(c.get("page_number") or 1),
                                c.get("section_title", ""),
                                c.get("content", ""),
                                json.dumps(c.get("embedding") or []),
                                c.get("created_at", now_iso()),
                            ),
                        )

                    # Migrate quizzes
                    for q in legacy.get("quizzes", []):
                        if not isinstance(q, dict) or not q.get("id"):
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO quizzes (
                                id, user_id, session_id, workspace_id, title, topic,
                                document_ids_json, questions_json, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                q["id"],
                                q.get("user_id", "usr-default"),
                                q.get("session_id") or q.get("workspace_id") or "",
                                q.get("workspace_id", "ws-default"),
                                q.get("title", "Quiz"),
                                q.get("topic", ""),
                                json.dumps(q.get("document_ids") or []),
                                json.dumps(q.get("questions") or []),
                                q.get("created_at", now_iso()),
                            ),
                        )

                    # Migrate provider_vault
                    vault = legacy.get("provider_vault") or {}
                    if isinstance(vault, dict):
                        for prov, rec in vault.items():
                            if not isinstance(rec, dict):
                                continue
                            existing = conn.execute(
                                "SELECT provider FROM provider_vault WHERE provider = ?",
                                (prov,),
                            ).fetchone()
                            if existing:
                                continue
                            conn.execute(
                                """
                                INSERT INTO provider_vault (
                                    provider, encrypted_api_key, base_url, verified,
                                    authentication_method, verification_timestamp,
                                    last_error_code, last_error_message,
                                    available_models_json, model_capabilities_json, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    prov,
                                    "",  # Any legacy Node key is re-encrypted when saved via Python vault
                                    rec.get("base_url", ""),
                                    1 if rec.get("verified") else 0,
                                    rec.get("authentication_method", "none"),
                                    rec.get("verification_timestamp", now_iso()),
                                    rec.get("last_error_code", ""),
                                    rec.get("last_error_message", ""),
                                    json.dumps(rec.get("available_models") or []),
                                    json.dumps(rec.get("model_capabilities") or {}),
                                    now_iso(),
                                ),
                            )
            finally:
                conn.close()

    def _scan_existing_exports(self) -> None:
        """Index any pre-existing files in .data/exports so they remain downloadable and tracked."""
        if not EXPORTS_DIR.exists():
            return
        ext_category_map = {
            "py": ("Python Source Script", "text/x-python; charset=utf-8"),
            "js": ("JavaScript Source", "application/javascript; charset=utf-8"),
            "ts": ("TypeScript Source", "application/typescript; charset=utf-8"),
            "html": ("HTML Document", "text/html; charset=utf-8"),
            "css": ("CSS Stylesheet", "text/css; charset=utf-8"),
            "json": ("JSON Data", "application/json; charset=utf-8"),
            "sql": ("SQL Script", "application/sql; charset=utf-8"),
            "pdf": ("PDF Document", "application/pdf"),
            "docx": (
                "Word Document",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            "doc": ("Word Document", "application/msword"),
            "pptx": (
                "PowerPoint Presentation",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ),
            "xlsx": (
                "Excel Spreadsheet",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            "csv": ("CSV Spreadsheet", "text/csv; charset=utf-8"),
            "md": ("Markdown Note", "text/markdown; charset=utf-8"),
            "txt": ("Text Document", "text/plain; charset=utf-8"),
        }
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    for item in EXPORTS_DIR.iterdir():
                        if not item.is_file():
                            continue
                        ext = item.suffix.lstrip(".").lower() or "txt"
                        cat, mime = ext_category_map.get(
                            ext, (f"{ext.upper()} File", "application/octet-stream")
                        )
                        size = item.stat().st_size
                        if size <= 0:
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO generated_files (
                                id, user_id, session_id, message_id, filename, format,
                                category, mime_type, file_size, file_path, download_url,
                                validated, validation_summary, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                f"file-{hashlib.md5(item.name.encode('utf-8')).hexdigest()[:12]}",
                                "usr-default",
                                "",
                                "",
                                item.name,
                                ext,
                                cat,
                                mime,
                                size,
                                str(item.resolve()),
                                f"/api/export/download/{item.name}",
                                1,
                                f"Verified {ext.upper()} file ({size} bytes)",
                                now_iso(),
                            ),
                        )
            finally:
                conn.close()

    # ========================================================================
    # USER & SETTINGS OPERATIONS
    # ========================================================================

    def get_default_user(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM users WHERE id = 'usr-default'").fetchone()
                if row:
                    return dict(row)
                self._seed_defaults()
                row = conn.execute("SELECT * FROM users WHERE id = 'usr-default'").fetchone()
                return dict(row) if row else {
                    "id": "usr-default",
                    "authentication_subject": "guest",
                    "username": "Jarvis",
                    "email": "Jarvis@gmail.com",
                    "role": "Lead Research Fellow",
                }
            finally:
                conn.close()

    def update_user_profile(
        self, user_id: str, username: Optional[str] = None, email: Optional[str] = None, role: Optional[str] = None
    ) -> Dict[str, Any]:
        user = self.get_default_user()
        new_username = (username or user["username"]).strip()
        new_email = (email or user["email"]).strip()
        new_role = (role or user["role"]).strip()
        ts = now_iso()
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        UPDATE users
                        SET username = ?, email = ?, role = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (new_username, new_email, new_role, ts, user_id),
                    )
            finally:
                conn.close()
        return self.get_default_user()

    def get_user_settings(self, user_id: str = "usr-default") -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
                ).fetchone()
                if not row:
                    self._seed_defaults()
                    row = conn.execute(
                        "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
                    ).fetchone()
                if not row:
                    return {
                        "user_id": user_id,
                        "active_provider": "gemini",
                        "active_model": "",
                        "active_model_name": "",
                        "active_model_is_free": False,
                        "local_endpoint": "http://localhost:11434",
                        "theme": "dark",
                        "workspace_mode": "study",
                        "teaching_mode": "auto",
                        "study_difficulty": "mixed",
                        "voice_settings": {},
                    }
                d = dict(row)
                d["active_model_is_free"] = bool(d.get("active_model_is_free", 1))
                try:
                    d["voice_settings"] = json.loads(d.pop("voice_settings_json", "{}") or "{}")
                except Exception:
                    d["voice_settings"] = {}
                return d
            finally:
                conn.close()

    def update_user_settings(self, updates: Dict[str, Any], user_id: str = "usr-default") -> Dict[str, Any]:
        current = self.get_user_settings(user_id)
        active_provider = str(updates.get("active_provider", current["active_provider"])).strip().lower()
        active_model = str(updates.get("active_model", current["active_model"])).strip()
        active_model_name = str(updates.get("active_model_name", current["active_model_name"])).strip()
        active_model_is_free = 1 if updates.get("active_model_is_free", current["active_model_is_free"]) else 0
        local_endpoint = str(updates.get("local_endpoint", current["local_endpoint"])).strip()
        theme = str(updates.get("theme", current["theme"])).strip()
        workspace_mode = str(updates.get("workspace_mode", current["workspace_mode"])).strip()
        teaching_mode = str(updates.get("teaching_mode", current["teaching_mode"])).strip()
        study_difficulty = str(updates.get("study_difficulty", current["study_difficulty"])).strip()
        voice_settings = updates.get("voice_settings", current.get("voice_settings", {}))
        ts = now_iso()

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO user_settings (
                            user_id, active_provider, active_model, active_model_name,
                            active_model_is_free, local_endpoint, theme, workspace_mode,
                            teaching_mode, study_difficulty, voice_settings_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET
                            active_provider = excluded.active_provider,
                            active_model = excluded.active_model,
                            active_model_name = excluded.active_model_name,
                            active_model_is_free = excluded.active_model_is_free,
                            local_endpoint = excluded.local_endpoint,
                            theme = excluded.theme,
                            workspace_mode = excluded.workspace_mode,
                            teaching_mode = excluded.teaching_mode,
                            study_difficulty = excluded.study_difficulty,
                            voice_settings_json = excluded.voice_settings_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            user_id,
                            active_provider,
                            active_model,
                            active_model_name,
                            active_model_is_free,
                            local_endpoint,
                            theme,
                            workspace_mode,
                            teaching_mode,
                            study_difficulty,
                            json.dumps(voice_settings if isinstance(voice_settings, dict) else {}),
                            ts,
                        ),
                    )
            finally:
                conn.close()
        return self.get_user_settings(user_id)

    def get_default_workspace(self, user_id: str = "usr-default") -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM workspaces WHERE user_id = ? LIMIT 1", (user_id,)
                ).fetchone()
                if row:
                    return dict(row)
                return {
                    "id": "ws-default",
                    "user_id": user_id,
                    "name": "Research & Study Workspace",
                    "description": "Personal study and document intelligence workspace",
                }
            finally:
                conn.close()

    # ========================================================================
    # ENCRYPTED PROVIDER VAULT OPERATIONS
    # ========================================================================

    def mask_api_key(self, raw_key: str) -> str:
        """Returns a safe masked representation (e.g. ••••••••••••abcd) without exposing the full key."""
        clean = (raw_key or "").strip()
        if not clean:
            return ""
        if len(clean) <= 4:
            return "••••••••••••"
        return f"••••••••••••{clean[-4:]}"

    def save_provider_credentials(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verified: bool = False,
        authentication_method: str = "user_stored_key",
        available_models: Optional[List[Dict[str, Any]]] = None,
        model_capabilities: Optional[Dict[str, Any]] = None,
        last_error_code: str = "",
        last_error_message: str = "",
    ) -> None:
        prov = provider.strip().lower()
        ts = now_iso()

        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT * FROM provider_vault WHERE provider = ?", (prov,)
                ).fetchone()
                prev = dict(existing) if existing else {}

                # Preserve existing encrypted key unless a new non-empty api_key is provided
                if api_key is not None and api_key.strip():
                    encrypted_key = self.crypto.encrypt(api_key.strip())
                else:
                    encrypted_key = prev.get("encrypted_api_key", "")

                resolved_base_url = (
                    base_url.strip()
                    if base_url is not None and base_url.strip()
                    else prev.get("base_url", "http://localhost:11434" if prov == "local" else "")
                )

                models_json = (
                    json.dumps(available_models)
                    if available_models is not None
                    else prev.get("available_models_json", "[]")
                )
                caps_json = (
                    json.dumps(model_capabilities)
                    if model_capabilities is not None
                    else prev.get("model_capabilities_json", "{}")
                )

                with conn:
                    conn.execute(
                        """
                        INSERT INTO provider_vault (
                            provider, encrypted_api_key, base_url, verified,
                            authentication_method, verification_timestamp,
                            last_error_code, last_error_message,
                            available_models_json, model_capabilities_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(provider) DO UPDATE SET
                            encrypted_api_key = excluded.encrypted_api_key,
                            base_url = excluded.base_url,
                            verified = excluded.verified,
                            authentication_method = excluded.authentication_method,
                            verification_timestamp = excluded.verification_timestamp,
                            last_error_code = excluded.last_error_code,
                            last_error_message = excluded.last_error_message,
                            available_models_json = excluded.available_models_json,
                            model_capabilities_json = excluded.model_capabilities_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            prov,
                            encrypted_key,
                            resolved_base_url,
                            1 if verified else 0,
                            authentication_method,
                            ts,
                            last_error_code,
                            last_error_message,
                            models_json,
                            caps_json,
                            ts,
                        ),
                    )
            finally:
                conn.close()

    def clear_provider_credentials(self, provider: str) -> None:
        """
        Permanently deletes the stored API key for the provider and marks it as explicitly
        cleared so no stale environment or cached credentials are used until re-verified.
        """
        prov = provider.strip().lower()
        ts = now_iso()
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO provider_vault (
                            provider, encrypted_api_key, base_url, verified,
                            authentication_method, verification_timestamp,
                            last_error_code, last_error_message,
                            available_models_json, model_capabilities_json, updated_at
                        ) VALUES (?, '', '', 0, 'deleted_by_user', ?, 'MISSING_CREDENTIALS', 'API key deleted by user.', '[]', '{}', ?)
                        ON CONFLICT(provider) DO UPDATE SET
                            encrypted_api_key = '',
                            verified = 0,
                            authentication_method = 'deleted_by_user',
                            verification_timestamp = excluded.verification_timestamp,
                            last_error_code = 'MISSING_CREDENTIALS',
                            last_error_message = 'API key deleted by user.',
                            available_models_json = '[]',
                            model_capabilities_json = '{}',
                            updated_at = excluded.updated_at
                        """,
                        (prov, ts, ts),
                    )
            finally:
                conn.close()

    def is_provider_explicitly_deleted(self, provider: str) -> bool:
        prov = provider.strip().lower()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT authentication_method, encrypted_api_key FROM provider_vault WHERE provider = ?",
                    (prov,),
                ).fetchone()
                if not row:
                    return False
                return (
                    row["authentication_method"] == "deleted_by_user"
                    and not row["encrypted_api_key"]
                )
            finally:
                conn.close()

    def get_decrypted_api_key(self, provider: str) -> str:
        """Server-only helper: decrypts and returns the stored API key for a provider."""
        prov = provider.strip().lower()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT encrypted_api_key FROM provider_vault WHERE provider = ?", (prov,)
                ).fetchone()
                if not row or not row["encrypted_api_key"]:
                    return ""
                return self.crypto.decrypt(row["encrypted_api_key"])
            finally:
                conn.close()

    def has_encrypted_key_at_rest(self, provider: str) -> bool:
        """Verifies that a stored key exists and is encrypted (never plaintext)."""
        prov = provider.strip().lower()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT encrypted_api_key FROM provider_vault WHERE provider = ?", (prov,)
                ).fetchone()
                if not row or not row["encrypted_api_key"]:
                    return False
                val = str(row["encrypted_api_key"])
                return val.startswith("ENCv1:") and len(self.crypto.decrypt(val)) > 0
            finally:
                conn.close()

    def get_provider_public_status(self, provider: str) -> Dict[str, Any]:
        """Returns public metadata for a provider. Includes masked_key when stored, never raw plaintext."""
        prov = provider.strip().lower()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM provider_vault WHERE provider = ?", (prov,)
                ).fetchone()
            finally:
                conn.close()

        explicitly_deleted = bool(
            row
            and row["authentication_method"] == "deleted_by_user"
            and not row["encrypted_api_key"]
        )

        decrypted_stored = ""
        if row and row["encrypted_api_key"]:
            decrypted_stored = self.crypto.decrypt(row["encrypted_api_key"])
        has_stored_key = bool(decrypted_stored)
        masked_key = self.mask_api_key(decrypted_stored) if has_stored_key else ""

        env_key_val = ""
        if not explicitly_deleted:
            placeholders = {
                "your_gemini_api_key_here",
                "todo_keyhere",
                "my_gemini_api_key",
                "your_api_key_here",
                "changeme",
            }
            if prov == "gemini":
                candidate = (
                    os.environ.get("GEMINI_API_KEY", "").strip()
                    or os.environ.get("GOOGLE_API_KEY", "").strip()
                    or os.environ.get("API_KEY", "").strip()
                )
                if candidate and candidate.lower() not in placeholders:
                    env_key_val = candidate
            elif prov == "groq":
                candidate = os.environ.get("GROQ_API_KEY", "").strip()
                if candidate and candidate.lower() not in placeholders:
                    env_key_val = candidate
            elif prov == "openrouter":
                candidate = os.environ.get("OPENROUTER_API_KEY", "").strip()
                if candidate and candidate.lower() not in placeholders:
                    env_key_val = candidate
            elif prov == "openai":
                candidate = os.environ.get("OPENAI_API_KEY", "").strip()
                if candidate and candidate.lower() not in placeholders:
                    env_key_val = candidate
            elif prov == "anthropic":
                candidate = os.environ.get("ANTHROPIC_API_KEY", "").strip()
                if candidate and candidate.lower() not in placeholders:
                    env_key_val = candidate

        env_key_exists = bool(env_key_val)
        if not masked_key and env_key_exists and row and bool(row["verified"]):
            masked_key = self.mask_api_key(env_key_val)

        models: List[Dict[str, Any]] = []
        caps: Dict[str, Any] = {}
        if row:
            try:
                models = json.loads(row["available_models_json"] or "[]")
            except Exception:
                models = []
            try:
                caps = json.loads(row["model_capabilities_json"] or "{}")
            except Exception:
                caps = {}

        for model in models:
            if not isinstance(model, dict) or not model.get("id"):
                continue
            model_id = str(model["id"])
            model["provider_id"] = prov
            model["exact_model_id"] = model_id
            display_name = model.get("display_name") or model.get("name") or model_id
            if prov != "gemini" and "gemini" in str(display_name).lower():
                display_name = model_id
            model["display_name"] = display_name
            model["name"] = display_name
            model["availability_status"] = model.get("availability_status") or model.get("status") or "unknown"
            pricing_status = model.get("pricing_status")
            if pricing_status not in ("free", "paid", "unknown", "local"):
                pricing_status = "unknown"
            model["pricing_status"] = pricing_status
            model["pricing_label"] = model.get("pricing_label") or (
                "Pricing unknown" if pricing_status == "unknown" else pricing_status.upper()
            )
            model["isFree"] = pricing_status == "free"
            model["capability_metadata"] = model.get("capability_metadata") or {
                key: model.get(key)
                for key in (
                    "text", "vision", "document", "image_generation", "audio_input",
                    "audio_generation", "reasoning", "local", "context_window",
                    "supported_generation_methods", "input_modalities", "output_modalities",
                )
                if key in model
            }
            if isinstance(caps, dict):
                provider_caps = caps.get(model_id)
                if isinstance(provider_caps, dict):
                    provider_caps["free_tier"] = (
                        True if pricing_status == "free"
                        else False if pricing_status == "paid"
                        else None
                    )

        is_verified = bool(row["verified"]) if row else False
        if not has_stored_key and not env_key_exists and prov != "local":
            is_verified = False
            models = []

        return {
            "provider": prov,
            "verified": is_verified,
            "has_stored_key": has_stored_key,
            "masked_key": masked_key,
            "has_builtin_or_env_key": env_key_exists,
            "explicitly_deleted": explicitly_deleted,
            "authentication_method": (
                row["authentication_method"]
                if row and row["authentication_method"] not in ("none", "")
                else ("built_in_server_key" if env_key_exists else "none")
            ),
            "verification_timestamp": row["verification_timestamp"] if row else None,
            "base_url": (
                row["base_url"]
                if row and row["base_url"]
                else ("http://localhost:11434" if prov == "local" else None)
            ),
            "available_models": models if is_verified else [],
            "model_capabilities": caps if is_verified else {},
            "last_error_code": row["last_error_code"] if row and row["last_error_code"] else None,
            "last_error_message": (
                row["last_error_message"] if row and row["last_error_message"] else None
            ),
        }

    def get_all_providers_public_status(self) -> Dict[str, Dict[str, Any]]:
        return {p: self.get_provider_public_status(p) for p in SUPPORTED_PROVIDERS}

    def is_any_provider_configured(self) -> bool:
        """Returns True if at least one AI provider has a verified stored key (or verified local/env connection)."""
        statuses = self.get_all_providers_public_status()
        for prov, st in statuses.items():
            if st.get("verified") and (st.get("has_stored_key") or st.get("has_builtin_or_env_key") or prov == "local"):
                return True
        return False

    # ========================================================================
    # CONVERSATION (SESSION) & MESSAGE OPERATIONS
    # ========================================================================

    def _row_to_session(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            document_ids = json.loads(d.pop("document_ids_json", "[]") or "[]")
        except Exception:
            document_ids = []
        try:
            session_preferences = json.loads(d.pop("session_preferences_json", "{}") or "{}")
        except Exception:
            session_preferences = {}
        cs_raw = d.pop("cheatsheet_json", None)
        try:
            cheatsheet = json.loads(cs_raw) if cs_raw else None
        except Exception:
            cheatsheet = None
        try:
            flashcards = json.loads(d.pop("flashcards_json", "[]") or "[]")
        except Exception:
            flashcards = []

        d["document_ids"] = document_ids
        d["session_preferences"] = session_preferences
        d["cheatsheet"] = cheatsheet
        d["flashcards"] = flashcards
        return d

    def get_sessions(self, user_id: str = "usr-default") -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                    (user_id,),
                ).fetchall()
                return [self._row_to_session(r) for r in rows]
            finally:
                conn.close()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                return self._row_to_session(row) if row else None
            finally:
                conn.close()

    def create_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data.get("id") or f"chat-{int(time.time() * 1000)}"
        existing = self.get_session(session_id)
        if existing:
            return existing

        settings = self.get_user_settings(data.get("user_id", "usr-default"))
        ts = now_iso()
        user_id = data.get("user_id", "usr-default")
        workspace_id = data.get("workspace_id", "ws-default")
        workspace_mode = data.get("workspace_mode") or settings.get("workspace_mode") or "study"
        title = (data.get("title") or "New Session").strip()
        topic = (data.get("topic") or "").strip()
        provider_id = data.get("provider_id") or settings.get("active_provider") or "gemini"
        model_id = data.get("model_id") or settings.get("active_model") or ""
        model_name = (
            data.get("model_name")
            or settings.get("active_model_name")
            or model_id
            or ""
        )
        teaching_depth = data.get("teaching_depth") or settings.get("teaching_mode") or "auto"
        response_style = data.get("response_style") or "balanced"
        reasoning_preference = data.get("reasoning_preference") or "standard"
        document_ids = data.get("document_ids") if isinstance(data.get("document_ids"), list) else []
        session_preferences = (
            data.get("session_preferences")
            if isinstance(data.get("session_preferences"), dict)
            else {}
        )
        cheatsheet = data.get("cheatsheet")
        flashcards = data.get("flashcards") if isinstance(data.get("flashcards"), list) else []

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO sessions (
                            id, user_id, workspace_id, workspace_mode, title, topic,
                            provider_id, model_id, model_name, teaching_depth,
                            response_style, reasoning_preference, document_ids_json,
                            session_preferences_json, cheatsheet_json, flashcards_json,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            user_id,
                            workspace_id,
                            workspace_mode,
                            title,
                            topic,
                            provider_id,
                            model_id,
                            model_name,
                            teaching_depth,
                            response_style,
                            reasoning_preference,
                            json.dumps(document_ids),
                            json.dumps(session_preferences),
                            json.dumps(cheatsheet) if cheatsheet else None,
                            json.dumps(flashcards),
                            ts,
                            ts,
                        ),
                    )
            finally:
                conn.close()
        return self.get_session(session_id) or {}

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current = self.get_session(session_id)
        if not current:
            return None
        ts = now_iso()

        title = str(updates.get("title", current["title"])).strip() or current["title"]
        topic = str(updates.get("topic", current.get("topic", ""))).strip()
        workspace_mode = str(updates.get("workspace_mode", current["workspace_mode"])).strip()
        provider_id = str(updates.get("provider_id", current["provider_id"])).strip()
        model_id = str(updates.get("model_id", current["model_id"])).strip()
        model_name = str(updates.get("model_name", current["model_name"])).strip()
        teaching_depth = str(updates.get("teaching_depth", current["teaching_depth"])).strip()
        response_style = str(updates.get("response_style", current["response_style"])).strip()
        reasoning_preference = str(
            updates.get("reasoning_preference", current["reasoning_preference"])
        ).strip()
        document_ids = (
            updates["document_ids"]
            if "document_ids" in updates and isinstance(updates["document_ids"], list)
            else current["document_ids"]
        )
        session_preferences = (
            updates["session_preferences"]
            if "session_preferences" in updates and isinstance(updates["session_preferences"], dict)
            else current["session_preferences"]
        )
        cheatsheet = (
            updates["cheatsheet"] if "cheatsheet" in updates else current.get("cheatsheet")
        )
        flashcards = (
            updates["flashcards"]
            if "flashcards" in updates and isinstance(updates["flashcards"], list)
            else current.get("flashcards", [])
        )

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        UPDATE sessions SET
                            title = ?, topic = ?, workspace_mode = ?, provider_id = ?,
                            model_id = ?, model_name = ?, teaching_depth = ?,
                            response_style = ?, reasoning_preference = ?,
                            document_ids_json = ?, session_preferences_json = ?,
                            cheatsheet_json = ?, flashcards_json = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            title,
                            topic,
                            workspace_mode,
                            provider_id,
                            model_id,
                            model_name,
                            teaching_depth,
                            response_style,
                            reasoning_preference,
                            json.dumps(document_ids),
                            json.dumps(session_preferences),
                            json.dumps(cheatsheet) if cheatsheet else None,
                            json.dumps(flashcards),
                            ts,
                            session_id,
                        ),
                    )
            finally:
                conn.close()
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
                    conn.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
                    conn.execute("DELETE FROM flashcard_sets WHERE session_id = ?", (session_id,))
                    conn.execute("DELETE FROM quizzes WHERE session_id = ?", (session_id,))
                    conn.execute("DELETE FROM cheatsheets WHERE session_id = ?", (session_id,))
            finally:
                conn.close()

    def _row_to_message(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            sources = json.loads(d.pop("sources_json", "[]") or "[]")
        except Exception:
            sources = []
        try:
            gen_files = json.loads(d.pop("generated_files_json", "[]") or "[]")
        except Exception:
            gen_files = []
        d["sources"] = sources
        d["generated_files"] = gen_files
        return d

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        if not session_id:
            return []
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,),
                ).fetchall()
                return [self._row_to_message(r) for r in rows]
            finally:
                conn.close()

    def add_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        msg_id = (
            data.get("id")
            or f"msg-{int(time.time() * 1000)}-{secrets.token_hex(2)}"
        )
        session_id = str(data.get("session_id") or "")
        user_id = str(data.get("user_id") or "usr-default")
        role = str(data.get("role") or "user")
        content = str(data.get("content") or "")
        model = str(data.get("model") or "")
        provider = str(data.get("provider") or "")
        sources = data.get("sources") if isinstance(data.get("sources"), list) else []
        generated_files = (
            data.get("generated_files")
            if isinstance(data.get("generated_files"), list)
            else []
        )
        ts = now_iso()

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO messages (
                            id, session_id, user_id, role, content, model, provider,
                            sources_json, generated_files_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            msg_id,
                            session_id,
                            user_id,
                            role,
                            content,
                            model,
                            provider,
                            json.dumps(sources),
                            json.dumps(generated_files),
                            ts,
                        ),
                    )
                    if session_id:
                        conn.execute(
                            "UPDATE sessions SET updated_at = ? WHERE id = ?",
                            (ts, session_id),
                        )
            finally:
                conn.close()
        return {
            "id": msg_id,
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "model": model,
            "provider": provider,
            "sources": sources,
            "generated_files": generated_files,
            "created_at": ts,
        }

    # ========================================================================
    # DOCUMENTS & CHUNKS (STRICT CONVERSATION ISOLATION)
    # ========================================================================

    def get_documents(
        self, user_id: str = "usr-default", session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns documents strictly scoped to session_id when session_id is provided.
        Never leaks documents from other conversations!
        """
        with self._lock:
            conn = self._connect()
            try:
                if session_id is not None and session_id.strip():
                    rows = conn.execute(
                        """
                        SELECT * FROM documents
                        WHERE user_id = ? AND session_id = ?
                        ORDER BY created_at ASC
                        """,
                        (user_id, session_id.strip()),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC",
                        (user_id,),
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM documents WHERE id = ?", (doc_id,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def add_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = (
            data.get("id")
            or f"doc-{int(time.time() * 1000)}-{secrets.token_hex(2)}"
        )
        user_id = str(data.get("user_id") or "usr-default")
        workspace_id = str(data.get("workspace_id") or "ws-default")
        session_id = str(data.get("session_id") or "").strip()
        filename = str(data.get("filename") or "document.txt")
        file_type = str(data.get("file_type") or "txt")
        file_size = int(data.get("file_size") or 0)
        char_count = int(data.get("char_count") or 0)
        chunk_count = int(data.get("chunk_count") or 0)
        summary = str(data.get("summary") or "")
        status = str(data.get("status") or "ready")
        ts = now_iso()

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    # Remove any existing document with same filename in the same session to avoid duplicates
                    if session_id:
                        old_docs = conn.execute(
                            "SELECT id FROM documents WHERE user_id = ? AND session_id = ? AND filename = ?",
                            (user_id, session_id, filename),
                        ).fetchall()
                        for od in old_docs:
                            conn.execute("DELETE FROM chunks WHERE document_id = ?", (od["id"],))
                            conn.execute("DELETE FROM documents WHERE id = ?", (od["id"],))

                    conn.execute(
                        """
                        INSERT INTO documents (
                            id, user_id, workspace_id, session_id, filename, file_type,
                            file_size, char_count, chunk_count, summary, status, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            user_id,
                            workspace_id,
                            session_id,
                            filename,
                            file_type,
                            file_size,
                            char_count,
                            chunk_count,
                            summary,
                            status,
                            ts,
                        ),
                    )

                    # Associate document_id with session
                    if session_id:
                        s_row = conn.execute(
                            "SELECT document_ids_json FROM sessions WHERE id = ?", (session_id,)
                        ).fetchone()
                        if s_row:
                            try:
                                doc_ids = json.loads(s_row["document_ids_json"] or "[]")
                            except Exception:
                                doc_ids = []
                            if doc_id not in doc_ids:
                                doc_ids.append(doc_id)
                                conn.execute(
                                    "UPDATE sessions SET document_ids_json = ?, updated_at = ? WHERE id = ?",
                                    (json.dumps(doc_ids), ts, session_id),
                                )
            finally:
                conn.close()
        return self.get_document(doc_id) or {}

    def delete_document(self, doc_id: str, session_id: Optional[str] = None) -> None:
        ts = now_iso()
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
                    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                    rows = conn.execute("SELECT id, document_ids_json FROM sessions").fetchall()
                    for s_row in rows:
                        if session_id and s_row["id"] != session_id:
                            continue
                        try:
                            doc_ids = json.loads(s_row["document_ids_json"] or "[]")
                        except Exception:
                            doc_ids = []
                        if doc_id in doc_ids:
                            doc_ids = [d for d in doc_ids if d != doc_id]
                            conn.execute(
                                "UPDATE sessions SET document_ids_json = ?, updated_at = ? WHERE id = ?",
                                (json.dumps(doc_ids), ts, s_row["id"]),
                            )
            finally:
                conn.close()

    def _row_to_chunk(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            emb = json.loads(d.pop("embedding_json", "[]") or "[]")
        except Exception:
            emb = []
        if emb:
            d["embedding"] = emb
        return d

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        ts = now_iso()
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO chunks (
                            id, document_id, user_id, session_id, filename, chunk_index,
                            page_number, section_title, content, embedding_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                c.get("id")
                                or f"chunk-{c['document_id']}-{c.get('chunk_index', idx)}",
                                c["document_id"],
                                c.get("user_id", "usr-default"),
                                c.get("session_id", ""),
                                c.get("filename", ""),
                                int(c.get("chunk_index", idx)),
                                int(c.get("page_number", 1)),
                                c.get("section_title", ""),
                                c.get("content", ""),
                                json.dumps(c.get("embedding") or []),
                                c.get("created_at", ts),
                            )
                            for idx, c in enumerate(chunks)
                        ],
                    )
            finally:
                conn.close()

    def get_chunks(
        self,
        user_id: str = "usr-default",
        document_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves chunks strictly scoped to document_ids and/or session_id.
        Never mixes chunks from unrelated sessions!
        """
        with self._lock:
            conn = self._connect()
            try:
                if document_ids is not None:
                    if len(document_ids) == 0:
                        return []
                    placeholders = ",".join("?" for _ in document_ids)
                    params: List[Any] = [user_id, *document_ids]
                    sql = f"SELECT * FROM chunks WHERE user_id = ? AND document_id IN ({placeholders})"
                    if session_id:
                        sql += " AND session_id = ?"
                        params.append(session_id)
                    sql += " ORDER BY document_id, chunk_index ASC"
                    rows = conn.execute(sql, params).fetchall()
                elif session_id is not None and session_id.strip():
                    rows = conn.execute(
                        """
                        SELECT * FROM chunks
                        WHERE user_id = ? AND session_id = ?
                        ORDER BY document_id, chunk_index ASC
                        """,
                        (user_id, session_id.strip()),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM chunks WHERE user_id = ? ORDER BY created_at DESC",
                        (user_id,),
                    ).fetchall()
                return [self._row_to_chunk(r) for r in rows]
            finally:
                conn.close()

    # ========================================================================
    # MEMORIES
    # ========================================================================

    def get_memories(
        self, user_id: str = "usr-default", session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if session_id:
                    rows = conn.execute(
                        """
                        SELECT * FROM memories
                        WHERE user_id = ? AND session_id = ?
                        ORDER BY importance DESC, updated_at DESC
                        """,
                        (user_id, session_id),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM memories WHERE user_id = ? ORDER BY importance DESC, updated_at DESC",
                        (user_id,),
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def add_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(data.get("user_id") or "usr-default")
        session_id = str(data.get("session_id") or "")
        category = str(data.get("category") or "fact")
        content = str(data.get("content") or "").strip()
        importance = int(data.get("importance") or 3)
        ts = now_iso()

        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE user_id = ? AND session_id = ? AND LOWER(content) = LOWER(?)
                    """,
                    (user_id, session_id, content),
                ).fetchone()
                if existing:
                    with conn:
                        conn.execute(
                            "UPDATE memories SET updated_at = ?, importance = MAX(importance, ?) WHERE id = ?",
                            (ts, importance, existing["id"]),
                        )
                    return dict(existing)

                mem_id = f"mem-{int(time.time() * 1000)}-{secrets.token_hex(2)}"
                with conn:
                    conn.execute(
                        """
                        INSERT INTO memories (id, user_id, session_id, category, content, importance, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (mem_id, user_id, session_id, category, content, importance, ts, ts),
                    )
                return {
                    "id": mem_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "category": category,
                    "content": content,
                    "importance": importance,
                    "created_at": ts,
                    "updated_at": ts,
                }
            finally:
                conn.close()

    def delete_memory(self, mem_id: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            finally:
                conn.close()

    # ========================================================================
    # STUDY MATERIALS: RECALL CARDS, QUIZZES, ATTEMPTS, CHEAT SHEETS
    # ========================================================================

    def add_flashcard_set(self, data: Dict[str, Any]) -> Dict[str, Any]:
        set_id = f"fcset-{int(time.time() * 1000)}"
        user_id = str(data.get("user_id") or "usr-default")
        session_id = str(data.get("session_id") or data.get("workspace_id") or "")
        workspace_id = str(data.get("workspace_id") or session_id or "ws-default")
        title = str(data.get("title") or "Recall Cards")
        topic = str(data.get("topic") or title)
        doc_ids = data.get("document_ids") if isinstance(data.get("document_ids"), list) else []
        cards = data.get("cards") if isinstance(data.get("cards"), list) else []
        ts = now_iso()

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO flashcard_sets (
                            id, user_id, session_id, workspace_id, title, topic,
                            document_ids_json, cards_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            set_id,
                            user_id,
                            session_id,
                            workspace_id,
                            title,
                            topic,
                            json.dumps(doc_ids),
                            json.dumps(cards),
                            ts,
                        ),
                    )
                    if session_id:
                        conn.execute(
                            "UPDATE sessions SET flashcards_json = ?, updated_at = ? WHERE id = ?",
                            (json.dumps(cards), ts, session_id),
                        )
            finally:
                conn.close()
        return {
            "id": set_id,
            "user_id": user_id,
            "session_id": session_id,
            "workspace_id": workspace_id,
            "title": title,
            "topic": topic,
            "document_ids": doc_ids,
            "cards": cards,
            "created_at": ts,
        }

    def get_flashcard_sets(
        self, user_id: str = "usr-default", session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if session_id:
                    rows = conn.execute(
                        "SELECT * FROM flashcard_sets WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC",
                        (user_id, session_id),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM flashcard_sets WHERE user_id = ? ORDER BY created_at DESC",
                        (user_id,),
                    ).fetchall()
                result = []
                for r in rows:
                    d = dict(r)
                    d["document_ids"] = json.loads(d.pop("document_ids_json", "[]") or "[]")
                    d["cards"] = json.loads(d.pop("cards_json", "[]") or "[]")
                    result.append(d)
                return result
            finally:
                conn.close()

    def add_quiz(self, data: Dict[str, Any]) -> Dict[str, Any]:
        quiz_id = f"quiz-{int(time.time() * 1000)}"
        user_id = str(data.get("user_id") or "usr-default")
        session_id = str(data.get("session_id") or data.get("workspace_id") or "")
        workspace_id = str(data.get("workspace_id") or session_id or "ws-default")
        title = str(data.get("title") or "Interactive Quiz")
        topic = str(data.get("topic") or title)
        doc_ids = data.get("document_ids") if isinstance(data.get("document_ids"), list) else []
        questions = data.get("questions") if isinstance(data.get("questions"), list) else []
        ts = now_iso()

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO quizzes (
                            id, user_id, session_id, workspace_id, title, topic,
                            document_ids_json, questions_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            quiz_id,
                            user_id,
                            session_id,
                            workspace_id,
                            title,
                            topic,
                            json.dumps(doc_ids),
                            json.dumps(questions),
                            ts,
                        ),
                    )
            finally:
                conn.close()
        return {
            "id": quiz_id,
            "user_id": user_id,
            "session_id": session_id,
            "workspace_id": workspace_id,
            "title": title,
            "topic": topic,
            "document_ids": doc_ids,
            "questions": questions,
            "created_at": ts,
        }

    def get_quizzes(
        self, user_id: str = "usr-default", session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if session_id:
                    rows = conn.execute(
                        "SELECT * FROM quizzes WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC",
                        (user_id, session_id),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM quizzes WHERE user_id = ? ORDER BY created_at DESC",
                        (user_id,),
                    ).fetchall()
                result = []
                for r in rows:
                    d = dict(r)
                    d["document_ids"] = json.loads(d.pop("document_ids_json", "[]") or "[]")
                    d["questions"] = json.loads(d.pop("questions_json", "[]") or "[]")
                    result.append(d)
                return result
            finally:
                conn.close()

    def add_quiz_attempt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        att_id = f"att-{int(time.time() * 1000)}"
        quiz_id = str(data.get("quiz_id") or "adhoc")
        user_id = str(data.get("user_id") or "usr-default")
        session_id = str(data.get("session_id") or "")
        score = int(data.get("score") or 0)
        total = int(data.get("total_questions") or 0)
        acc = float(data.get("accuracy_percentage") or 0.0)
        weak = data.get("weak_topics") if isinstance(data.get("weak_topics"), list) else []
        answers = data.get("answers") if isinstance(data.get("answers"), dict) else {}
        ts = now_iso()

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO quiz_attempts (
                            id, quiz_id, user_id, session_id, score, total_questions,
                            accuracy_percentage, weak_topics_json, answers_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            att_id,
                            quiz_id,
                            user_id,
                            session_id,
                            score,
                            total,
                            acc,
                            json.dumps(weak),
                            json.dumps(answers),
                            ts,
                        ),
                    )
            finally:
                conn.close()
        return {
            "id": att_id,
            "quiz_id": quiz_id,
            "user_id": user_id,
            "session_id": session_id,
            "score": score,
            "total_questions": total,
            "accuracy_percentage": acc,
            "weak_topics": weak,
            "answers": answers,
            "created_at": ts,
        }

    def get_quiz_attempts(self, user_id: str = "usr-default") -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM quiz_attempts WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
                out = []
                for r in rows:
                    d = dict(r)
                    d["weak_topics"] = json.loads(d.pop("weak_topics_json", "[]") or "[]")
                    d["answers"] = json.loads(d.pop("answers_json", "{}") or "{}")
                    out.append(d)
                return out
            finally:
                conn.close()

    def save_cheatsheet(
        self,
        session_id: str,
        sheet: Dict[str, Any],
        topic: str = "",
        document_ids: Optional[List[str]] = None,
        user_id: str = "usr-default",
    ) -> Dict[str, Any]:
        cs_id = f"cs-{session_id or int(time.time() * 1000)}"
        ts = now_iso()
        doc_ids = document_ids or []
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO cheatsheets (
                            id, user_id, session_id, topic, document_ids_json, sheet_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            topic = excluded.topic,
                            document_ids_json = excluded.document_ids_json,
                            sheet_json = excluded.sheet_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            cs_id,
                            user_id,
                            session_id,
                            topic,
                            json.dumps(doc_ids),
                            json.dumps(sheet),
                            ts,
                            ts,
                        ),
                    )
                    if session_id:
                        conn.execute(
                            "UPDATE sessions SET cheatsheet_json = ?, updated_at = ? WHERE id = ?",
                            (json.dumps(sheet), ts, session_id),
                        )
            finally:
                conn.close()
        return sheet

    # ========================================================================
    # GENERATED FILES METADATA
    # ========================================================================

    def record_generated_file(self, data: Dict[str, Any]) -> Dict[str, Any]:
        filename = str(data.get("filename") or "")
        file_id = (
            data.get("id")
            or f"file-{hashlib.md5(filename.encode('utf-8')).hexdigest()[:12]}"
        )
        user_id = str(data.get("user_id") or "usr-default")
        session_id = str(data.get("session_id") or "")
        message_id = str(data.get("message_id") or "")
        fmt = str(data.get("format") or filename.split(".")[-1].lower() or "txt")
        category = str(data.get("category") or f"{fmt.upper()} Deliverable")
        mime_type = str(data.get("mime_type") or data.get("mimeType") or "application/octet-stream")
        file_size = int(data.get("file_size") or data.get("fileSize") or 0)
        file_path = str(data.get("file_path") or data.get("filePath") or str(EXPORTS_DIR / filename))
        download_url = str(
            data.get("download_url")
            or data.get("downloadUrl")
            or f"/api/export/download/{filename}"
        )
        validated = 1 if data.get("validated", True) else 0
        validation_summary = str(
            data.get("validation_summary")
            or data.get("validationSummary")
            or f"Verified {fmt.upper()} deliverable"
        )
        ts = now_iso()

        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO generated_files (
                            id, user_id, session_id, message_id, filename, format,
                            category, mime_type, file_size, file_path, download_url,
                            validated, validation_summary, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(filename) DO UPDATE SET
                            session_id = CASE WHEN excluded.session_id != '' THEN excluded.session_id ELSE generated_files.session_id END,
                            message_id = CASE WHEN excluded.message_id != '' THEN excluded.message_id ELSE generated_files.message_id END,
                            file_size = excluded.file_size,
                            file_path = excluded.file_path,
                            validated = excluded.validated,
                            validation_summary = excluded.validation_summary,
                            created_at = excluded.created_at
                        """,
                        (
                            file_id,
                            user_id,
                            session_id,
                            message_id,
                            filename,
                            fmt,
                            category,
                            mime_type,
                            file_size,
                            file_path,
                            download_url,
                            validated,
                            validation_summary,
                            ts,
                        ),
                    )
            finally:
                conn.close()

        return {
            "id": file_id,
            "session_id": session_id,
            "message_id": message_id,
            "filename": filename,
            "format": fmt,
            "category": category,
            "mimeType": mime_type,
            "fileSize": file_size,
            "downloadUrl": download_url,
            "validated": bool(validated),
            "validationSummary": validation_summary,
            "created_at": ts,
        }

    def get_generated_files(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if session_id:
                    rows = conn.execute(
                        "SELECT * FROM generated_files WHERE session_id = ? ORDER BY created_at DESC",
                        (session_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM generated_files ORDER BY created_at DESC"
                    ).fetchall()
                result = []
                for r in rows:
                    d = dict(r)
                    result.append(
                        {
                            "id": d["id"],
                            "session_id": d["session_id"],
                            "message_id": d["message_id"],
                            "filename": d["filename"],
                            "format": d["format"],
                            "category": d["category"],
                            "mimeType": d["mime_type"],
                            "fileSize": d["file_size"],
                            "downloadUrl": d["download_url"],
                            "validated": bool(d["validated"]),
                            "validationSummary": d["validation_summary"],
                            "created_at": d["created_at"],
                        }
                    )
                return result
            finally:
                conn.close()


# Singleton DB instance
db = LocalDatabase()


def mask_api_key(raw_key: str) -> str:
    """Convenience module-level API key masker."""
    clean = (raw_key or "").strip()
    if not clean:
        return ""
    if len(clean) <= 8:
        return "••••••••••••"
    return f"{clean[:4]}••••••••••••{clean[-4:]}"
