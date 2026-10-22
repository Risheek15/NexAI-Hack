"""
Regression tests for Section 9: Backend Authentication & Authorization.
Verifies salted PBKDF2 password hashing, legacy hash migration,
cryptographic session tokens, and server-enforced role governance.
"""

import hashlib
import pytest
from fastapi.testclient import TestClient
from src.db.database import (
    DatabaseManager,
    _hash_password,
    _verify_password,
    create_session_token,
    verify_session_token
)
from src.api.main import app


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_auth.db"
    return DatabaseManager(db_path=str(db_file))


class TestPasswordSecurityAndMigration:
    """Tests for PBKDF2 hashing with salt and seamless legacy hash upgrade."""

    def test_pbkdf2_hash_format_and_salt(self):
        pwd = "securepassword123"
        h1 = _hash_password(pwd)
        h2 = _hash_password(pwd)

        assert h1.startswith("pbkdf2:sha256:100000:")
        assert h2.startswith("pbkdf2:sha256:100000:")
        # Different random salts produce different hashes
        assert h1 != h2

        is_valid, needs_upgrade = _verify_password(h1, pwd)
        assert is_valid is True
        assert needs_upgrade is False

    def test_legacy_sha256_verification_and_upgrade(self, test_db):
        pwd = "mypassword"
        legacy_hash = hashlib.sha256(pwd.encode("utf-8")).hexdigest()

        # Direct legacy verification
        is_valid, needs_upgrade = _verify_password(legacy_hash, pwd)
        assert is_valid is True
        assert needs_upgrade is True

        # Insert user with legacy hash into DB
        with test_db._get_connection() as conn:
            conn.execute("""
                INSERT INTO users (username, password_hash, full_name, role, created_at)
                VALUES ('legacyuser', ?, 'Legacy Technician', 'technician', '2026-01-01');
            """, (legacy_hash,))
            conn.commit()

        # First login succeeds and automatically upgrades hash in DB
        user = test_db.authenticate_user("legacyuser", pwd)
        assert user is not None
        assert user["username"] == "legacyuser"

        # Check that hash in DB is now salted PBKDF2
        with test_db._get_connection() as conn:
            row = conn.execute("SELECT password_hash FROM users WHERE username = 'legacyuser'").fetchone()
            assert row["password_hash"].startswith("pbkdf2:sha256:100000:")


class TestSessionTokensAndRoleGovernance:
    """Tests for token authentication and API endpoint authorization."""

    def test_session_token_lifecycle(self):
        token = create_session_token(user_id=1, username="admin", role="admin")
        assert token is not None

        decoded = verify_session_token(token)
        assert decoded is not None
        assert decoded["id"] == 1
        assert decoded["username"] == "admin"
        assert decoded["role"] == "admin"

        # Tampered token fails verification
        tampered = token[:-4] + "ffff"
        assert verify_session_token(tampered) is None

    def test_unauthenticated_admin_access_rejected(self):
        client = TestClient(app)
        res = client.get("/api/admin/users")
        assert res.status_code == 401

    def test_technician_cannot_access_admin_endpoints(self):
        client = TestClient(app)
        tech_token = create_session_token(user_id=2, username="technician1", role="technician")
        headers = {"Authorization": f"Bearer {tech_token}"}

        res = client.get("/api/admin/users", headers=headers)
        assert res.status_code == 403

        res_approve = client.post("/api/admin/records/REQ-123/approve", headers=headers)
        assert res_approve.status_code == 403

    def test_admin_can_access_admin_endpoints(self):
        client = TestClient(app)
        admin_token = create_session_token(user_id=1, username="admin", role="admin")
        headers = {"Authorization": f"Bearer {admin_token}"}

        res = client.get("/api/admin/users", headers=headers)
        assert res.status_code == 200
        assert isinstance(res.json(), list)
