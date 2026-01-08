"""
Authentication and session management for Streamlit UI.

Provides:
- User authentication with password hashing
- Session management
- Role-based access control (RBAC)
- Password security
"""

import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import streamlit as st


class AuthManager:
    """Manages user authentication and sessions."""

    def __init__(self, db_path: str = "users.db"):
        self.db_path = Path(db_path)
        self._init_database()
        self._init_default_admin()

    def _init_database(self):
        """Initialize the users database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                email TEXT,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """
        )

        # Sessions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """
        )

        # Audit log table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                action TEXT NOT NULL,
                success INTEGER NOT NULL,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    def _init_default_admin(self):
        """Create default admin user if no users exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]

        if count == 0:
            # Create default admin with password 'admin123'
            # User should change this immediately!
            self.create_user(
                username="admin",
                password="admin123",
                role="admin",
                email="admin@amperedata.local",
                full_name="System Administrator",
            )
            print("⚠️  Default admin user created: admin/admin123")
            print("    Please change the password immediately!")

        conn.close()

    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using SHA-256."""
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

    def _generate_salt(self) -> str:
        """Generate a random salt."""
        return secrets.token_hex(16)

    def _generate_session_id(self) -> str:
        """Generate a secure session ID."""
        return secrets.token_urlsafe(32)

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "user",
        email: Optional[str] = None,
        full_name: Optional[str] = None,
    ) -> bool:
        """
        Create a new user.

        Args:
            username: Unique username
            password: Plain text password (will be hashed)
            role: User role ('admin', 'user', 'viewer')
            email: User email
            full_name: User's full name

        Returns:
            True if user created successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Generate salt and hash password
            salt = self._generate_salt()
            password_hash = self._hash_password(password, salt)

            cursor.execute(
                """
                INSERT INTO users (username, password_hash, salt, role, email, full_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (username, password_hash, salt, role, email, full_name),
            )

            conn.commit()
            conn.close()

            self._log_audit(username, "user_created", True)
            return True

        except sqlite3.IntegrityError:
            return False

    def verify_credentials(self, username: str, password: str) -> bool:
        """
        Verify username and password.

        Args:
            username: Username to verify
            password: Password to verify

        Returns:
            True if credentials are valid
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT password_hash, salt, is_active
            FROM users
            WHERE username = ?
        """,
            (username,),
        )

        result = cursor.fetchone()
        conn.close()

        if not result:
            self._log_audit(username, "login_failed", False, "User not found")
            return False

        password_hash, salt, is_active = result

        if not is_active:
            self._log_audit(username, "login_failed", False, "User inactive")
            return False

        # Verify password
        computed_hash = self._hash_password(password, salt)
        if computed_hash == password_hash:
            self._log_audit(username, "login_success", True)
            self._update_last_login(username)
            return True
        else:
            self._log_audit(username, "login_failed", False, "Invalid password")
            return False

    def create_session(self, username: str, duration_hours: int = 24) -> str:
        """
        Create a new session for user.

        Args:
            username: Username for the session
            duration_hours: Session duration in hours

        Returns:
            Session ID
        """
        session_id = self._generate_session_id()
        expires_at = datetime.now() + timedelta(hours=duration_hours)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO sessions (session_id, username, expires_at)
            VALUES (?, ?, ?)
        """,
            (session_id, username, expires_at),
        )

        conn.commit()
        conn.close()

        return session_id

    def validate_session(self, session_id: str) -> Optional[str]:
        """
        Validate a session and return username if valid.

        Args:
            session_id: Session ID to validate

        Returns:
            Username if session is valid, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT username, expires_at
            FROM sessions
            WHERE session_id = ?
        """,
            (session_id,),
        )

        result = cursor.fetchone()
        conn.close()

        if not result:
            return None

        username, expires_at = result
        expires_at = datetime.fromisoformat(expires_at)

        if datetime.now() > expires_at:
            self.delete_session(session_id)
            return None

        return username

    def delete_session(self, session_id: str):
        """Delete a session (logout)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

        conn.commit()
        conn.close()

    def get_user_info(self, username: str) -> Optional[dict[str, Any]]:
        """
        Get user information.

        Args:
            username: Username to look up

        Returns:
            Dict with user info or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT username, role, email, full_name, created_at, last_login
            FROM users
            WHERE username = ?
        """,
            (username,),
        )

        result = cursor.fetchone()
        conn.close()

        if not result:
            return None

        return {
            "username": result[0],
            "role": result[1],
            "email": result[2],
            "full_name": result[3],
            "created_at": result[4],
            "last_login": result[5],
        }

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        Change user password.

        Args:
            username: Username
            old_password: Current password
            new_password: New password

        Returns:
            True if password changed successfully
        """
        # Verify old password
        if not self.verify_credentials(username, old_password):
            return False

        # Generate new salt and hash
        salt = self._generate_salt()
        password_hash = self._hash_password(new_password, salt)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE users
            SET password_hash = ?, salt = ?
            WHERE username = ?
        """,
            (password_hash, salt, username),
        )

        conn.commit()
        conn.close()

        self._log_audit(username, "password_changed", True)
        return True

    def _update_last_login(self, username: str):
        """Update last login timestamp."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE users
            SET last_login = CURRENT_TIMESTAMP
            WHERE username = ?
        """,
            (username,),
        )

        conn.commit()
        conn.close()

    def _log_audit(self, username: str, action: str, success: bool, details: Optional[str] = None):
        """Log authentication audit event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO auth_audit (username, action, success, details)
            VALUES (?, ?, ?, ?)
        """,
            (username, action, 1 if success else 0, details),
        )

        conn.commit()
        conn.close()

    def list_users(self) -> list:
        """List all users (admin only)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT username, role, email, full_name, created_at, last_login, is_active
            FROM users
            ORDER BY created_at DESC
        """
        )

        users = []
        for row in cursor.fetchall():
            users.append(
                {
                    "username": row[0],
                    "role": row[1],
                    "email": row[2],
                    "full_name": row[3],
                    "created_at": row[4],
                    "last_login": row[5],
                    "is_active": bool(row[6]),
                }
            )

        conn.close()
        return users


# Streamlit-specific authentication helpers


def init_session_state():
    """Initialize Streamlit session state for auth."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None


def require_auth(allowed_roles: Optional[list] = None):
    """
    Decorator to require authentication for Streamlit pages.

    Args:
        allowed_roles: List of roles allowed to access (None = all authenticated users)
    """
    init_session_state()

    if not st.session_state.authenticated:
        show_login_page()
        st.stop()

    if allowed_roles and st.session_state.user_info:
        user_role = st.session_state.user_info.get("role")
        if user_role not in allowed_roles:
            st.error("🚫 Access Denied")
            st.write(f"This page requires one of these roles: {', '.join(allowed_roles)}")
            st.write(f"Your role: {user_role}")
            st.stop()


def show_login_page():
    """Display the login page."""
    st.title("🔐 AmpereData Login")
    st.write("Please log in to access the platform.")

    auth = AuthManager()

    # Login form
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button("🚀 Login", use_container_width=True)

        if submit:
            if auth.verify_credentials(username, password):
                # Create session
                session_id = auth.create_session(username)
                user_info = auth.get_user_info(username)

                # Update session state
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_info = user_info
                st.session_state.session_id = session_id

                st.success(f"✅ Welcome, {user_info.get('full_name', username)}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    # Info box
    with st.expander("ℹ️ Default Credentials"):
        st.info(
            """
        **Default Admin Account:**
        - Username: `admin`
        - Password: `admin123`

        ⚠️ **Important:** Change the default password after first login!
        """
        )


def show_logout_button():
    """Display logout button in sidebar."""
    if st.session_state.get("authenticated"):
        with st.sidebar:
            st.divider()
            user_info = st.session_state.get("user_info", {})
            st.write(f"👤 **{user_info.get('full_name', st.session_state.username)}**")
            st.write(f"Role: `{user_info.get('role', 'user')}`")

            if st.button("🚪 Logout", use_container_width=True):
                auth = AuthManager()
                if st.session_state.session_id:
                    auth.delete_session(st.session_state.session_id)

                # Clear session
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.user_info = None
                st.session_state.session_id = None
                st.rerun()


def get_current_user() -> Optional[str]:
    """Get current logged-in username."""
    return st.session_state.get("username")


def get_current_user_info() -> Optional[dict]:
    """Get current user information."""
    return st.session_state.get("user_info")


def is_admin() -> bool:
    """Check if current user is admin."""
    user_info = get_current_user_info()
    return user_info and user_info.get("role") == "admin"
