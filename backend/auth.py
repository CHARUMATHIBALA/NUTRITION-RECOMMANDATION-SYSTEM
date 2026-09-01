# backend/auth.py
"""Authentication utilities for the Smart Health Dashboard.

Wraps streamlit-authenticator with a YAML credentials file.
Credentials file location: project root / credentials.yaml

Minimal YAML format::

    credentials:
      usernames:
        admin:
          name: "Admin"
          password: "$2b$12$..."   # bcrypt hash
          email: "admin@example.com"
"""

import os
import yaml
import streamlit as st
import streamlit_authenticator as stauth

CREDENTIALS_FILE = "credentials.yaml"


def _load_credentials() -> dict:
    """Load credentials YAML, creating a default file if absent."""
    if not os.path.exists(CREDENTIALS_FILE):
        import bcrypt
        hashed = bcrypt.hashpw(b"password", bcrypt.gensalt()).decode()
        default = {
            "credentials": {
                "usernames": {
                    "admin": {
                        "name": "Admin",
                        "password": hashed,
                        "email": "admin@example.com",
                    }
                }
            }
        }
        with open(CREDENTIALS_FILE, "w") as f:
            yaml.safe_dump(default, f)
    with open(CREDENTIALS_FILE) as f:
        return yaml.safe_load(f)


def authenticate():
    """Render the login widget and return (name, auth_status, username).

    Returns
    -------
    name : str or None
    auth_status : bool or None
        True  — logged in successfully
        False — wrong credentials
        None  — not yet attempted
    username : str or None
    """
    creds = _load_credentials()
    authenticator = stauth.Authenticate(
        credentials=creds["credentials"],
        cookie_name="health_dashboard_v3",
        key="smart_health_dashboard_secure_jwt_signing_key_2024_minimum_32_bytes_required_for_security",
        cookie_expiry_days=30,
    )
    authenticator.login(location="main")
    st.session_state["authenticator"] = authenticator

    auth_status = st.session_state.get("authentication_status")
    name       = st.session_state.get("name")
    username   = st.session_state.get("username")
    return name, auth_status, username


def logout():
    """Log out the current user and clear all auth-related session keys."""
    if "authenticator" in st.session_state:
        try:
            st.session_state["authenticator"].logout()
        except Exception:
            pass
    # Clear every auth key — use the correct session-state key names
    for key in ["name", "username", "authentication_status", "auth_status",
                "logout", "authenticator"]:
        st.session_state.pop(key, None)
