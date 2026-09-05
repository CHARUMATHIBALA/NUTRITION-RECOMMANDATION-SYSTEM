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

Secret key configuration
------------------------
The JWT/cookie signing key is loaded from the JWT_SECRET_KEY environment
variable.  Set it before starting the application:

    export JWT_SECRET_KEY="<your-secure-random-secret>"   # Linux/macOS
    $env:JWT_SECRET_KEY="<your-secure-random-secret>"     # Windows PowerShell

For local development you may place it in a .env file at the project root.
See .env.example for the expected format.
"""

import os
import yaml
import streamlit as st
import streamlit_authenticator as stauth

# ---------------------------------------------------------------------------
# Load python-dotenv when available so that a local .env file is picked up
# automatically during development.  This import is optional — if the package
# is absent the application still works as long as JWT_SECRET_KEY is set in
# the shell environment before launching Streamlit.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()           # silently loads .env from the project root if present
except ImportError:
    pass                    # python-dotenv not installed — rely on shell env vars


def _get_jwt_secret() -> str:
    """Return the JWT signing key from the environment.

    Raises
    ------
    RuntimeError
        If JWT_SECRET_KEY is not set or is an empty string.
        The error message is safe to display — it does not reveal any secret.
    """
    secret = os.environ.get("JWT_SECRET_KEY", "").strip()
    if not secret:
        raise RuntimeError(
            "JWT_SECRET_KEY is not configured. "
            "Please set the JWT_SECRET_KEY environment variable before "
            "starting the application. "
            "See .env.example for the expected format."
        )
    return secret

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

    Raises
    ------
    RuntimeError (via _get_jwt_secret)
        If JWT_SECRET_KEY environment variable is not set.
    """
    creds = _load_credentials()

    # _get_jwt_secret() raises a clear RuntimeError if the env var is absent.
    # The RuntimeError propagates to Streamlit which renders it as a fatal
    # error page — no secret value is exposed in the message.
    jwt_secret = _get_jwt_secret()

    authenticator = stauth.Authenticate(
        credentials=creds["credentials"],
        cookie_name="health_dashboard_v3",
        key=jwt_secret,
        cookie_expiry_days=30,
    )

    # Only render the login form when the user is NOT yet authenticated.
    # Calling login() unconditionally on every rerun causes it to render
    # a redundant widget in the main area after a successful login, which
    # can interfere with the page layout.
    if not st.session_state.get("authentication_status"):
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
