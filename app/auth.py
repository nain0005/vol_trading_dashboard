"""Robinhood login handling.

robin_stocks caches its own session token under ~/.tokens/robinhood.pickle,
so once you've logged in once with `store_session=True` (the default), most
days you won't be prompted for MFA again. We still support a TOTP secret in
.env for fully unattended re-auth (e.g. after the cached session expires).
"""
import os

import pyotp
import robin_stocks.robinhood as rh
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


class LoginError(RuntimeError):
    pass


def _mfa_code() -> str | None:
    secret = os.getenv("ROBINHOOD_TOTP_SECRET", "").strip()
    if not secret:
        return None
    return pyotp.TOTP(secret).now()


def login(mfa_from_user: str | None = None) -> dict:
    """Log in to Robinhood. Returns robin_stocks' login payload.

    Raises LoginError with a user-facing message on failure so the Streamlit
    UI can show it instead of crashing.
    """
    username = os.getenv("ROBINHOOD_USERNAME", "").strip()
    password = os.getenv("ROBINHOOD_PASSWORD", "").strip()
    if not username or not password:
        raise LoginError(
            "Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD in your .env file "
            "(copy .env.example to .env first)."
        )

    mfa_code = mfa_from_user or _mfa_code()

    try:
        login_payload = rh.login(
            username=username,
            password=password,
            mfa_code=mfa_code,
            store_session=True,
        )
    except Exception as exc:  # robin_stocks raises bare Exceptions on auth failure
        raise LoginError(f"Robinhood login failed: {exc}") from exc

    if not login_payload or "access_token" not in login_payload:
        raise LoginError(
            "Robinhood login did not return an access token — check your "
            "credentials, or enter the MFA code sent to you and try again."
        )
    return login_payload


def logout() -> None:
    try:
        rh.logout()
    finally:
        for key in ("rh_logged_in",):
            st.session_state.pop(key, None)


def ensure_logged_in() -> bool:
    """Streamlit-friendly login gate. Returns True once authenticated."""
    if st.session_state.get("rh_logged_in"):
        return True

    st.subheader("Log in to Robinhood")
    st.caption(
        "Credentials are read from your local .env file and sent directly to "
        "Robinhood — nothing is stored anywhere else."
    )

    needs_manual_mfa = st.session_state.get("rh_needs_mfa", False)
    mfa_input = None
    if needs_manual_mfa:
        mfa_input = st.text_input("Enter the MFA/2FA code Robinhood sent you", key="mfa_code_input")

    if st.button("Log in", type="primary"):
        try:
            login(mfa_from_user=mfa_input)
            st.session_state["rh_logged_in"] = True
            st.session_state["rh_needs_mfa"] = False
            st.rerun()
        except LoginError as exc:
            msg = str(exc)
            if "verification" in msg.lower() or "mfa" in msg.lower():
                st.session_state["rh_needs_mfa"] = True
                st.rerun()
            else:
                st.error(msg)
    return False
