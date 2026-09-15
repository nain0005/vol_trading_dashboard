"""Run this once, interactively, in your own terminal to create a fresh
Robinhood session. When Robinhood texts/emails you a verification code,
type it into the prompt that appears and press Enter.

    .venv/bin/python login_once.py
"""
from app.auth import login

login()
print("Logged in — session cached to ~/.tokens/robinhood.pickle")
