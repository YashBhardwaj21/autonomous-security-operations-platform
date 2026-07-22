"""JWT auth — real, with the etbackend M-5 bugs fixed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Protocol

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.config.settings import Settings, get_settings

_bearer = HTTPBearer(auto_error=True)


def _hash_pw(password: str) -> str:
    # bcrypt caps input at 72 bytes; encode and cap explicitly (documented).
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def _verify_pw(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


class UserProvider(Protocol):
    def get(self, username: str) -> Optional[Dict]:
        """Return {'username','password_hash','roles':[...]} or None."""


class InMemoryUserProvider:
    """Operator-supplied users with bcrypt hashes (NOT plaintext)."""

    def __init__(self, users: Dict[str, Dict]):
        self._users = users

    def get(self, username: str) -> Optional[Dict]:
        return self._users.get(username)

    @staticmethod
    def hash_password(pw: str) -> str:
        return _hash_pw(pw)


def authenticate(provider: UserProvider, username: str, password: str) -> Optional[Dict]:
    user = provider.get(username)
    if not user:
        return None
    if not _verify_pw(password, user.get("password_hash", "")):
        return None
    return user


def issue_access_token(username: str, roles, settings: Optional[Settings] = None) -> str:
    s = settings or get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username, "roles": list(roles), "type": "access",
        "iat": now, "exp": now + timedelta(minutes=s.access_token_minutes),
    }
    return jwt.encode(payload, s.require_jwt_secret(), algorithm=s.jwt_algorithm)


def decode_token(token: str, expected_type: str = "access",
                 settings: Optional[Settings] = None) -> Dict:
    s = settings or get_settings()
    try:
        claims = jwt.decode(token, s.require_jwt_secret(), algorithms=[s.jwt_algorithm])
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {e}")
    if claims.get("type") != expected_type:            # reject refresh tokens on access paths
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong token type")
    return claims


def current_user(cred: HTTPAuthorizationCredentials = Depends(_bearer)) -> Dict:
    return decode_token(cred.credentials, expected_type="access")
