"""Single configuration surface — fixes REPORT.md M7.

Every key here is READ by real code in this repo (unlike asop's 7 configs that
orchestrated non-existent modules, or etbackend's dead `soar_auto_execute_max_tier`).
Values come from environment variables with safe defaults; secrets have NO baked
production default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache


class ResponseMode(str, Enum):
    """Org-level SOAR policy — REPORT.md W1 (was undocumented in code)."""
    MANUAL = "manual"          # every action requires human approval
    SEMI = "semi"              # low-impact auto-executes; high-impact -> approval
    FULL = "full"              # auto-execute up to soar_auto_execute_max_tier


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # SOAR gate
    soar_confidence_threshold: float = 0.85
    response_mode: ResponseMode = ResponseMode.SEMI
    soar_auto_execute_max_tier: int = 3
    # UEBA
    ueba_anomaly_threshold: float = 0.5
    ueba_high_sensitivity_threshold: float = 0.3
    # auth
    jwt_secret: str = ""                              # NO production default (M-5)
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    # artifacts
    attribution_artifact: str = os.path.join("models", "attribution.joblib")
    transition_matrix: str = os.path.join("models", "transition_matrix.json")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            soar_confidence_threshold=float(_get("SOAR_CONFIDENCE_THRESHOLD", "0.85")),
            response_mode=ResponseMode(_get("RESPONSE_MODE", "semi")),
            soar_auto_execute_max_tier=int(_get("SOAR_AUTO_EXECUTE_MAX_TIER", "3")),
            ueba_anomaly_threshold=float(_get("UEBA_ANOMALY_THRESHOLD", "0.5")),
            ueba_high_sensitivity_threshold=float(_get("UEBA_HIGH_SENSITIVITY_THRESHOLD", "0.3")),
            jwt_secret=_get("JWT_SECRET", ""),
            jwt_algorithm=_get("JWT_ALGORITHM", "HS256"),
            access_token_minutes=int(_get("ACCESS_TOKEN_MINUTES", "60")),
            attribution_artifact=_get("ATTRIBUTION_ARTIFACT", os.path.join("models", "attribution.joblib")),
            transition_matrix=_get("TRANSITION_MATRIX", os.path.join("models", "transition_matrix.json")),
        )

    def require_jwt_secret(self) -> str:
        if not self.jwt_secret:
            raise RuntimeError(
                "JWT_SECRET is not set. Refusing to run auth with a baked default "
                "secret (REPORT.md M-5). Set the JWT_SECRET environment variable."
            )
        return self.jwt_secret


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
