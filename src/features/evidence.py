"""Modular Evidence Filter — identifies sessions carrying explicit attack/security indicators.

Keeps ingestion independent of evidence policy (REPORT.md alignment).
"""
from __future__ import annotations

from typing import Dict


class EvidenceFilter:
    """Evaluates whether a feature map contains explicit attack/process/network indicators."""

    def __init__(self, check_lsass: bool = True, check_process_access: bool = True,
                 check_network: bool = True, check_registry: bool = True,
                 check_failed_logins: bool = True):
        self.check_lsass = check_lsass
        self.check_process_access = check_process_access
        self.check_network = check_network
        self.check_registry = check_registry
        self.check_failed_logins = check_failed_logins

    def has_evidence(self, fmap: Dict[str, float]) -> bool:
        if self.check_lsass and fmap.get("lsass_targeted_count", 0.0) > 0:
            return True
        if self.check_process_access and fmap.get("process_access_count", 0.0) > 0:
            return True
        if self.check_network and fmap.get("network_flow_count", 0.0) > 0:
            return True
        if self.check_registry and fmap.get("registry_mod_count", 0.0) > 0:
            return True
        if self.check_failed_logins and fmap.get("failed_login_count", 0.0) > 0:
            return True
        for k, v in fmap.items():
            if k.startswith("proc_") and v > 0:
                return True
        return False


# Global default instance
DEFAULT_EVIDENCE_FILTER = EvidenceFilter()
