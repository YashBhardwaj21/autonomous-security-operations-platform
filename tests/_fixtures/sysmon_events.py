"""Hand-built raw-event dicts shaped like real OTRF Sysmon/Security records.

ISOLATED TEST FIXTURES ONLY. Nothing in src/ imports this module (enforced by
scripts/check_no_dummy_in_src.py). These are test *inputs* that mirror the real
event schema; they are never application data and never enter the runtime path.

For end-to-end tests on REAL OTRF telemetry, see tests/integration (auto-skips
when data/raw/Security-Datasets is absent).
"""
from __future__ import annotations

# A small two-host credential-access-shaped burst.
HOSTA_PROCESS = {
    "EventID": 1, "UtcTime": "2021-01-01 10:00:00.000", "Hostname": "HOSTA",
    "ProcessGuid": "{proc-a}", "ProcessId": "100",
    "Image": "C:/Windows/System32/powershell.exe", "CommandLine": "powershell -enc AAAA",
    "ParentImage": "C:/Windows/explorer.exe",
}
HOSTA_LSASS_ACCESS = {
    "EventID": 10, "UtcTime": "2021-01-01 10:00:05.000", "Hostname": "HOSTA",
    "SourceProcessGUID": "{proc-a}", "TargetProcessGUID": "{proc-lsass}", "TargetProcessId": "8",
    "TargetImage": "C:/Windows/System32/lsass.exe",
    "SourceImage": "C:/Windows/System32/powershell.exe",
}
HOSTA_LOGON = {
    "EventID": 4624, "UtcTime": "2021-01-01 10:00:01.000", "Hostname": "HOSTA",
    "TargetUserSid": "S-1-5-21-1", "TargetUserName": "alice", "TargetDomainName": "CORP",
    "TargetLogonId": "0x111", "LogonType": "2",
}
HOSTB_LOGON = {
    "EventID": 4624, "UtcTime": "2021-01-01 10:00:02.000", "Hostname": "HOSTB",
    "TargetUserSid": "S-1-5-21-2", "TargetUserName": "bob", "TargetDomainName": "CORP",
    "TargetLogonId": "0x222", "LogonType": "3",
}
SYSTEM_LOGON = {  # well-known logon id -> must fall through to gap-window, not group
    "EventID": 4624, "UtcTime": "2021-01-01 10:00:03.000", "Hostname": "HOSTA",
    "TargetUserSid": "S-1-5-18", "TargetUserName": "SYSTEM", "TargetDomainName": "NT AUTHORITY",
    "TargetLogonId": "0x3e7", "LogonType": "5",
}
BAD_TIMESTAMP = {"EventID": 1, "Hostname": "HOSTA", "ProcessId": "1", "Image": "x.exe"}
UNSUPPORTED_EVENT = {"EventID": 4688, "UtcTime": "2021-01-01 10:00:00.000", "Hostname": "HOSTA"}

TWO_HOST_BURST = [HOSTA_PROCESS, HOSTA_LSASS_ACCESS, HOSTA_LOGON, HOSTB_LOGON]

# Real-OTRF-shaped compound scenario metadata (verified structure: GoldenSAML).
GOLDEN_SAML_METADATA = {
    "title": "Golden SAML AD FS Mail Access", "type": "compound",
    "attack_mappings": [
        {"technique": "T1552", "sub-technique": "004", "tactics": ["TA0006"]},
        {"technique": "T1606", "sub-technique": "002", "tactics": ["TA0006"]},
        {"technique": "T1078", "sub-technique": "004", "tactics": ["TA0001", "TA0003", "TA0004", "TA0005"]},
        {"technique": "T1114", "sub-technique": None, "tactics": ["TA0009"]},
    ],
}
