"""Feature extractors — Activity -> dict[str, float].

Ported faithfully from asop ``archive/features/extractors/*`` (real, data-driven
logic: indicator binaries, known ports, registry categories) and adapted to the
canonical schema: ``event.event_type`` (str) -> ``str(event.event_id)``,
``event.raw_data`` -> ``event.raw_event``.

Note (REPORT.md H4 synergy): the process extractor keys primarily off Sysmon
EventID 10 (ProcessAccess). The old parser dropped EventID 10 entirely, so these
features were structurally always-zero. The merged parser now emits EventID 10/7,
so these features carry real signal for the first time.

These are the ATTRIBUTION feature space only. UEBA has its own space (src/ueba).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict

from src.canon.schema import Activity, CanonicalEvent


def _eid(e: CanonicalEvent) -> str:
    return str(e.event_id)


def _basename(path: str) -> str:
    return path.split("\\")[-1].split("/")[-1]


INDICATOR_BINARIES = [
    "powershell.exe", "cmd.exe", "wmic.exe", "mshta.exe", "rundll32.exe",
    "regsvr32.exe", "cscript.exe", "wscript.exe", "schtasks.exe", "sc.exe",
    "net.exe", "net1.exe", "lsass.exe", "svchost.exe", "services.exe",
    "wmiprvse.exe", "msbuild.exe", "installutil.exe", "cmstp.exe", "hh.exe",
    "control.exe", "msiexec.exe", "dllhost.exe", "mavinject.exe", "taskhost.exe",
    "taskhostw.exe", "conhost.exe", "explorer.exe", "auditpol.exe", "wevtutil.exe",
    "reg.exe", "vbc.exe",
]

KNOWN_PORTS = [21, 22, 80, 135, 139, 389, 443, 445, 1433, 3389, 5985, 5986, 8080, 9090]

REGISTRY_CATEGORIES = {
    "reg_run_keys": ["\\currentversion\\run", "\\currentversion\\runonce"],
    "reg_services": ["\\services\\", "\\system\\currentcontrolset\\services"],
    "reg_audit_policy": ["\\audit\\", "\\auditpol", "\\eventlog"],
    "reg_security": ["\\security\\", "\\sam\\", "\\lsa\\"],
    "reg_image_hijack": ["\\image file execution", "\\ifeo\\"],
    "reg_com_objects": ["\\clsid\\", "\\inprocserver"],
}

_EID_HISTOGRAM = ["1", "3", "5", "7", "8", "10", "11", "12", "13", "14", "15",
                  "17", "18", "22", "23", "4103", "4104", "4624", "4625", "4688",
                  "4697", "4720", "4732"]
_KNOWN_EXT = ["exe", "dll", "ps1", "bat", "vbs", "js", "hta", "tmp", "dat"]


def extract_process(a: Activity) -> Dict[str, float]:
    source_images, target_images, loaded_dlls = [], [], []
    for e in a.events:
        etype, raw = _eid(e), e.raw_event
        if etype == "10":  # ProcessAccess — LSASS/credential-access signal
            src, tgt = str(raw.get("SourceImage", "")).lower(), str(raw.get("TargetImage", "")).lower()
            if src:
                source_images.append(_basename(src))
            if tgt:
                target_images.append(_basename(tgt))
        elif etype in ("1", "4688"):
            img = str(raw.get("Image", raw.get("NewProcessName", ""))).lower()
            parent = str(raw.get("ParentImage", raw.get("ParentProcessName", ""))).lower()
            if img:
                source_images.append(_basename(img))
            if parent:
                target_images.append(_basename(parent))
        elif etype == "7":
            loaded = str(raw.get("ImageLoaded", "")).lower()
            if loaded:
                loaded_dlls.append(_basename(loaded))

    all_images = source_images + target_images
    out: Dict[str, float] = {f"proc_{b.replace('.exe', '')}": float(all_images.count(b))
                             for b in INDICATOR_BINARIES}
    create_events = [e for e in a.events if _eid(e) in ("1", "4688")]
    cmd_lengths = [len(str(e.raw_event.get("CommandLine", ""))) for e in create_events]
    out.update({
        "process_access_count": float(sum(1 for e in a.events if _eid(e) == "10")),
        "process_create_count": float(len(create_events)),
        "image_load_count": float(sum(1 for e in a.events if _eid(e) == "7")),
        "lsass_targeted_count": float(target_images.count("lsass.exe")),
        "unique_source_images": float(len(set(source_images))),
        "unique_target_images": float(len(set(target_images))),
        "unique_dlls_loaded": float(len(set(loaded_dlls))),
        "max_command_line_length": float(max(cmd_lengths) if cmd_lengths else 0.0),
    })
    return out


def extract_network(a: Activity) -> Dict[str, float]:
    net = [e for e in a.events if _eid(e) in ("3", "5156")]
    dst_ips, dst_ports = set(), []
    for e in net:
        ip = e.raw_event.get("DestinationIp")
        if ip:
            dst_ips.add(ip)
        p = e.raw_event.get("DestinationPort")
        if p is not None:
            try:
                dst_ports.append(int(p))
            except (ValueError, TypeError):
                pass
    out = {f"net_port_{port}": float(dst_ports.count(port)) for port in KNOWN_PORTS}
    out.update({
        "network_fanout": float(len(dst_ips)),
        "network_flow_count": float(len(net)),
        "unique_port_count": float(len(set(dst_ports))),
    })
    return out


def extract_host(a: Activity) -> Dict[str, float]:
    reg = [e for e in a.events if _eid(e) in ("12", "13", "14")]
    files = [e for e in a.events if _eid(e) == "11"]
    counter = Counter(_eid(e) for e in a.events)
    out = {f"eid_{eid}_count": float(counter.get(eid, 0)) for eid in _EID_HISTOGRAM}
    reg_cat = {cat: 0.0 for cat in REGISTRY_CATEGORIES}
    for e in reg:
        target = str(e.raw_event.get("TargetObject", "")).lower()
        for cat, patterns in REGISTRY_CATEGORIES.items():
            if any(p in target for p in patterns):
                reg_cat[cat] += 1.0
    exts = []
    for e in files:
        fn = str(e.raw_event.get("TargetFilename", ""))
        if "." in fn:
            exts.append(fn.rsplit(".", 1)[-1].lower())
    out.update({f"file_ext_{ext}": float(exts.count(ext)) for ext in _KNOWN_EXT})
    out.update(reg_cat)
    out.update({
        "registry_mod_count": float(len(reg)),
        "file_creation_count": float(len(files)),
        "unique_event_types": float(len(counter)),
    })
    return out


def extract_authentication(a: Activity) -> Dict[str, float]:
    failed = sum(1 for e in a.events if _eid(e) == "4625")
    ok = sum(1 for e in a.events if _eid(e) == "4624")
    dur = max(1.0, (a.end_time - a.start_time).total_seconds())
    return {
        "failed_login_count": float(failed),
        "successful_login_count": float(ok),
        "login_frequency": float(failed + ok) / (dur / 60.0),
    }


def extract_temporal(a: Activity) -> Dict[str, float]:
    h = a.start_time.hour
    return {
        "is_off_hours": 1.0 if (h < 9 or h >= 17) else 0.0,
        "window_duration": float((a.end_time - a.start_time).total_seconds()),
    }


ALL_EXTRACTORS = (
    extract_authentication, extract_process, extract_network,
    extract_host, extract_temporal,
)
