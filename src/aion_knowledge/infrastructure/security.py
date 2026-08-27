"""SSRF-safe URL validation and HTTP client."""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Any, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class SSRFError(ValueError):
    """Raised when a URL fails SSRF validation."""
    pass


# ── Restricted hostnames ──────────────────────────────────────
_RESTRICTED_HOSTNAMES: frozenset[str] = frozenset({
    "localhost", "127.0.0.1", "::1", "0.0.0.0",
    "metadata.google.internal", "metadata.tencentyun.com",
    "metadata.aws.internal",
    "host.docker.internal", "gateway.docker.internal",
    "kubernetes.docker.internal",
    "kubernetes", "kubernetes.default", "kubernetes.default.svc",
})

_RESTRICTED_HOST_SUFFIXES: tuple[str, ...] = (
    ".local", ".localhost", ".internal", ".corp", ".lan", ".home",
    ".localdomain", ".svc.cluster.local", ".pod.cluster.local",
)

# ── Restricted IP networks ─────────────────────────────────────
_RESTRICTED_NETWORKS: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("100.64.0.0/10"),
    ipaddress.IPv4Network("198.18.0.0/15"),
    ipaddress.IPv4Network("198.51.100.0/24"),
    ipaddress.IPv4Network("203.0.113.0/24"),
    ipaddress.IPv4Network("192.0.0.0/24"),
    ipaddress.IPv4Network("192.0.2.0/24"),
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv4Network("240.0.0.0/4"),
    ipaddress.IPv4Network("255.255.255.255/32"),
    ipaddress.IPv4Network("172.17.0.0/16"),
    ipaddress.IPv4Network("172.18.0.0/16"),
    ipaddress.IPv4Network("172.19.0.0/16"),
    ipaddress.IPv4Network("172.20.0.0/16"),
]

# ── Blocked ports ──────────────────────────────────────────────
_BLOCKED_PORTS: frozenset[int] = frozenset({
    22, 23, 25, 445, 3389, 5432, 3306, 6379, 27017, 9200, 2379, 2380, 8500, 4001,
})

# ── SSRF Whitelist ─────────────────────────────────────────────
_SSRF_WHITELIST: Optional[List[str]] = None


def reset_ssrf_whitelist() -> None:
    """Reset whitelist cache so next call re-reads env var. 测试用。"""
    global _SSRF_WHITELIST
    _SSRF_WHITELIST = None


def _load_ssrf_whitelist() -> List[str]:
    """Parse comma-separated SSRF_WHITELIST env var into a list of patterns."""
    global _SSRF_WHITELIST
    if _SSRF_WHITELIST is not None:
        return _SSRF_WHITELIST
    raw = os.environ.get("SSRF_WHITELIST", "")
    _SSRF_WHITELIST = [p.strip() for p in raw.split(",") if p.strip()]
    return _SSRF_WHITELIST


def _is_whitelisted(hostname: str) -> bool:
    """Check hostname against SSRF whitelist."""
    hostname_lower = hostname.lower()
    for pattern in _load_ssrf_whitelist():
        if pattern.startswith("*."):
            suffix = pattern[1:]  # ".example.com"
            if hostname_lower.endswith(suffix) or hostname_lower == suffix[1:]:
                return True
        elif hostname_lower == pattern.lower():
            return True
    return False


def _is_ip_restricted(ip_str: str | int) -> tuple[bool, str]:
    """Check if an IP string falls within any restricted range."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False, ""
    for net in _RESTRICTED_NETWORKS:
        if ip in net:
            return True, f"restricted network {net}"
    if ip.is_multicast:
        return True, "multicast address"
    if ip.is_unspecified:
        return True, "unspecified address"
    return False, ""


def _resolve_and_check(hostname: str) -> None:
    """Resolve hostname and verify all resolved IPs are safe."""
    try:
        addrs = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise SSRFError(f"DNS resolution failed for {hostname}: {e}") from e
    for family, *_rest, sockaddr in addrs:
        ip_str = sockaddr[0]
        restricted, reason = _is_ip_restricted(ip_str)
        if restricted:
            raise SSRFError(
                f"hostname {hostname} resolves to restricted IP {ip_str} ({reason})"
            )


def validate_url_for_ssrf(url: str) -> None:
    """Validate URL for SSRF safety. Raises SSRFError if unsafe."""
    if not url:
        raise SSRFError("URL is empty")
    if len(url) > 2048:
        raise SSRFError("URL exceeds maximum length (2048)")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise SSRFError(f"invalid scheme: {scheme} (only http/https allowed)")

    hostname = parsed.hostname or ""
    if not hostname:
        raise SSRFError("URL has no hostname")

    # Whitelist check — skip all further checks if whitelisted
    if _is_whitelisted(hostname):
        return

    # Exact hostname match
    if hostname.lower() in _RESTRICTED_HOSTNAMES:
        raise SSRFError(f"hostname {hostname} is restricted")

    # Suffix match
    hostname_lower = hostname.lower()
    for suffix in _RESTRICTED_HOST_SUFFIXES:
        if hostname_lower.endswith(suffix):
            raise SSRFError(f"hostname suffix {suffix} is restricted")

    # IP literal check
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass  # not an IP literal — continue to DNS
    else:
        restricted, reason = _is_ip_restricted(hostname)
        if restricted:
            raise SSRFError(f"direct IP {hostname} is blocked: {reason}")
        raise SSRFError("direct IP address is not allowed, use domain name")

    # Port check
    port = parsed.port
    if port and port in _BLOCKED_PORTS:
        raise SSRFError(f"port {port} is blocked for security reasons")

    # DNS resolution
    _resolve_and_check(hostname)


class SSRFSafeHTTPClient:
    """HTTP client that validates every request and redirect against SSRF rules."""

    def __init__(self, timeout: float = 15.0, max_redirects: int = 5):
        self._timeout = timeout
        self._max_redirects = max_redirects

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform GET with SSRF-safe redirect following."""
        validate_url_for_ssrf(url)
        with httpx.Client(timeout=self._timeout, follow_redirects=False) as client:
            resp = client.get(url, **kwargs)
            redirects = 0
            while resp.is_redirect and redirects < self._max_redirects:
                location = resp.headers.get("location", "")
                if not location:
                    break
                validate_url_for_ssrf(location)
                resp = client.get(location, **kwargs)
                redirects += 1
            return resp
