"""Validation des saisies utilisateur (URL de flux, index de caméra, bornes)."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

ALLOWED_SCHEMES = ("http", "https", "rtsp", "rtsps")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str = ""

    def __bool__(self) -> bool:  # permet ``if validate_...(x):``
        return self.ok


def validate_stream_url(url: str) -> ValidationResult:
    """Validation basique d'une URL de flux (syntaxe uniquement, pas de test réseau)."""
    url = (url or "").strip()
    if not url:
        return ValidationResult(False, "L'adresse du flux est vide.")
    if any(ch.isspace() for ch in url):
        return ValidationResult(False, "L'adresse ne doit pas contenir d'espaces.")
    try:
        parsed = urlparse(url)
        port = parsed.port  # lève ValueError si hors plage / invalide
    except ValueError:
        return ValidationResult(False, "Adresse ou port invalide.")
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return ValidationResult(
            False, f"Protocole non pris en charge (attendu : {', '.join(ALLOWED_SCHEMES)})."
        )
    if not parsed.hostname:
        return ValidationResult(False, "Adresse sans nom d'hôte (ex. 192.168.1.20).")
    if port is not None and not (1 <= port <= 65535):
        return ValidationResult(False, "Port hors plage (1-65535).")
    return ValidationResult(True, "Adresse valide.")


def validate_camera_index(value: object, max_index: int = 63) -> ValidationResult:
    try:
        index = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ValidationResult(False, "L'index de caméra doit être un entier.")
    if not (0 <= index <= max_index):
        return ValidationResult(False, f"Index de caméra hors plage (0-{max_index}).")
    return ValidationResult(True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
