"""Configuration de la journalisation avec rotation des fichiers."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

LOGGER_NAME = "app"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s"


def setup_logging(
    log_dir: Union[str, Path],
    level: int = logging.INFO,
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """Configure le logger racine de l'application (idempotent).

    Les traces techniques vont dans ``logs/app.log`` ; l'utilisateur ne voit que
    des messages lisibles dans la barre d'état.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Idempotence : on retire les handlers existants (rechargement, tests).
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT)

    try:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path / "app.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:  # dossier non inscriptible : on continue en console
        print(f"[logger] Impossible de créer le fichier de log : {exc}", file=sys.stderr)

    if console:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    return logger
