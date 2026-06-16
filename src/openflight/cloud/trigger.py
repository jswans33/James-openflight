"""Fire-and-forget push trigger for the server's session-end happy path.

Spawns ``openflight-cloud push`` as a detached subprocess so an upload never
blocks or delays shot processing. The systemd timer remains the safety net that
heals wifi outages; this is just the fast path.
"""

import subprocess
import sys
from pathlib import Path
from typing import Callable

from .config import CONFIG_PATH, CloudConfig


def fire_push_async(
    config: CloudConfig,
    log_dir: Path,
    config_path: Path = CONFIG_PATH,
    popen_fn: Callable[..., object] = subprocess.Popen,
) -> bool:
    """Spawn a detached ``push`` if the uploader is active. Never raises.

    Returns True if a push was spawned, False otherwise (inactive or spawn
    failed). The caller is on the shot/session path, so all errors are
    swallowed.
    """
    if not config.is_active():
        return False

    cmd = [
        sys.executable,
        "-m",
        "openflight.cloud.cli",
        "--config",
        str(config_path),
        "--log-dir",
        str(log_dir),
        "push",
    ]
    try:
        popen_fn(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except (OSError, ValueError):
        return False
