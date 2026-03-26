from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv


def load_release_env(project_root: Path) -> List[Path]:
    """Load repo-root .env first, then project-local fallback without overriding."""
    loaded: List[Path] = []
    repo_root = project_root.parent

    for env_path in (repo_root / ".env", project_root / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)
            loaded.append(env_path)

    return loaded
