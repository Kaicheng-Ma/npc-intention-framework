"""Shared local Qwen configuration for the standalone NPC experiments."""

from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values


# 2026 migration: resolve configuration and data independently of the shell's cwd.
WORKSPACE_ROOT = Path(__file__).resolve().parent
QWEN_CONFIG_PATH = WORKSPACE_ROOT / "config" / "qwen.env"


def load_qwen_config() -> dict:
    """Read explicit local settings without reusing historical or shell credentials."""
    if not QWEN_CONFIG_PATH.is_file():
        raise ValueError(f"Create {QWEN_CONFIG_PATH} from config/qwen.env.example.")

    # Disable interpolation so a key is treated literally; never print its value.
    values = dotenv_values(QWEN_CONFIG_PATH, encoding="utf-8-sig", interpolate=False)
    required = ("QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL")
    missing = [name for name in required if not (values.get(name) or "").strip()]
    if missing:
        raise ValueError(f"Fill {', '.join(missing)} in {QWEN_CONFIG_PATH}.")

    base_url = values["QWEN_BASE_URL"].strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("QWEN_BASE_URL must be an HTTPS API base URL without query or fragment.")
    if not parsed.path.endswith("/compatible-mode/v1"):
        raise ValueError(
            "QWEN_BASE_URL must be the official OpenAI-compatible base URL ending "
            "in /compatible-mode/v1, without /chat/completions."
        )

    # Optional provider setting: a blank value preserves the model's own default.
    # This controls Qwen reasoning, not the experiment's explicit analysis stages.
    thinking = (values.get("QWEN_ENABLE_THINKING") or "").strip().lower()
    if thinking not in ("", "true", "false"):
        raise ValueError("QWEN_ENABLE_THINKING must be blank, true, or false.")
    extra_body = {} if not thinking else {"enable_thinking": thinking == "true"}
    return {
        "api_key": values["QWEN_API_KEY"].strip(),
        "base_url": base_url,
        "model": values["QWEN_MODEL"].strip(),
        "extra_body": extra_body,
    }
