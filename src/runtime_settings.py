import os


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def env_bool(name: str, default: bool, environ: dict[str, str] | None = None) -> bool:
    environ = environ or os.environ
    raw = str(environ.get(name, "")).strip().lower()
    if not raw:
        return default
    if raw in TRUE_VALUES:
        return True
    if raw in FALSE_VALUES:
        return False
    return default


def get_runtime_settings(environ: dict[str, str] | None = None) -> dict[str, str | bool]:
    environ = environ or os.environ
    return {
        "demo_mode": env_bool("DEMO_MODE", default=True, environ=environ),
        "allow_live_runs": env_bool("ALLOW_LIVE_RUNS", default=False, environ=environ),
        "default_config_path": str(environ.get("DEFAULT_CONFIG_PATH", "evals/config.yaml")),
        "reports_dir": str(environ.get("REPORTS_DIR", "reports")),
        "hf_token_present": bool(str(environ.get("HF_TOKEN", "")).strip()),
    }


def resolve_backend_choice(
    requested_backend: str,
    demo_mode: bool,
    allow_live_runs: bool,
) -> str:
    if demo_mode or not allow_live_runs:
        return "demo"
    if requested_backend not in {"auto", "demo", "clip"}:
        return "auto"
    return requested_backend
