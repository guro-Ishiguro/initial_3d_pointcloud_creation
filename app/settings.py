import os


def apply_env_overrides():
    """
    Central place to set defaults or transform environment variables before pipeline runs.
    Extend as needed. Returns a dict of effective settings for logging/debug.
    """
    effective = {}
    # Example: ensure log level default
    effective["PM_LOG_LEVEL"] = os.getenv("PM_LOG_LEVEL", "INFO")
    os.environ["PM_LOG_LEVEL"] = effective["PM_LOG_LEVEL"]

    # Propagation neighbor directions
    effective["PM_PROP_DIRS"] = os.getenv("PM_PROP_DIRS", "4")
    os.environ["PM_PROP_DIRS"] = effective["PM_PROP_DIRS"]

    # Priority sweeps
    effective["PM_PRIORITY_SWEEPS"] = os.getenv("PM_PRIORITY_SWEEPS", "8")
    os.environ["PM_PRIORITY_SWEEPS"] = effective["PM_PRIORITY_SWEEPS"]

    return effective
