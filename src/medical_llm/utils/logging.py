"""
Logging helper functions.
"""

from medical_llm.config import DEFAULT_CONFIG


def log_event(message: str = "", log_file: str = DEFAULT_CONFIG.log_file) -> None:
    """Prints message to stdout and appends to specified log file."""
    print(message, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
