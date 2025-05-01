import logging
import sys
from datetime import datetime
from pathlib import Path

# Rich imports
from rich.logging import RichHandler

LOGS_DIR = Path("logs")

# --- Custom Logging Handler Removed ---

# --- Setup Function ---


def setup_logging(command_name: str, log_level: int = logging.INFO):
    """
    Configures logging to write to a timestamped file in the logs directory
    and use RichHandler for console output.

    Args:
        command_name: The name of the CLI command being run (used for filename).
        log_level: The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOGS_DIR / f"repobird_{command_name}_{timestamp}.log"

    # --- File Handler ---
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    # File handler should capture everything from the specified level
    file_handler.setLevel(log_level)

    # --- Console Handler (Standard RichHandler) ---
    # Keep console output clean, show INFO and above by default
    # RichHandler automatically handles markup
    console_handler = RichHandler(
        level=logging.INFO,  # Show INFO and above on console
        rich_tracebacks=True,
        show_path=False,  # Don't show module path
        show_time=False,  # Don't show timestamp
        markup=True,  # Enable Rich markup like [bold]
    )

    # --- Root Logger Configuration ---
    # Get the root logger
    root_logger = logging.getLogger()
    # Set the root logger level (this is the minimum level for *all* handlers)
    root_logger.setLevel(log_level)

    # Clear existing handlers (important if this function is called multiple times)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add the configured handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Optional: Silence overly verbose libraries if needed
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("playwright").setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Log file: {log_filename}")
    logging.info(f"Command run: {' '.join(sys.argv)}")


# Example usage (for testing the setup function itself)
if __name__ == "__main__":
    setup_logging("test_setup", log_level=logging.DEBUG)
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    try:
        1 / 0
    except ZeroDivisionError:
        logging.exception("This is an exception message.")
