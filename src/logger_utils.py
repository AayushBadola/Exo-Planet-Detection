import logging
import sys

def setup_logging(log_file: str, level=logging.INFO):
    """
    Sets up logging to both console and a file.
    """
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'), # Append mode
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Set higher level for noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING) # Matplotlib uses PIL

    # Get the root logger
    logger = logging.getLogger() # This refers to the root logger
    return logger