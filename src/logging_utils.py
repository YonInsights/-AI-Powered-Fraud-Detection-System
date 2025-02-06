import os
import sys
import logging

def setup_logging():
    """Configures logging settings."""
    log_dir = "logs"
    log_file = os.path.join(log_dir, "preprocessing.log")

    # Check if the directory exists, if not, create it
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)
