import os
import sys
import logging
import uuid
from pythonjsonlogger import jsonlogger
from datetime import datetime


def create_logger():
    """
    Configure the root logger with a JSON format and handlers.
    """
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(logging.INFO)  # Set logging level

    # Create a console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Define the JSON format for logs
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        json_ensure_ascii=False
    )
    handler.setFormatter(formatter)

    # Avoid adding multiple handlers if already exists
    if not logger.hasHandlers():
        logger.addHandler(handler)
