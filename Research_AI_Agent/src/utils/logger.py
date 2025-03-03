import os
import logging
import uuid
from pythonjsonlogger import jsonlogger
from datetime import datetime


def create_logger(logger_name: str) -> logging.Logger:
    """
    Creates and configures a logger with JSON formatting and a unique ID.

    Args:
        logger_name: The name of the logger to create.

    Returns:
        A configured logger instance with a unique ID and JSON formatting.
    """

    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        json_ensure_ascii=False
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel("DEBUG")

    logger.addHandler(stream_handler)
    logger.setLevel("DEBUG")

    logger.id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4()}"
    return logger
