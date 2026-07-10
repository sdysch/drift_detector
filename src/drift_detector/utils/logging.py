import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Set up root logger with timestamped format.

    Args:
        level: Logging level for the root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
