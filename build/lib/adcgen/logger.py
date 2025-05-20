from pathlib import Path
import os
import logging
import json


logger: logging.Logger = logging.getLogger("adcgen")
_config_file = "logger_config.json"


def set_log_level(level: str) -> None:
    """Set the level of the adcgen logger."""
    logger.setLevel(level)


def _config_logger() -> None:
    """
    Config the logger.
    The path to a logging configuration JSON file can be provided through the
    'ADCGEN_LOG_CONFIG' environment variable. By default
    'logging_config.json' will be used.
    The level of the adcgen logger can additionally be modified through the
    'ADCGEN_LOG_LEVEL' environment variable. The level will be set after
    reading the config.
    """
    import logging.config

    # load the configuration
    config = os.environ.get("ADCGEN_LOG_CONFIG", None)
    if config is None:
        config = Path(__file__).parent.resolve() / _config_file
    else:
        config = Path(config).resolve()
    if not config.exists:
        raise FileNotFoundError(f"logging config file {config} does not exist")
    config = json.load(open(config, "r"))
    logging.config.dictConfig(config)
    # set the print level
    level = os.environ.get("ADCGEN_LOG_LEVEL", None)
    if level is not None:
        logger.setLevel(level)


class Formatter(logging.Formatter):
    colors = {
        "WARNING": "\033[93m",  # yellow
        "ERROR": "\033[91m",  # red
        "CRITICAL": "\033[95m"  # pink
    }
    reset_color = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Color the log message
        col = self.colors.get(record.levelname, None)
        if col is not None:
            record.msg = f"{col}{record.msg}{self.reset_color}"
        return super().format(record)


class DropErrors(logging.Filter):
    # Only keep debug and info messages
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING
