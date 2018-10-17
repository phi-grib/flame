import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import appdirs

try:
    import coloredlogs  # add color in the future with this
except ImportError as e:
    pass


def get_log_file() -> Path:
    log_filename_path = appdirs.user_log_dir(appname='flame')
    log_filename_path = Path(log_filename_path)
    if not log_filename_path.exists():
        log_filename_path.mkdir(parents=True)
    log_filename = log_filename_path / 'flame.log'

    # check if exists to not erase current file
    if not log_filename.exists():
        log_filename.touch()
    return log_filename


def get_logger(name) -> logging.Logger:
    """ inits a logger and adds the handlers.
    If the logger is already created doesn't adds new handlers
    since those are set at interpreter level and already exists."""
    # Create the log file
    log_file = get_log_file()
    # create logger
    logger = logging.getLogger(name)
    # set base logger level to DEBUG but fine tu the handlers
    # for custom level
    logger.setLevel(logging.DEBUG)

    # create formatter fdor file handler (more explicit)
    file_formatter = logging.Formatter(
        '[%(asctime)s] - %(name)s - %(levelname)s - %(message)s'
    )

    # formater for stream handler (less info)
    stdout_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    # datefmt='%d-%m-%Y %I:%M %p')

    # create console and file handler
    # if not already created
    if not logger.handlers:
        # 512 Kb file log
        fh = RotatingFileHandler(log_file, maxBytes=1_024_000, backupCount=5)
        fh.setLevel('DEBUG')
        # add formatter to handler
        fh.setFormatter(file_formatter)
        # add handler to logger
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel('INFO')
        ch.setFormatter(stdout_formatter)
        logger.addHandler(ch)

    # if there already handlers just return the logger
    # since its already configured
    else:
        return logger
    # logger.propagate = False
    return logger


# app code examples:

# logger.debug('debug message')
# logger.info('info message')
# logger.warn('warn message')
# logger.error('error message')
# logger.critical('critical message')
