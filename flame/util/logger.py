import logging

try:
    import coloredlogs
except ImportError as e:
    pass


def get_logger(name) -> logging.Logger:

    # create logger
    logger = logging.getLogger(name)
    # logger.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
    # datefmt='%d-%m-%Y %I:%M %p')

    # create console handler and set level to debug
    # if not already created

    if logger.handlers:
        ch = logger.handlers[-1]
    else:
        ch = logging.StreamHandler()
        # ch.setLevel(logging.DEBUG)

    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.propagate = False

    return logger


# app code examples:

# logger.debug('debug message')
# logger.info('info message')
# logger.warn('warn message')
# logger.error('error message')
# logger.critical('critical message')
