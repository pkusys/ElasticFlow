import logging


def getLogger(name='logger', level='INFO', mode='w'):
    # create logger
    logger = logging.getLogger(name)

    fh = logging.FileHandler(name + '.log', mode)

    ch = logging.StreamHandler()

    if level == "INFO":
        logger.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    if level == "DEBUG":
        logger.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    if level == "ERROR":
        logger.setLevel(logging.ERROR)
        fh.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger