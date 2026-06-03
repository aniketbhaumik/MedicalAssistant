import logging

def setup_logger(name="MedicalAssistantLogger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler() # print the logs to console
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("[%(asctime)s] - [%(levelname)s] --- [%(message)s]") 
    ch.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)
    
    return logger


logger = setup_logger()

logger.info("RAG process started")
logger.debug("debugging info")
logger.error("failed to load")
logger.warning("warning message")
logger.critical("critical error")


