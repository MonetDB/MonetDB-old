import logging
import sys

import os

if sys.platform in ("linux", "linux2", "darwin"):
    logging_location = '/var/log/iot/iotapi.log'
elif sys.platform == "win32":
    logging_location = os.path.join(os.path.dirname(__file__), os.pardir, 'iotapi.log')

logger = logging.getLogger("IOTAPILog")


def set_logging_location(new_location):
    global logging_location
    logging_location = new_location


def init_logging():
    global logger
    try:
        logger = logging.getLogger("IOTLog")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        logging_path = os.path.dirname(logging_location)
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
        log_handler = logging.FileHandler(logging_location, mode='a+')
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
    except (Exception, OSError) as ex:
        print >> sys.stdout, ex
        sys.exit(1)


def add_log(lvl, message, *args, **kwargs):
    logger.log(lvl, message, *args, **kwargs)
