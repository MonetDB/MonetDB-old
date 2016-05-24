import logging
import sys

import os

Logger = logging.getLogger("IOTAPILog")


def init_logging(new_location):
    global Logger

    if new_location is None:
        if sys.platform in ("linux", "linux2", "darwin"):
            logging_location = '/var/log/iot/iotapi.log'
        elif sys.platform == "win32":
            logging_location = os.path.join(os.path.dirname(__file__), os.pardir, 'iotapi.log')
    else:
        logging_location = new_location

    try:
        logger = logging.getLogger("IOTAPILog")
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
    Logger.log(lvl, message, *args, **kwargs)
