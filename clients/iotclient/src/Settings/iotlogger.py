import logging
import sys
import os

Logger = logging.getLogger("IOTServerLog")

if sys.platform in ("linux", "linux2", "darwin"):
    DEFAULT_LOGGING = '/var/log/iot/iotserver.log'
elif sys.platform == "win32":
    DEFAULT_LOGGING = os.path.join(os.path.dirname(__file__), os.pardir, 'iotserver.log')


def init_logging(logging_location):
    try:
        logging_path = os.path.dirname(logging_location)
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
        log_handler = logging.FileHandler(logging_location, mode='a+')
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        Logger.addHandler(log_handler)
    except (Exception, OSError) as ex:
        print ex
        sys.exit(1)


def add_log(lvl, message, *args, **kwargs):
    Logger.log(lvl, message, *args, **kwargs)
