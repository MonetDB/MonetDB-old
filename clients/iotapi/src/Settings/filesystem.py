import sys

import os

from iotlogger import add_log

BASKETS_BASE_DIRECTORY = "baskets"
CONFIG_FILE_DEFAULT_NAME = "config.json"

if sys.platform in ("linux", "linux2", "darwin"):
    filesystem_location = '/etc/iotcollector'
elif sys.platform == "win32":
    filesystem_location = os.path.join(os.path.dirname(__file__), os.pardir)


def set_filesystem_location(new_location):
    global filesystem_location
    filesystem_location = new_location


def init_file_system():
    try:
        if not os.path.exists(filesystem_location):
            os.makedirs(filesystem_location)
    except (Exception, OSError) as ex:
        print >> sys.stdout, ex
        add_log(50, ex)
        sys.exit(1)
