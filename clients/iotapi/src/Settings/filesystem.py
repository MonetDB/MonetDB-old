import sys

import os

from iotlogger import add_log

BASKETS_BASE_DIRECTORY = "baskets"

if sys.platform in ("linux", "linux2", "darwin"):
    Filesystem_Location = '/etc/iotcollector'
elif sys.platform == "win32":
    Filesystem_Location = os.path.join(os.path.dirname(__file__), os.pardir)

Baskets_Location = None


def set_filesystem_location(new_location):
    global Filesystem_Location
    Filesystem_Location = new_location


def init_file_system():
    global Baskets_Location

    try:
        Baskets_Location = os.path.join(Filesystem_Location, BASKETS_BASE_DIRECTORY)
        if not os.path.exists(Baskets_Location):
            os.makedirs(Baskets_Location)

        if not os.path.exists(Filesystem_Location):
            os.makedirs(Filesystem_Location)
    except (Exception, OSError) as ex:
        print >> sys.stdout, ex
        add_log(50, ex)
        sys.exit(1)


def get_baskets_base_location():
    return Baskets_Location
