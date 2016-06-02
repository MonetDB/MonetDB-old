import os
import sys

from iotlogger import add_log
from Utilities.filecreator import create_file_if_not_exists

Baskets_Location = None

if sys.platform in ("linux", "linux2", "darwin"):
    DEFAULT_FILESYSTEM = '/etc/iotserver'
elif sys.platform == "win32":
    DEFAULT_FILESYSTEM = os.path.join(os.path.dirname(__file__), os.pardir)


def init_file_system(filesystem_location):
    global Baskets_Location

    try:
        Baskets_Location = os.path.join(filesystem_location, "baskets")
        if not os.path.exists(Baskets_Location):
            os.makedirs(Baskets_Location)
    except (Exception, OSError) as ex:
        print ex
        add_log(50, ex)
        sys.exit(1)


def get_baskets_location():
    return Baskets_Location
