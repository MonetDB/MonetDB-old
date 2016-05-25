import os
import sys

from iotlogger import add_log
from Utilities.filecreator import create_file_if_not_exists

Baskets_Location = None
Configfile_Location = None


def init_file_system(new_location=None):
    global Baskets_Location, Configfile_Location

    if new_location is None:
        if sys.platform in ("linux", "linux2", "darwin"):
            filesystem_location = '/etc/iotclient'
        elif sys.platform == "win32":
            filesystem_location = os.path.join(os.path.dirname(__file__), os.pardir)
    else:
        filesystem_location = new_location

    try:
        Baskets_Location = os.path.join(filesystem_location, "baskets")
        if not os.path.exists(Baskets_Location):
            os.makedirs(Baskets_Location)
        Configfile_Location = create_file_if_not_exists(os.path.join(filesystem_location, "config.json"),
                                                        init_text='[]')
    except (Exception, OSError) as ex:
        print ex
        add_log(50, ex)
        sys.exit(1)


def get_baskets_location():
    return Baskets_Location


def get_configfile_location():
    return Configfile_Location
