import sys

import os
from Utilities.filecreator import create_file_if_not_exists

from iotlogger import add_log

BASKETS_BASE_DIRECTORY = "baskets"
CONFIG_FILE_DEFAULT_NAME = "config.json"

if sys.platform in ("linux", "linux2", "darwin"):
    Filesystem_Location = '/etc/iotcollector'
elif sys.platform == "win32":
    Filesystem_Location = os.path.join(os.path.dirname(__file__), os.pardir)

Baskets_Location = None
Config_File_Location = None
Host_Identifier = None


def set_filesystem_location(new_location):
    global Filesystem_Location
    Filesystem_Location = new_location


def init_file_system(host_identifier=None, new_configfile_location=None):
    global Baskets_Location, Config_File_Location, Host_Identifier

    try:
        Baskets_Location = os.path.join(Filesystem_Location, BASKETS_BASE_DIRECTORY)
        if not os.path.exists(Baskets_Location):
            os.makedirs(Baskets_Location)

        if new_configfile_location is not None:
            Config_File_Location = create_file_if_not_exists(new_configfile_location, hidden=False, init_text='[]')
        else:
            Config_File_Location = create_file_if_not_exists(
                os.path.join(Filesystem_Location, CONFIG_FILE_DEFAULT_NAME), hidden=False, init_text='[]')

        Host_Identifier = host_identifier
    except (Exception, OSError) as ex:
        print >> sys.stdout, ex
        add_log(50, ex)
        sys.exit(1)


def get_baskets_base_location():
    return Baskets_Location


def get_configfile_location():
    return Config_File_Location


def get_host_identifier():
    return Host_Identifier
