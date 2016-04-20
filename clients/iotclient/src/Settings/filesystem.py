import os
import sys

from iotlogger import add_log
from Utilities.filecreator import create_file_if_not_exists

BASKETS_BASE_DIRECTORY = "baskets"
CONFIG_FILE_DEFAULT_NAME = "config.json"

if sys.platform in ("linux", "linux2", "darwin"):
    filesystem_location = '/etc/iotcollector'
elif sys.platform == "win32":
    filesystem_location = os.path.join(os.path.dirname(__file__), os.pardir)

Baskets_Base_Location = None
Config_File_Location = None
Host_Identifier = None


def set_filesystem_location(new_location):
    global filesystem_location
    filesystem_location = new_location


def init_file_system(host_identifier=None, new_configfile_location=None):
    global Baskets_Base_Location, Config_File_Location, Host_Identifier

    try:
        Baskets_Base_Location = os.path.join(filesystem_location, BASKETS_BASE_DIRECTORY)
        if not os.path.exists(Baskets_Base_Location):
            os.makedirs(Baskets_Base_Location)

        if new_configfile_location is not None:
            Config_File_Location = create_file_if_not_exists(new_configfile_location, hidden=False, init_text='[]')
        else:
            Config_File_Location = create_file_if_not_exists(
                os.path.join(filesystem_location, CONFIG_FILE_DEFAULT_NAME), hidden=False, init_text='[]')

        Host_Identifier = host_identifier
    except (Exception, OSError) as ex:
        print >> sys.stderr, ex
        add_log(50, ex.message)
        sys.exit(1)


def get_baskets_base_location():
    return Baskets_Base_Location


def get_configfile_location():
    return Config_File_Location


def get_host_identifier():
    return Host_Identifier
