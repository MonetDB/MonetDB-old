import os
import sys

baskets_base_location = None
BASKETS_BASE_DIRECTORY = "baskets"


if sys.platform in ("linux", "linux2", "darwin"):
    filesystem_location = '/etc/iotcollector'
elif sys.platform == "win32":
    filesystem_location = os.path.dirname(os.path.realpath(__file__))


def set_filesystem_location(new_location):
    global filesystem_location

    try:
        if os.path.isdir(new_location):
            filesystem_location = new_location
        else:
            print >> sys.stderr, "The provided filesystem doesn't exist!"
            sys.exit(1)
    except (Exception, OSError) as ex:
        print >> sys.stderr, ex
        sys.exit(1)


def init_file_system():
    global baskets_base_location

    try:
        baskets_base_location = os.path.join(filesystem_location, BASKETS_BASE_DIRECTORY)
        if not os.path.exists(baskets_base_location):
            os.makedirs(baskets_base_location)
    except (Exception, OSError) as ex:
        print >> sys.stderr, ex
        sys.exit(1)


def get_baskets_base_location():
    return baskets_base_location
