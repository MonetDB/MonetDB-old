import sys
import os

from iotlogger import add_log

Baskets_Location = None


def init_file_system(new_location=None):
    global Baskets_Location

    if new_location is None:
        if sys.platform in ("linux", "linux2", "darwin"):
            new_location = '/etc/iotcollector'
        elif sys.platform == "win32":
            new_location = os.path.join(os.path.dirname(__file__), os.pardir)
    else:
        new_location = new_location

    try:
        Baskets_Location = os.path.join(new_location, "baskets")
        if not os.path.exists(Baskets_Location):
            os.makedirs(Baskets_Location)
    except (Exception, OSError) as ex:
        print >> sys.stdout, ex
        add_log(50, ex)
        sys.exit(1)


def get_baskets_base_location():
    return Baskets_Location
