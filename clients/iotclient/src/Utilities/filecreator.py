import ctypes
import os
import sys


# check http://stackoverflow.com/questions/25432139/python-cross-platform-hidden-file


def create_file_if_not_exists(path, hidden=False):
    if hidden and sys.platform in ("linux", "linux2", "darwin"):
        head, tail = os.path.split(path)
        path = os.path.join(head, '.' + tail)
    if not os.path.isfile(path):
        with open(path, 'a'):
            os.utime(path, None)
        if hidden and sys.platform == "win32":
            ctypes.windll.kernel32.SetFileAttributesW(path, 0X02)
    return path


def get_hidden_file_name(path):
    if sys.platform in ("linux", "linux2", "darwin"):
        head, tail = os.path.split(path)
        path = os.path.join(head, '.' + tail)
    return path
