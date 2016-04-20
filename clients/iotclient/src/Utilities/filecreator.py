import ctypes
import os
import sys

# check http://stackoverflow.com/questions/25432139/python-cross-platform-hidden-file


def create_file_if_not_exists(path, hidden=False, init_text=None):
    if hidden and sys.platform in ("linux", "linux2", "darwin"):
        head, tail = os.path.split(path)
        path = os.path.join(head, '.' + tail)

    path_dir = os.path.dirname(path)  # create the directory path if it doesn't exist
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    if not os.path.isfile(path):  # create the file
        with open(path, 'w') as fp:
            os.utime(path, None)
            if init_text is not None:
                fp.write(init_text)
        if hidden and sys.platform == "win32":
            ctypes.windll.kernel32.SetFileAttributesW(path, 0X02)
    return path


def get_hidden_file_name(path):
    if sys.platform in ("linux", "linux2", "darwin"):
        head, tail = os.path.split(path)
        path = os.path.join(head, '.' + tail)
    return path
