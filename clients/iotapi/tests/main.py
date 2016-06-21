import argparse
import getpass
import os
import shutil
import subprocess
import signal
import sys
import time

from unittest import TextTestRunner, TestSuite
from frontendtests import NullablesTest


def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_path(value):
    if not os.path.isabs(value):
        raise argparse.ArgumentTypeError("%s is an invalid path" % value)
    return value


def main():
    parser = argparse.ArgumentParser(description='IOT Front-End Test', add_help=False)
    parser.add_argument('-n', '--number', type=check_positive_int, nargs='?', default=1000,
                        help='Number of inserts (default: 1000)', metavar='NUMBER')
    parser.add_argument('-f', '--filepath', type=check_path, nargs='?', default='/tmp',
                        help='Temp file location (default: %s)' % '/tmp', metavar='FILE_PATH')
    parser.add_argument('-h', '--host', nargs='?', default='127.0.0.1',
                        help='MonetDB database host (default: 127.0.0.1)', metavar='HOST')
    parser.add_argument('-p', '--port', type=check_positive_int, nargs='?', default=50000,
                        help='Database listening port (default: 50000)', metavar='PORT')
    parser.add_argument('-d', '--database', nargs='?', default='iotdb', help='Database name (default: iotdb)')
    parser.add_argument('-u', '--user', nargs='?', default='monetdb', help='Database user (default: monetdb)')
    parser.add_argument('-?', '--help', action='store_true', help='Display this help')

    try:
        args = vars(parser.parse_args())
    except BaseException as ex:
        print ex
        sys.exit(1)

    if args['help']:
        parser.print_help()
        sys.exit(0)

    test_dir = os.path.join(args['filepath'], 'test_dir')
    shutil.rmtree(test_dir, ignore_errors=True)
    iot_client_log = os.path.join(test_dir, 'iotclient.log')
    iot_api_log = os.path.join(test_dir, 'iotapi.log')

    iot_client_path = os.path.join(test_dir, 'iotclient')
    if not os.path.exists(iot_client_path):
        os.makedirs(iot_client_path)
    iot_api_path = os.path.join(test_dir, 'iotapi')
    if not os.path.exists(iot_api_path):
        os.makedirs(iot_api_path)

    con_pass = getpass.getpass(prompt='Insert password for user ' + args['user'] + ':')
    other_arguments = ["-h", args['host'], "-p", str(args['port']), "-d", args['database'], "-po", "1"]

    head, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))  # get the iotapi path
    iot_client_exec_dir = os.path.join(os.path.split(head)[0], "iotclient", "src", "main.py")
    iot_api_exec_dir = os.path.join(head, "src", "main.py")

    iot_client = subprocess.Popen([sys.executable, iot_client_exec_dir, "-f", iot_client_path,
                                   "-l", iot_client_log] + other_arguments, stdin=subprocess.PIPE)
    iot_api = subprocess.Popen([sys.executable, iot_api_exec_dir, "-f", iot_api_path,
                                "-l", iot_api_log] + other_arguments, stdin=subprocess.PIPE)
    iot_client.stdin.write(con_pass + os.linesep)
    iot_client.stdin.flush()
    iot_api.stdin.write(con_pass + os.linesep)
    iot_api.stdin.flush()

    time.sleep(5)

    if iot_client.returncode is None and iot_api.returncode is None:
        TextTestRunner(verbosity=2).run(TestSuite(tests=[NullablesTest(iot_client_path=iot_client_path,
                                                                       iot_api_path=iot_api_path)]))
    else:
        print 'Processes finished', iot_client.returncode, iot_api.returncode
        shutil.rmtree(test_dir, ignore_errors=True)
        sys.exit(1)

    iot_client.send_signal(signal.SIGINT)
    iot_api.send_signal(signal.SIGINT)
    time.sleep(1)
    shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == '__main__':
    main()
