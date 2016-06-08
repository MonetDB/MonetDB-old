import argparse
import getpass
import os
import signal
import sys

from IPy import IP
from multiprocessing import Process
from threading import Thread
from Settings.filesystem import init_file_system, DEFAULT_FILESYSTEM
from Settings.iotlogger import init_logging, add_log, DEFAULT_LOGGING
from Settings.mapiconnection import init_monetdb_connection
from Streams.streampolling import init_stream_polling_thread
from WebSockets.websockets import init_websockets

subprocess = None


def signal_handler(signal, frame):
    subprocess.terminate()
    add_log(20, 'Stopped IOT API Server')


def start_process(polling_interval, filesystem_location, sockets_host, sockets_port, connection_hostname, con_port,
                  con_user, con_password, con_database):
    # WARNING The initiation order must be this!!!
    init_file_system(filesystem_location)  # init filesystem
    # init mapi connection
    init_monetdb_connection(connection_hostname, con_port, con_user, con_password, con_database)
    init_stream_polling_thread(polling_interval)  # start polling

    thread1 = Thread(target=init_websockets, args=(sockets_host, sockets_port))
    thread1.start()
    add_log(20, 'Started IOT API Server')
    thread1.join()


def check_path(value):
    if not os.path.isabs(value):
        raise argparse.ArgumentTypeError("%s is an invalid path" % value)
    return value


def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_ipv4_address(value):
    try:
        IP(value)
    except:
        raise argparse.ArgumentTypeError("%s is an invalid IPv4 address" % value)
    return value


def main():
    global subprocess

    parser = argparse.ArgumentParser(description='IOT Web API for MonetDB', epilog="There might exist bugs!",
                                     add_help=False)
    parser.add_argument('-f', '--filesystem', type=check_path, nargs='?', default=DEFAULT_FILESYSTEM,
                        help='Baskets location directory (default: %s)' % DEFAULT_FILESYSTEM, metavar='DIRECTORY')
    parser.add_argument('-l', '--log', type=check_path, nargs='?', default=DEFAULT_LOGGING,
                        help='Logging file location (default: %s)' % DEFAULT_LOGGING, metavar='FILE_PATH')
    parser.add_argument('-po', '--polling', type=check_positive_int, nargs='?', default=60, metavar='POLLING',
                        help='Polling interval in seconds to the database for streams updates (default: 60)')
    parser.add_argument('-sh', '--shost', type=check_ipv4_address, nargs='?', default='0.0.0.0',
                        help='Web API server host (default: 0.0.0.0)', metavar='HOST')
    parser.add_argument('-sp', '--sport', type=check_positive_int, nargs='?', default=8002,
                        help='Web API server port (default: 8002)', metavar='PORT')
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

    con_password = getpass.getpass(prompt='Insert password for user ' + args['user'] + ':')
    init_logging(args['log'])  # init logging context
    subprocess = Process(target=start_process, args=(args['polling'], args['filesystem'], args['shost'], args['sport'],
                                                     args['host'], args['port'], args['user'], con_password,
                                                     args['database']))
    subprocess.start()
    signal.signal(signal.SIGINT, signal_handler)
    subprocess.join()

if __name__ == "__main__":
    main()
