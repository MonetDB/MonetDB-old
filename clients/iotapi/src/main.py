import getopt
import getpass
import signal
import sys

from multiprocessing import Process
from threading import Thread
from Settings.filesystem import init_file_system, set_filesystem_location
from Settings.iotlogger import init_logging, add_log, set_logging_location
from Settings.mapiconnection import init_monetdb_connection, close_monetdb_connection
from Streams.streampolling import init_stream_polling_thread
from WebSockets.websockets import init_websockets, terminate_websockets

subprocess = None


def signal_handler(signal, frame):
    subprocess.terminate()


def start_process(sockets_host, sockets_port, connection_hostname, connection_port, connection_user,
                  connection_password, connection_database):
    # WARNING The initiation order must be this!!!
    init_logging()  # init logging context
    init_file_system()  # init filesystem
    # init mapi connection
    init_monetdb_connection(connection_hostname, connection_port, connection_user, connection_password,
                            connection_database)
    init_stream_polling_thread(60)  # start polling

    thread1 = Thread(target=init_websockets, args=(sockets_host, sockets_port))
    thread1.start()
    add_log(20, 'Started IOT API Server')
    thread1.join()

    terminate_websockets()
    close_monetdb_connection()
    add_log(20, 'Stopped IOT API Server')


def main(argv):
    global subprocess

    try:
        opts, args = getopt.getopt(argv[1:], 'f:l:sh:sp:h:p:d:u',
                                   ['filesystem=', 'log=', 'shost=', 'sport=', 'host=', 'port=', 'database=', 'user='])
    except getopt.GetoptError:
        print >> sys.stdout, "Error while parsing the arguments!"
        sys.exit(1)

    sockets_host = '0.0.0.0'
    sockets_port = 8002

    connection_hostname = '127.0.0.1'
    connection_port = 50000
    connection_user = 'monetdb'
    connection_database = 'iotdb'

    for opt, arg in opts:
        if opt in ('-f', '--filesystem'):
            set_filesystem_location(arg)
        elif opt in ('-l', '--log'):
            set_logging_location(arg)
        elif opt in ('-sh', '--shost'):
            sockets_host = arg
        elif opt in ('-sp', '--sport'):
            sockets_port = int(arg)

        elif opt in ('-h', '--host'):
            connection_hostname = arg
        elif opt in ('-p', '--port'):
            connection_port = int(arg)
        elif opt in ('-u', '--user'):
            connection_user = arg
        elif opt in ('-d', '--database'):
            connection_database = arg

    connection_password = getpass.getpass(prompt='Insert password for user ' + connection_user + ':')
    subprocess = Process(target=start_process, args=(sockets_host, sockets_port, connection_hostname, connection_port,
                                                     connection_user, connection_password, connection_database))
    subprocess.start()
    signal.signal(signal.SIGINT, signal_handler)
    subprocess.join()

if __name__ == "__main__":
    main(sys.argv)
