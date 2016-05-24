import getopt
import getpass
import signal
import sys

from multiprocessing import Process
from threading import Thread
from Settings.filesystem import init_file_system
from Settings.iotlogger import init_logging, add_log
from Settings.mapiconnection import init_monetdb_connection, close_monetdb_connection
from Streams.streampolling import init_stream_polling_thread
from WebSockets.websockets import init_websockets, terminate_websockets

subprocess = None


def signal_handler(signal, frame):
    subprocess.terminate()


def start_process(filesystem_location, logging_location, sockets_host, sockets_port, connection_hostname, con_port,
                  con_user, con_password, con_database):
    # WARNING The initiation order must be this!!!
    init_logging(logging_location)  # init logging context
    init_file_system(filesystem_location)  # init filesystem
    # init mapi connection
    init_monetdb_connection(connection_hostname, con_port, con_user, con_password, con_database)
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

    filesystem_location = None
    logging_location = None
    sockets_host = '0.0.0.0'
    sockets_port = 8002

    con_hostname = '127.0.0.1'
    con_port = 50000
    con_user = 'monetdb'
    con_database = 'iotdb'

    for opt, arg in opts:
        if opt in ('-f', '--filesystem'):
            filesystem_location = arg
        elif opt in ('-l', '--log'):
            logging_location = arg
        elif opt in ('-sh', '--shost'):
            sockets_host = arg
        elif opt in ('-sp', '--sport'):
            sockets_port = int(arg)

        elif opt in ('-h', '--host'):
            con_hostname = arg
        elif opt in ('-p', '--port'):
            con_port = int(arg)
        elif opt in ('-u', '--user'):
            con_user = arg
        elif opt in ('-d', '--database'):
            con_database = arg

    con_password = getpass.getpass(prompt='Insert password for user ' + con_user + ':')
    subprocess = Process(target=start_process, args=(filesystem_location, logging_location, sockets_host, sockets_port,
                                                     con_hostname, con_port, con_user, con_password, con_database))
    subprocess.start()
    signal.signal(signal.SIGINT, signal_handler)
    subprocess.join()

if __name__ == "__main__":
    main(sys.argv)
