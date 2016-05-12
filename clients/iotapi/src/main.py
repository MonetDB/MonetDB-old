import getopt
import signal
import sys

from Settings.filesystem import init_file_system, set_filesystem_location
from Settings.iotlogger import init_logging, add_log, set_logging_location
from Settings.mapiconnection import init_monetdb_connection
from Streams.streampolling import init_stream_polling_thread
from WebSockets.websockets import init_websockets, terminate_websockets


def close_sig_handler(signal, frame):
    terminate_websockets()


def main(argv):
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

    # WARNING The initiation order must be this!!!
    init_logging()  # init logging context
    init_file_system()  # init filesystem
    # init mapi connection
    init_monetdb_connection(connection_hostname, connection_port, connection_user, connection_database)
    init_stream_polling_thread(60)  # start polling

    add_log(20, 'Started IOT API Server')
    signal.signal(signal.SIGINT, close_sig_handler)
    init_websockets(sockets_host, sockets_port)
    add_log(20, 'Stopped IOT API Server')


if __name__ == "__main__":
    main(sys.argv)
