import getopt
import getpass
import signal
import sys
import time
import os

from multiprocessing import Process
from threading import Thread
from uuid import getnode as get_mac
from Flask.app import start_flask_iot_app, start_flask_admin_app
from Flask.restresources import init_rest_resources
from Settings.filesystem import init_file_system
from Settings.iotlogger import init_logging, add_log
from Settings.mapiconnection import init_monetdb_connection, close_monetdb_connection
from Streams.streams import init_streams_hosts
from Streams.streamscontext import init_streams_context

subprocess = None


def signal_handler(signal, frame):
    subprocess.terminate()


def start_process(filesystem_location, logging_location, use_host_identifier, host_identifier, admin_host, admin_port,
                  app_host, app_port, con_hostname, con_port, con_user, con_password, con_database):
    # WARNING The initiation order must be this!!!
    init_logging(logging_location)  # init logging context
    init_file_system(filesystem_location)  # init filesystem
    init_streams_hosts(use_host_identifier, host_identifier)  # init hostname column for streams
    # init mapi connection
    init_monetdb_connection(con_hostname, con_port, con_user, con_password, con_database)
    init_streams_context()  # init streams context
    init_rest_resources()  # init validators for RESTful requests

    thread1 = Thread(target=start_flask_admin_app, args=(admin_host, admin_port))
    thread2 = Thread(target=start_flask_iot_app, args=(app_host, app_port))
    thread1.start()
    time.sleep(1)  # problem while handling Flask's loggers, so it is used this sleep
    thread2.start()
    add_log(20, 'Started IOT Stream Server')
    thread1.join()
    thread2.join()
    close_monetdb_connection()
    add_log(20, 'Stopped IOT Stream Server')


def main(argv):
    global subprocess

    try:
        opts, args = getopt.getopt(argv[1:], 'f:l:ui:in:ih:ip:ah:ap:h:p:d:u',
                                   ['filesystem=', 'log=', 'useidentifier', 'name=', 'ihost=', 'iport=', 'ahost=',
                                    'aport=', 'host=', 'port=', 'database=', 'user='])
    except getopt.GetoptError:
        print 'Error while parsing the arguments!'
        sys.exit(1)

    filesystem_location = None
    logging_location = None
    use_host_identifier = False
    host_identifier = None

    app_host = '0.0.0.0'
    app_port = 8000
    admin_host = '127.0.0.1'
    admin_port = 8001

    con_hostname = '127.0.0.1'
    con_port = 50000
    con_user = 'monetdb'
    con_database = 'iotdb'

    for opt, arg in opts:
        if opt in ('-f', '--filesystem'):
            filesystem_location = arg
        elif opt in ('-l', '--log'):
            logging_location = arg
        elif opt in ('-ui', '--useidentifier'):
            use_host_identifier = True
        elif opt in ('-in', '--name'):
            host_identifier = arg

        elif opt in ('-ih', '--ihost'):
            app_host = arg
        elif opt in ('-ip', '--iport'):
            app_port = int(arg)
        elif opt in ('-ah', '--ahost'):
            admin_host = arg
        elif opt in ('-ap', '--aport'):
            admin_port = int(arg)

        elif opt in ('-h', '--host'):
            con_hostname = arg
        elif opt in ('-p', '--port'):
            con_port = int(arg)
        elif opt in ('-u', '--user'):
            con_user = arg
        elif opt in ('-d', '--database'):
            con_database = arg

    if host_identifier is None:  # get the machine MAC address as default identifier
        host_identifier = ':'.join(("%012X" % get_mac())[i:i + 2] for i in range(0, 12, 2))
        print 'Using host identifier: ', host_identifier, os.linesep

    con_password = getpass.getpass(prompt='Insert password for user ' + con_user + ':')
    subprocess = Process(target=start_process, args=(filesystem_location, logging_location, use_host_identifier,
                                                     host_identifier, admin_host, admin_port, app_host, app_port,
                                                     con_hostname, con_port, con_user, con_password, con_database))
    subprocess.start()
    signal.signal(signal.SIGINT, signal_handler)
    subprocess.join()

if __name__ == "__main__":
    main(sys.argv)
