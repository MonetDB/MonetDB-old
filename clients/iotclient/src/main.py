import getopt
import signal
import sys
import time

from multiprocessing import Process
from threading import Thread
from uuid import getnode as get_mac
from Flask.app import start_flask_iot_app, start_flask_admin_app
from Flask.restresources import init_rest_resources
from Settings.filesystem import init_file_system, set_filesystem_location
from Settings.iotlogger import init_logging, add_log, set_logging_location
from Settings.mapiconnection import init_monetdb_connection
from Streams.streamscontext import init_streams_context
from Streams.streams import init_streams_hosts

subprocess = None


def signal_handler(signal, frame):
    subprocess.terminate()


def start_process(admin_host, admin_port, app_host, app_port):
    thread1 = Thread(target=start_flask_admin_app, args=(admin_host, admin_port))
    thread2 = Thread(target=start_flask_iot_app, args=(app_host, app_port))
    thread1.start()
    time.sleep(1)  # problem while handling Flask's loggers, so it is used this sleep
    thread2.start()
    thread1.join()
    thread2.join()


def main(argv):
    global subprocess

    try:
        opts, args = getopt.getopt(argv[1:], 'f:l:c:ui:in:ih:ip:ah:ap:h:p:d:u',
                                   ['filesystem=', 'log=', 'config=', 'useidentifier', 'name='
                                    'ihost=', 'iport=', 'ahost=', 'aport=',
                                    'host=', 'port=', 'database=', 'user='])
    except getopt.GetoptError:
        print >> sys.stdout, "Error while parsing the arguments!"
        sys.exit(1)

    app_host = '0.0.0.0'
    app_port = 8000

    admin_host = '127.0.0.1'
    admin_port = 8001

    connection_hostname = '127.0.0.1'
    connection_port = 50000
    connection_user = 'monetdb'
    connection_database = 'iotdb'

    new_configfile_location = None
    use_host_identifier = False
    host_identifier = None

    for opt, arg in opts:
        if opt in ('-f', '--filesystem'):
            set_filesystem_location(arg)
        elif opt in ('-l', '--log'):
            set_logging_location(arg)
        elif opt in ('-c', '--config'):
            new_configfile_location = arg
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
            connection_hostname = arg
        elif opt in ('-p', '--port'):
            connection_port = int(arg)
        elif opt in ('-u', '--user'):
            connection_user = arg
        elif opt in ('-d', '--database'):
            connection_database = arg

    if use_host_identifier and host_identifier is None:  # get the machine MAC address as default identifier
        host_identifier = ':'.join(("%012X" % get_mac())[i:i + 2] for i in range(0, 12, 2))
    if not use_host_identifier:  # in case of the user sets the host_identifier but not the use_host_identifier flag
        host_identifier = None

    # WARNING The initiation order must be this!!!
    init_logging()  # init logging context
    init_file_system(host_identifier, new_configfile_location)  # init filesystem
    init_streams_hosts()  # init hostname column for streams
    # init mapi connection
    init_monetdb_connection(connection_hostname, connection_port, connection_user, connection_database)
    init_streams_context()  # init streams context
    init_rest_resources()  # init validators for RESTful requests

    subprocess = Process(target=start_process, args=(admin_host, admin_port, app_host, app_port))
    subprocess.start()
    add_log(20, 'Started IOT Stream Server')
    signal.signal(signal.SIGINT, signal_handler)
    subprocess.join()
    add_log(20, 'Stopped IOT Stream Server')

if __name__ == "__main__":
    main(sys.argv)
