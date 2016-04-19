import getopt
import sys
import threading

from Settings import filesystem, iotlogger
from Streams import streamscontext
from Flask import restresources
from Flask.app import start_flask_iot_app, start_flask_admin_app
from Settings.mapiconnection import init_monetdb_connection


def main(argv):
    try:
        opts, args = getopt.getopt(argv[1:], 'f:l:c:ih:ip:ah:ap:ch:cp:cd:cu',
                                   ['filesystem=', 'logfile=', 'configfile=', 'ihost=', 'iport=', 'ahost=', 'aport=',
                                    'chostname=', 'cport=', 'cdatabase=', 'cuser='])
    except getopt.GetoptError:
        print >> sys.stderr, "Error while parsing the arguments!"
        sys.exit(1)

    app_host = '0.0.0.0'
    app_port = 8000

    admin_host = '0.0.0.0'
    admin_port = 8001

    connection_hostname = '127.0.0.1'
    connection_port = 50000
    connection_user = 'monetdb'
    connection_database = 'iotdb'
    new_configfile_location = None

    for opt, arg in opts:
        if opt in ('-f', '--filesystem'):
            filesystem.set_filesystem_location(arg)
        elif opt in ('-l', '--logfile'):
            iotlogger.set_logging_location(arg)
        elif opt in ('-c', '--configfile'):
            new_configfile_location = arg

        elif opt in ('-ih', '--ihost'):
            app_host = arg
        elif opt in ('-ip', '--iport'):
            app_port = int(arg)
        elif opt in ('-ah', '--ahost'):
            admin_host = arg
        elif opt in ('-ap', '--aport'):
            admin_port = int(arg)

        elif opt in ('-ch', '--chostname'):
            connection_hostname = arg
        elif opt in ('-cp', '--cport'):
            connection_port = int(arg)
        elif opt in ('-cu', '--cuser'):
            connection_user = arg
        elif opt in ('-cd', '--cdatabase'):
            connection_database = arg

    # WARNING The initiation order must be this!!!
    iotlogger.init_logging()  # init logging context
    filesystem.init_file_system(new_configfile_location)  # init filesystem
    # init mapi connection
    init_monetdb_connection(connection_hostname, connection_port, connection_user, connection_database)
    streamscontext.init_streams_context()  # init streams context
    restresources.init_rest_resources()  # init validators for RESTful requests

    t1 = threading.Thread(target=start_flask_admin_app, args=(admin_host, admin_port))
    t2 = threading.Thread(target=start_flask_iot_app, args=(app_host, app_port))
    t1.start()
    t2.start()
    iotlogger.add_log(20, 'Started IOT Stream Server')
    t1.join()
    t2.join()
    iotlogger.add_log(20, 'Stopped IOT Stream Server')

if __name__ == "__main__":
    main(sys.argv)
