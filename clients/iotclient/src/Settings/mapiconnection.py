import sys
import pymonetdb
import getpass

from Settings.iotlogger import add_log

Connection = None


def init_monetdb_connection(hostname, port, user_name, database):
    global Connection

    user_password = getpass.getpass(prompt='Insert password for user ' + user_name + ':')

    if user_password == '':
        user_password = 'monetdb'

    try:  # the autocommit is set to true so each statement will be independent
        Connection = pymonetdb.connect(hostname=hostname, port=port, username=user_name, password=user_password,
                                       database=database, autocommit=True)
        log_message = 'User %s connected successfully to database %s' % (user_name, database)
        print >> sys.stdout, log_message
        add_log(20, log_message)
    except BaseException as ex:
        print >> sys.stdout, ex.message
        add_log(50, ex.message)
        sys.exit(1)


def close_monetdb_connection():
    Connection.close()


def mapi_create_stream(schema, stream, columns):
    try:  # create schema if not exists, ignore the error if already exists
        Connection.execute("CREATE SCHEMA " + schema + ";")
    except:
        pass

    try:  # attempt to create te stream table
        Connection.execute(''.join(["CREATE STREAM TABLE ", schema, ".", stream, " (", columns, ");"]))
    except BaseException as ex:
        add_log(40, ex.message)
        pass


def mapi_flush_baskets(schema, stream, baskets):
    try:
        Connection.execute(''.join(["CALL iot.push(\"", schema, "\",\"", stream, "\",\"", baskets, "\");"]))
    except BaseException as ex:
        add_log(40, ex.message)
        pass
