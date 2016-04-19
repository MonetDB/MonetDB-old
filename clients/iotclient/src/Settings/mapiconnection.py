import sys
import pymonetdb
import getpass

Connection = None


def init_monetdb_connection(hostname, port, user_name, database):
    global Connection

    user_password = getpass.getpass(prompt='User password:')

    try:  # the autocommit is set to true so each statement will be independent
        Connection = pymonetdb.connect(hostname=hostname, port=port, username=user_name, password=user_password,
                                       database=database, autocommit=True)
        print >> sys.stdout, 'User %s connected successfully to database %s' % (user_name, database)
    except BaseException as ex:
        print >> sys.stderr, ex.message
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
    except:
        pass


def mapi_flush_baskets(schema, stream, baskets):
    try:
        Connection.execute(''.join(["CALL iot.push(\"", schema, "\",\"", stream, "\",\"", baskets, "\");"]))
    except:
        pass
