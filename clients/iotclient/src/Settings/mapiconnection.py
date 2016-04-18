import sys
import pymonetdb
import getpass

Connection = None


def init_monetdb_connection(hostname, port, user_name, database):
    global Connection

    user_password = getpass.getpass(prompt='User password:')

    try:
        Connection = pymonetdb.connect(hostname=hostname, port=port, username=user_name, password=user_password,
                                       database=database)
        print >> sys.stdout, 'User %s connection successful to the database %s' % (user_name, database)
    except Exception as ex:
        print >> sys.stderr, ex.message
        sys.exit(1)


def close_monetdb_connection():
    Connection.close()


def mapi_create_stream(schema, stream, columns):
    try:  # create schema if not exists, ignore the error if already exists
        Connection.execute("CREATE SCHEMA " + schema + ";")
    except:
        pass
    sql_str = ''.join(["CREATE STREAM TABLE ", schema, ".", stream, " (", columns, ");"])
    Connection.execute(sql_str)


def mapi_flush_baskets(schema, stream, baskets):
    Connection.execute(''.join(["CALL iot.push(\"", schema, "\",\"", stream, "\",\"", baskets, "\");"]))
