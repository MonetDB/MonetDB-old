import sys
import pymonetdb

from Settings.iotlogger import add_log

Connection = None


def init_monetdb_connection(hostname, port, user_name, user_password, database):
    global Connection

    try:  # the autocommit is set to true so each statement will be independent
        Connection = pymonetdb.connect(hostname=hostname, port=port, username=user_name, password=user_password,
                                       database=database, autocommit=True)
        log_message = 'User %s connected successfully to database %s' % (user_name, database)
        print >> sys.stdout, log_message
        add_log(20, log_message)
    except BaseException as ex:
        print ex
        add_log(50, ex)
        sys.exit(1)


def close_monetdb_connection():
    Connection.close()

""""
def mapi_fetch_all_streams():
    try:
        cursor = Connection.cursor()
        sql = ''.join(['SELECT storage."schema", storage."table", storage."column", storage."type", storage',
                       '."typewidth" FROM (SELECT "schema", "table", "column", "type", "typewidth" FROM sys.storage)',
                       ' AS storage INNER JOIN (SELECT "name" FROM sys.tables WHERE type=4) AS tables ON'
                       ' (storage."table"=tables."name") INNER JOIN (SELECT "name" FROM sys.schemas)',
                       ' AS schemas ON (storage."schema"=schemas."name");'])
        cursor.execute(sql)
        return cursor.fetchall()
    except BaseException as ex:
        add_log(50, ex)
        raise


def mapi_check_stream_data(schema, stream):
    try:
        cursor = Connection.cursor()
        sql = ''.join(['SELECT storage."schema", storage."table", storage."column", storage."type", storage.',
                       '"typewidth" FROM (SELECT "schema", "table", "column", "type", "typewidth" FROM sys.storage)',
                       ' AS storage INNER JOIN (SELECT "name" FROM sys.tables WHERE type=4 AND "name"="', stream,
                       '") AS tables ON (storage."table"=tables."name") INNER JOIN (SELECT "name" FROM sys.schemas',
                       ' WHERE "name"="',schema, '") AS schemas ON (storage."schema"=schemas."name");'])
        cursor.execute(sql)
        return cursor.fetchall()
    except BaseException as ex:
        add_log(50, ex)
"""""


def mapi_create_stream(schema, stream, columns):
    try:  # create schema if not exists, ignore the error if already exists
        Connection.execute("CREATE SCHEMA " + schema + ";")
    except:
        pass

    try:  # attempt to create the stream table
        Connection.execute("SET SCHEMA " + schema + ";")
        Connection.execute(''.join(["CREATE STREAM TABLE ", stream, " (", columns, ");"]))
    except BaseException as ex:
        add_log(40, ex)


def mapi_flush_baskets(schema, stream, baskets):
    try:
        Connection.execute("SET SCHEMA iot;")
    except:
        pass

    try:
        Connection.execute(''.join(["CALL iot.basket('", schema, "','", stream, "','", baskets, "');"]))
    except BaseException as ex:
        add_log(40, ex)
