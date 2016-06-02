import sys
import pymonetdb

from Settings.iotlogger import add_log

Connection = None


def init_monetdb_connection(hostname, port, user_name, user_password, database):
    global Connection

    try:  # the autocommit is set to true so each statement will be independent
        Connection = pymonetdb.connect(hostname=hostname, port=port, username=user_name, password=user_password,
                                       database=database, autocommit=False)
        log_message = 'User %s connected successfully to database %s' % (user_name, database)
        print log_message
        add_log(20, log_message)
    except BaseException as ex:
        print ex
        add_log(50, ex)
        sys.exit(1)


def close_monetdb_connection():
    Connection.close()


def mapi_get_webserver_streams():
    try:
        Connection.execute("BEGIN TRANSACTION")
        cursor = Connection.cursor()
        sql_string = """SELECT tables."id", tables."name", schemas."name" as schema, tables."name" as table,
            flushing."flushing", flushing."unit", flushing."interval" FROM (SELECT "id", "name", "schema_id"
            FROM sys.tables) AS tables INNER JOIN (SELECT "id", "name" FROM sys.schemas) AS schemas
            ON (tables."schema_id"=schemas."id") INNER JOIN (SELECT "table_id", "flushing", "unit", "interval"
            FROM iot.webserverflushing) AS flushing ON (tables."id"=flushing."table_id")""".replace('\n', ' ')
        cursor.execute(sql_string)
        tables = cursor.fetchall()

        cursor = Connection.cursor()
        sql_string = """SELECT columns."table_id", columns."name" as column, columns."type", columns."type_digits",
            columns."type_scale", columns."default", columns."null", extras."special", extras."validation1",
            extras."validation2" FROM (SELECT "id", "table_id", "name", "type", "type_digits", "type_scale", "default",
            "null" FROM sys.columns) AS columns INNER JOIN (SELECT "column_id", "special", "validation1", "validation2"
            FROM iot.webservervalidation) AS extras ON (columns."id"=extras."column_id")""".replace('\n', ' ')
        cursor.execute(sql_string)
        columns = cursor.fetchall()

        Connection.commit()
        return tables, columns
    except BaseException as ex:
        add_log(50, ex)
        return [], []


def mapi_create_stream(schema, stream, columns):
    try:
        Connection.execute("BEGIN TRANSACTION")
        try:  # create schema if not exists, ignore the error if already exists
            Connection.execute("CREATE SCHEMA " + schema + ";")
        except:
            pass
        Connection.execute(''.join(["CREATE STREAM TABLE ", stream, " (", columns, ")"]))  # TODO concat!!
        # TODO insert on the tables
        return Connection.commit()
    except BaseException as ex:
        add_log(50, ex)
        return 0


def mapi_delete_stream(schema, stream, stream_id, columns_ids):
    try:
        Connection.execute("BEGIN TRANSACTION")
        Connection.execute("DROP TABLE " + stream)  # TODO concat!!
        Connection.execute("DELETE FROM iot.webserverflushing WHERE table_id=" + stream_id)
        Connection.execute("DELETE FROM iot.webservervalidation WHERE column_id IN (" + ','.join(columns_ids) + ")")
        return Connection.commit()
    except BaseException as ex:
        add_log(50, ex)
        return 0


def mapi_flush_baskets(schema, stream, baskets):
    try:
        Connection.execute("BEGIN TRANSACTION")
        Connection.execute(''.join(["CALL iot.basket('", schema, "','", stream, "','", baskets, "');"]))
        return Connection.commit()
    except BaseException as ex:
        add_log(40, ex)
        return 0
