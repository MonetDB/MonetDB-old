import sys
import pymonetdb

from Settings.iotlogger import add_log
from src.Streams.streamscontext import IOTStreams
from src.Streams.jsonschemas import init_create_streams_schema
from src.Streams.streamcreator import creator_add_hugeint_type
from src.Streams.streampolling import polling_add_hugeint_type

Connection = None


def init_monetdb_connection(hostname, port, user_name, user_password, database):
    global Connection

    try:  # the autocommit is set to true so each statement will be independent
        Connection = pymonetdb.connect(hostname=hostname, port=port, username=user_name, password=user_password,
                                       database=database, autocommit=False)
        log_message = 'User %s connected successfully to database %s' % (user_name, database)
        print log_message
        add_log(20, log_message)

        if check_hugeint_type()[0] > 0:
            polling_add_hugeint_type()
            creator_add_hugeint_type()
            init_create_streams_schema(add_hugeint=True)
        else:
            init_create_streams_schema(add_hugeint=False)
    except BaseException as ex:
        print ex
        add_log(50, ex)
        sys.exit(1)


def close_monetdb_connection():
    Connection.close()


def check_hugeint_type():
    Connection.execute("START TRANSACTION")
    cursor = Connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM sys.types WHERE sqlname='hugeint'")
    result = cursor.fetchall()
    Connection.commit()
    return result


def mapi_get_webserver_streams():
    try:
        Connection.execute("START TRANSACTION")
        cursor = Connection.cursor()
        sql_string = """SELECT tables."id", tables."name", schemas."name" as schema, tables."name" as table,
            flushing."flushing", flushing."unit", flushing."interval" FROM (SELECT "id", "name", "schema_id"
            FROM sys.tables WHERE type=4) AS tables INNER JOIN (SELECT "id", "name" FROM sys.schemas) AS schemas ON
            (tables."schema_id"=schemas."id") LEFT JOIN (SELECT "table_id", "flushing", "unit", "interval"
            FROM iot.webserverflushing) AS flushing ON (tables."id"=flushing."table_id")""".replace('\n', ' ')
        cursor.execute(sql_string)
        tables = cursor.fetchall()

        cursor = Connection.cursor()
        sql_string = """SELECT columns."table_id", columns."name" as column, columns."type", columns."type_digits",
            columns."type_scale", columns."default", columns."null", extras."special", extras."validation1",
            extras."validation2" FROM (SELECT "id", "table_id", "name", "type", "type_digits", "type_scale",
            "default", "null" FROM sys.columns) AS columns INNER JOIN (SELECT "id" FROM sys.tables WHERE type=4)
            AS tables ON (tables."id"=columns."table_id") LEFT JOIN (SELECT "column_id", "special", "validation1",
            "validation2" FROM iot.webservervalidation) AS extras ON (columns."id"=extras."column_id")"""\
            .replace('\n', ' ')
        cursor.execute(sql_string)
        columns = cursor.fetchall()

        Connection.commit()
        return tables, columns
    except BaseException as ex:
        add_log(50, ex)
        return [], []


def mapi_create_stream(stream):
    schema = stream.get_schema_name()
    table = stream.get_stream_name()

    try:
        Connection.execute("START TRANSACTION")
        try:  # create schema if not exists, ignore the error if already exists
            Connection.execute("CREATE SCHEMA " + schema)
        except:
            pass
        Connection.execute(''.join(["CREATE STREAM TABLE ", IOTStreams.get_context_entry_name(schema, table), " (",
                                    stream.get_sql_create_statement(), ")"]))

        cursor = Connection.cursor()
        cursor.execute("SELECT id from sys.schemas where name='" + schema + "'")  # get the created table schema_id
        schema_id = cursor.fetchall()[0]
        cursor.execute(''.join(["SELECT id from sys.tables where schema_id=", str(schema_id), " AND name='", stream,
                                "'"]))  # get the created table id
        table_id = int(cursor.fetchall()[0])
        cursor.execute(''.join(["INSERT INTO iot.webserverflushing VALUES (", str(table_id),
                                stream.get_flushing_sql_statement(), ")"]))

        Connection.commit()
        # TODO insert on the tables
        stream.set_table_id(table_id)
    except BaseException as ex:
        add_log(50, ex)
        return None


def mapi_delete_stream(schema, stream, stream_id, columns_ids):
    try:
        Connection.execute("START TRANSACTION")
        Connection.execute("DROP TABLE " + IOTStreams.get_context_entry_name(schema, stream))
        Connection.execute("DELETE FROM iot.webserverflushing WHERE table_id=" + stream_id)
        Connection.execute("DELETE FROM iot.webservervalidation WHERE column_id IN (" + ','.join(columns_ids) + ")")
        return Connection.commit()
    except BaseException as ex:
        add_log(50, ex)
        return None


def mapi_flush_baskets(schema, stream, baskets):
    try:
        Connection.execute("START TRANSACTION")
        Connection.execute(''.join(["CALL iot.basket('", schema, "','", stream, "','", baskets, "')"]))
        return Connection.commit()
    except BaseException as ex:
        add_log(40, ex)
        return 0
