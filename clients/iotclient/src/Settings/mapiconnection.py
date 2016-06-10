import pymonetdb
import sys

from Settings.iotlogger import add_log
from Streams.streamscontext import IOTStreams
from Streams.jsonschemas import init_create_streams_schema
from Streams.streamcreator import creator_add_hugeint_type
from Streams.streampolling import polling_add_hugeint_type

Connection = None


def init_monetdb_connection(hostname, port, user_name, user_password, database):
    global Connection

    try:  # the autocommit is set to true so each statement will be independent
        Connection = pymonetdb.connect(hostname=hostname, port=port, username=user_name, password=user_password,
                                       database=database, autocommit=False)
        log_message = 'User %s connected successfully to database %s' % (user_name, database)
        print log_message
        add_log(20, log_message)

        if check_hugeint_type() > 0:
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
    result = cursor.fetchall()[0]
    Connection.commit()
    return result


def mapi_get_webserver_streams():
    try:
        Connection.execute("START TRANSACTION")
        cursor = Connection.cursor()
        sql_string = """SELECT tables."id", schemas."name" AS schema, tables."name" AS table, extras."base",
            extras."interval", extras."unit" FROM (SELECT "id", "name", "schema_id" FROM sys.tables WHERE type=4)
            AS tables INNER JOIN (SELECT "id", "name" FROM sys.schemas) AS schemas ON (tables."schema_id"=schemas."id")
            LEFT JOIN (SELECT "table_id", "has_hostname", "base", "interval", "unit" FROM iot.webserverstreams)
            AS extras ON (tables."id"=extras."table_id")""".replace('\n', ' ')
        cursor.execute(sql_string)
        tables = cursor.fetchall()

        cursor = Connection.cursor()
        sql_string = """SELECT columns."column_id", columns."table_id", columns."name" AS column, columns."type",
            columns."type_digits", columns."type_scale", columns."default", columns."null", extras."special",
            extras."validation1", extras."validation2" FROM (SELECT "id", "table_id", "name", "type", "type_digits",
            "type_scale", "default", "null" FROM sys.columns) AS columns INNER JOIN (SELECT "id" FROM sys.tables
            WHERE type=4) AS tables ON (tables."id"=columns."table_id") LEFT JOIN (SELECT "column_id", "special",
            "validation1", "validation2" FROM iot.webservercolumns) AS extras ON (columns."id"=extras."column_id")"""\
            .replace('\n', ' ')
        cursor.execute(sql_string)
        columns = cursor.fetchall()

        Connection.commit()
        return tables, columns
    except BaseException as ex:
        add_log(50, ex)
        raise


def mapi_create_stream(stream):
    schema = stream.get_schema_name()
    table = stream.get_stream_name()
    flush_statement = stream.get_webserverstreams_sql_statement()
    columns_dictionary = stream.get_columns_extra_sql_statements()  # dictionary of column_name -> partial SQL statement

    try:
        try:  # create schema if not exists, ignore the error if already exists
            Connection.execute("START TRANSACTION")
            Connection.execute("CREATE SCHEMA " + schema)
            Connection.commit()
        except:
            Connection.commit()
        Connection.execute("START TRANSACTION")
        Connection.execute(''.join(["CREATE STREAM TABLE ", IOTStreams.get_context_entry_name(schema, table), " (",
                                    stream.get_sql_create_statement(), ")"]))
        cursor = Connection.cursor()
        cursor.execute("SELECT id FROM sys.schemas WHERE \"name\"='" + schema + "'")
        schema_id = str(cursor.fetchall()[0][0])
        cursor.execute(''.join(["SELECT id FROM sys.tables WHERE schema_id=", schema_id, " AND \"name\"='", stream,
                                "'"]))  # get the created table id
        table_id = str(cursor.fetchall()[0][0])
        cursor.execute(''.join(["INSERT INTO iot.webserverstreams VALUES (", table_id, flush_statement, ")"]))
        cursor.execute("SELECT id, \"name\" FROM sys.columns WHERE table_id=" + table_id)
        columns = cursor.fetchall()

        inserts = []
        colums_ids = ','.join(map(lambda x: str(x[0]), columns))
        for key, value in columns_dictionary.iteritems():
            for entry in columns:  # the imp_timestamp and host identifier are also fetched!!
                if entry[1] == key:  # check for column's name
                    inserts.append(''.join(['(', entry[0], value, ')']))  # append the sql statement
                    break

        cursor.execute("INSERT INTO iot.webservercolumns VALUES " + ','.join(inserts))
        Connection.commit()
        stream.set_delete_ids(table_id, colums_ids)
    except BaseException as ex:
        add_log(50, ex)
        raise


def mapi_delete_stream(schema, stream, stream_id, columns_ids):
    try:
        Connection.execute("START TRANSACTION")
        Connection.execute("DROP TABLE " + IOTStreams.get_context_entry_name(schema, stream))
        Connection.execute("DELETE FROM iot.webserverstreams WHERE table_id=" + stream_id)
        Connection.execute("DELETE FROM iot.webservercolumns WHERE column_id IN (" + columns_ids + ")")
        Connection.commit()
    except BaseException as ex:
        add_log(50, ex)
        raise


def mapi_flush_baskets(schema, stream, baskets):
    try:
        Connection.execute("START TRANSACTION")
        Connection.execute(''.join(["CALL iot.basket('", schema, "','", stream, "','", baskets, "')"]))
        Connection.commit()
    except BaseException as ex:
        add_log(40, ex)
