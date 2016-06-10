from pymonetdb import connect
from Settings.iotlogger import add_log


def init_monetdb_connection(hostname, port, user_name, user_password, database):
    return connect(hostname=hostname, port=port, username=user_name, password=user_password, database=database,
                   autocommit=False)  # the autocommit is set to true so each statement will be independent


def close_monetdb_connection(connection):
    connection.close()


def check_hugeint_type(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM sys.types WHERE sqlname='hugeint'")
    result = cursor.fetchall()[0][0]
    connection.commit()
    return result > 0


def mapi_get_webserver_streams(connection):
    try:
        cursor = connection.cursor()
        sql_string = """SELECT tables."id", schemas."name" AS schema, tables."name" AS table, extras."base",
            extras."interval", extras."unit" FROM (SELECT "id", "name", "schema_id" FROM sys.tables WHERE type=4)
            AS tables INNER JOIN (SELECT "id", "name" FROM sys.schemas) AS schemas ON (tables."schema_id"=schemas."id")
            LEFT JOIN (SELECT "table_id", "base", "interval", "unit" FROM iot.webserverstreams) AS extras
            ON (tables."id"=extras."table_id")""".replace('\n', ' ')
        cursor.execute(sql_string)
        tables = cursor.fetchall()

        cursor = connection.cursor()
        sql_string = """SELECT columns."id", columns."table_id", columns."name" AS column, columns."type",
            columns."type_digits", columns."type_scale", columns."default", columns."null", extras."special",
            extras."validation1", extras."validation2" FROM (SELECT "id", "table_id", "name", "type", "type_digits",
            "type_scale", "default", "null" FROM sys.columns) AS columns INNER JOIN (SELECT "id" FROM sys.tables
            WHERE type=4) AS tables ON (tables."id"=columns."table_id") LEFT JOIN (SELECT "column_id", "special",
            "validation1", "validation2" FROM iot.webservercolumns) AS extras ON (columns."id"=extras."column_id")"""\
            .replace('\n', ' ')
        cursor.execute(sql_string)
        columns = cursor.fetchall()

        connection.commit()
        return tables, columns
    except BaseException as ex:
        add_log(50, ex)
        raise


def mapi_create_stream(connection, concatenated_name, stream):
    schema = stream.get_schema_name()
    flush_statement = stream.get_webserverstreams_sql_statement()
    columns_dictionary = stream.get_columns_extra_sql_statements()  # dictionary of column_name -> partial SQL statement

    try:
        try:  # create schema if not exists, ignore the error if already exists
            connection.execute("CREATE SCHEMA " + schema)
            connection.commit()
        except:
            pass
        connection.execute(''.join(["CREATE STREAM TABLE ", concatenated_name, " (", stream.get_sql_create_statement(),
                                    ")"]))
        cursor = connection.cursor()
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
        connection.commit()
        stream.set_delete_ids(table_id, colums_ids)
    except BaseException as ex:
        add_log(50, ex)
        raise


def mapi_delete_stream(connection, concatenated_name, stream_id, columns_ids):
    try:
        connection.execute("DROP TABLE " + concatenated_name)
        connection.execute("DELETE FROM iot.webserverstreams WHERE table_id=" + stream_id)
        connection.execute("DELETE FROM iot.webservercolumns WHERE column_id IN (" + columns_ids + ")")
        connection.commit()
    except BaseException as ex:
        add_log(50, ex)
        raise


def mapi_flush_baskets(connection, schema, stream, baskets):
    try:
        connection.execute(''.join(["CALL iot.basket('", schema, "','", stream, "','", baskets, "')"]))
        connection.commit()
    except BaseException as ex:
        add_log(40, ex)
