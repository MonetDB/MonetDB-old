from monetdb.sql import connect
from .iotlogger import add_log


def init_monetdb_connection(hostname, port, user_name, user_password, database):
    return connect(hostname=hostname, port=port, username=user_name, password=user_password, database=database)


def close_monetdb_connection(connection):
    connection.close()


def check_hugeint_type(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM sys.types WHERE sqlname='hugeint'")
    result = cursor.fetchall()[0][0]
    connection.commit()
    return result > 0


def mapi_get_database_streams(connection):
    try:
        cursor = connection.cursor()
        sql_string = """SELECT tables."id", schemas."name" AS schema, tables."name" AS table FROM
             (SELECT "id", "name", "schema_id" FROM sys.tables WHERE type=4) AS tables INNER JOIN (SELECT "id", "name"
             FROM sys.schemas) AS schemas ON (tables."schema_id"=schemas."id")""".replace('\n', ' ')
        cursor.execute(sql_string)
        tables = cursor.fetchall()

        cursor = connection.cursor()
        sql_string = """SELECT columns."table_id", columns."name" AS column, columns."type", columns."type_digits",
            columns."type_scale", columns."default", columns."null" FROM (SELECT "table_id", "name", "type",
            "type_digits", "type_scale", "default", "null", "number" FROM sys.columns) AS columns INNER JOIN
            (SELECT "id" FROM sys.tables WHERE type=4) AS tables ON (tables."id"=columns."table_id")
            ORDER BY columns."table_id", columns."number" """.replace('\n', ' ')
        cursor.execute(sql_string)
        columns = cursor.fetchall()

        connection.commit()
        return tables, columns
    except BaseException as ex:
        add_log(50, ex)
        connection.rollback()
        raise
