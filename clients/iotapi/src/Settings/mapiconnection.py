import sys

from pymonetdb import connect
from ..Settings.iotlogger import add_log

Connection = None


def init_monetdb_connection(hostname, port, user_name, user_password, database):
    global Connection

    try:  # the autocommit is set to true so each statement will be independent
        Connection = connect(hostname=hostname, port=port, username=user_name, password=user_password,
                             database=database, autocommit=True)
        log_message = 'User %s connected successfully to database %s' % (user_name, database)
        print log_message
        add_log(20, log_message)
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
    result = cursor.fetchall()[0][0]
    Connection.commit()
    return result


def fetch_streams():
    try:
        cursor = Connection.cursor()
        sql_string = """SELECT schemas."name" as schema, tables."name" as table, columns."name" as column,
             columns."type", columns."type_digits", columns."type_scale", columns."default", columns."null" FROM
             (SELECT "id", "name", "schema_id" FROM sys.tables WHERE type=4) AS tables INNER JOIN (SELECT "id", "name"
             FROM sys.schemas) AS schemas ON (tables."schema_id"=schemas."id") INNER JOIN (SELECT "table_id", "name",
             "type", "type_digits", "type_scale", "default", "null" FROM sys.columns) AS columns ON
             (columns."table_id"=tables."id")""".replace('\n', ' ')
        cursor.execute(sql_string)
        return cursor.fetchall()
    except BaseException as ex:
        add_log(50, ex)
        raise
