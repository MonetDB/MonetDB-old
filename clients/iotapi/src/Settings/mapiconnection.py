import getpass
import sys

import pymonetdb
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
        Connection.execute("SET SCHEMA iot;")
        log_message = 'User %s connected successfully to database %s' % (user_name, database)
        print >> sys.stdout, log_message
        add_log(20, log_message)
    except BaseException as ex:
        print >> sys.stdout, ex
        add_log(50, ex)
        sys.exit(1)


def close_monetdb_connection():
    Connection.close()


def fetch_streams():
    try:  # TODO paginate results?
        cursor = Connection.cursor()
        sql_string = """
          SELECT storage."schema", storage."table", storage."column", storage."type", storage."location",
          storage."typewidth"
          FROM (SELECT "schema", "table", "column", "type" FROM sys.storage) AS storage
          INNER JOIN (SELECT "name" FROM sys.tables WHERE type=4) AS tables ON (storage."table"=tables."name")
          INNER JOIN (SELECT "name" FROM sys.schemas) AS schemas ON (storage."schema"=schemas."name");
        """.replace('\n', ' ')
        cursor.execute(sql_string)
        return cursor.fetchall()
    except BaseException as ex:
        print >> sys.stdout, ex
        add_log(50, ex)
