import argparse
import getpass
import os
import pymonetdb
import sys
import unittest
from datatypestests import TextTest, URLTest, INetTest, UUIDTest, BooleanTest, TinyIntegerTest, SmallIntegerTest, \
    IntegerTest, BigIntegerTest, HugeIntegerTest, RealPointTest, FloatPointTest, DecimalTest, DateTest, \
    TimeWithoutTimezoneTest, TimeWithTimezoneTest, TimestampWithoutTimezoneTest, TimestampWithTimezoneTest


def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_path(value):
    if not os.path.isabs(value):
        raise argparse.ArgumentTypeError("%s is an invalid path" % value)
    return value


def main():
    parser = argparse.ArgumentParser(description='IOT Web Server Tester', add_help=False)
    parser.add_argument('-n', '--number', type=check_positive_int, nargs='?', default=1000,
                        help='Number of inserts (default: 1000)', metavar='NUMBER')
    parser.add_argument('-f', '--filepath', type=check_path, nargs='?', default='/tmp',
                        help='Temp file location (default: %s)' % '/tmp', metavar='FILE_PATH')
    parser.add_argument('-h', '--host', nargs='?', default='127.0.0.1',
                        help='MonetDB database host (default: 127.0.0.1)', metavar='HOST')
    parser.add_argument('-p', '--port', type=check_positive_int, nargs='?', default=50000,
                        help='Database listening port (default: 50000)', metavar='PORT')
    parser.add_argument('-d', '--database', nargs='?', default='iotdb', help='Database name (default: iotdb)')
    parser.add_argument('-u', '--user', nargs='?', default='monetdb', help='Database user (default: monetdb)')
    parser.add_argument('-he', '--help', action='store_true', help='Display this help')

    try:
        args = vars(parser.parse_args())
    except BaseException as ex:
        print ex
        sys.exit(1)

    if args['help']:
        parser.print_help()
        sys.exit(0)

    try:
        con_password = getpass.getpass(prompt='Insert password for user ' + args['user'] + ':')
        mapi_connection = pymonetdb.connect(hostname=args['host'], port=args['port'], username=args['user'],
                                            password=con_password, database=args['database'], autocommit=True)
    except BaseException as ex:
        print ex
        sys.exit(1)

    inserts = args['number']
    temp_path = args['filepath']
    tests = (TextTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             URLTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             INetTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             UUIDTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             BooleanTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             TinyIntegerTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             SmallIntegerTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             IntegerTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             BigIntegerTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             HugeIntegerTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             RealPointTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             FloatPointTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             DecimalTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path, precision=5,
                         scale=1),
             DateTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             TimeWithoutTimezoneTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             TimeWithTimezoneTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             TimestampWithoutTimezoneTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path),
             TimestampWithTimezoneTest(mapi_connection=mapi_connection, number_inserts=inserts, temp_path=temp_path))

    suite = unittest.TestSuite(tests=tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
    mapi_connection.close()

if __name__ == '__main__':
    main()
