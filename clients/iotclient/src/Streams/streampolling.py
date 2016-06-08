from itertools import groupby
from Settings.mapiconnection import mapi_get_webserver_streams
from Utilities.customthreading import PeriodicalThread
from datatypes import *
from streamscontext import Streams_Context

Switcher = [{'types': ['clob'], 'class': 'TextType'},
            {'types': ['url'], 'class': 'URLType'},
            {'types': ['char', 'varchar'], 'class': 'LimitedTextType'},
            {'types': ['tinyint', 'smallint', 'int', 'wrd', 'bigint'], 'class': 'SmallIntegerType'},
            {'types': ['real', 'double'], 'class': 'FloatType'},
            {'types': ['decimal'], 'class': 'DecimalType'},
            {'types': ['boolean'], 'class': 'BooleanType'},
            {'types': ['date'], 'class': 'DateType'},
            {'types': ['time'], 'class': 'TimeWithoutTimeZoneType'},
            {'types': ['timetz'], 'class': 'TimeWithTimeZoneType'},
            {'types': ['timestamp'], 'class': 'TimestampWithoutTimeZoneType'},
            {'types': ['timestamptz'], 'class': 'TimestampWithTimeZoneType'},
            {'types': ['inet'], 'class': 'INetType'},
            {'types': ['uuid'], 'class': 'UUIDType'}]

FLUSHING_STREAMS = {1: 'TupleBasedStream', 2: 'TimeBasedStream', 3: 'AutoFlushedStream'}  # Use Try Get
SPECIAL_TYPES = {1: 'MACType', 2: 'RegexType', 3: 'EnumType', 4: 'INetSixType'}  # Use Try Get


def polling_add_hugeint_type():
    Switcher.append({'types': ['hugeint'], 'class': 'HugeIntegerType'})


def init_stream_polling_thread(interval):
    thread = PeriodicalThread(interval=interval, worker_func=stream_polling)
    thread.start()


def stream_polling():
    current_streams = Streams_Context.get_existing_streams()
    retained_streams = []
    new_streams = {}
    tables, columns = mapi_get_webserver_streams()  # TODO check whenever stream's columns are updated

    # TODO finish stream polling

    retained_streams_final = [key for key in current_streams if key in retained_streams]
    Streams_Context.merge_context(retained_streams_final, new_streams)
