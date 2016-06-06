from itertools import groupby

from Settings.mapiconnection import fetch_streams
from Utilities.customthreading import PeriodicalThread
from datatypes import *
from streams import BaseIOTStream
from streamscontext import Streams_Context

SWITCHER = [{'types': ['clob', 'url'], 'class': 'TextType'},
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
    SWITCHER.append({'types': ['hugeint'], 'class': 'HugeIntegerType'})


def init_stream_polling_thread(interval):
    thread = PeriodicalThread(interval=interval, worker_func=stream_polling)
    thread.start()


# elem[0] is schema. elem[1] is name, elem[2] is column name, elem[3] is type, elem[4] is type_digits
# elem[5] is type_scale elem[6] is default value elem[7] is nullable
def stream_polling():
    current_streams = Streams_Context.get_existing_streams()
    retained_streams = []
    new_streams = {}
    array = fetch_streams()  # TODO check whenever stream's columns are updated

    if array is not None:
        for key, group in groupby(array, lambda x: Streams_Context.get_context_entry_name(x[0], x[1])):
            if key not in current_streams:
                columns = {}

                for elem in group:
                    for entry in SWITCHER:  # allocate the proper type wrapper
                        if elem[3] in entry['types']:
                            reflection_class = globals()[entry['class']]  # import everything from datatypes!!!
                            new_column = reflection_class(*elem[2:])
                            columns[elem[2]] = new_column  # add new column to the dictionary
                            break
                new_streams[key] = BaseIOTStream(schema_name=elem[0], stream_name=elem[1], columns=columns)
            else:
                retained_streams.append(key)

    retained_streams_final = [key for key in current_streams if key in retained_streams]
    Streams_Context.merge_context(retained_streams_final, new_streams)
