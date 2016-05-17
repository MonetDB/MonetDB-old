from itertools import groupby

from Settings.mapiconnection import fetch_streams
from Utilities.customthreading import PeriodicalThread

from streamscontext import Streams_context, DataCellStream

SWITCHER = [{'types': ['clob', 'char', 'varchar', 'url'], 'class': 'TextType'},
            {'types': ['tinyint', 'smallint', 'int', 'bigint'], 'class': 'SmallIntegerType'},
            {'types': ['hugeint'], 'class': 'HugeIntegerType'},
            {'types': ['real', 'double'], 'class': 'FloatType'},
            {'types': ['decimal'], 'class': 'DecimalType'},
            {'types': ['boolean'], 'class': 'BooleanType'},
            {'types': ['date'], 'class': 'DateType'},
            {'types': ['time'], 'class': 'TimeType'},
            {'types': ['timestamp'], 'class': 'TimestampType'},
            {'types': ['inet'], 'class': 'INetType'},
            {'types': ['uuid'], 'class': 'UUIDType'}]


def init_stream_polling_thread(interval):
    thread = PeriodicalThread(interval=interval, worker_func=stream_polling)
    thread.start()


# elem[0] is schema. elem[1] is name, elem[2] is column name, elem[3] is type, elem[4] is location, elem[5] is typewidth
def stream_polling():
    current_streams = Streams_context.get_existing_streams()
    retained_streams = []
    new_streams = {}
    array = fetch_streams()  # TODO check whenever stream's columns are updated

    for key, group in groupby(array, lambda x: Streams_context.get_context_entry_name(x[0], x[1])):
        if key not in current_streams:
            columns = {}

            for elem in group:
                reflection_class = globals()[elem[3]]  # import everything from datatypes!!!
                kwargs = {'name': elem[2], 'type': elem[3], 'location': elem[4], 'typewidth': elem[5]}
                new_column = reflection_class(kwargs)
                columns[elem[2]] = new_column
                new_streams[key] = DataCellStream(key, columns)
        else:
            retained_streams.append(key)

    retained_streams_final = [key for key in current_streams if key in retained_streams]
    Streams_context.merge_context(retained_streams_final, new_streams)
