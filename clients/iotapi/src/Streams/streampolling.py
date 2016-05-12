from itertools import groupby

from Settings.mapiconnection import fetch_streams
from Utilities.customthreading import PeriodicalThread

from streamscontext import Streams_context, DataCellStream

SWITCHER = [{'types': ['clob', 'char', 'varchar'], 'class': 'TextType'},
            {'types': ['boolean'], 'class': 'BooleanType'},
            {'types': ['tinyint', 'smallint', 'int', 'bigint'], 'class': 'SmallIntegerType'},
            {'types': ['hugeint'], 'class': 'HugeIntegerType'},
            {'types': ['real', 'double'], 'class': 'FloatType'},
            {'types': ['decimal'], 'class': 'DecimalType'},
            {'types': ['date'], 'class': 'DateType'},
            {'types': ['time'], 'class': 'TimeType'},
            {'types': ['timestamp'], 'class': 'TimestampType'}]


def init_stream_polling_thread(interval):
    thread = PeriodicalThread(interval=interval, worker_func=stream_polling)
    thread.start()


# elem[0] is schema. elem[1] is name, elem[2] is column name, elem[3] is type, elem[4] is location, elem[5] is typewidth
def stream_polling():
    array = fetch_streams()
    for key, group in groupby(array, lambda x: x[0] + '.' + x[1]):
        if not Streams_context.is_stream_in_context(key):
            columns = {}

            for elem in group:
                reflection_class = globals()[elem[3]]  # import everything from datatypes!!!
                kwargs = {'name': elem[2], 'type': elem[3], 'location': elem[4], 'typewidth': elem[5]}
                new_column = reflection_class(kwargs)
                columns[elem[2]] = new_column

            Streams_context.add_stream(key, DataCellStream(elem[0], elem[1], columns))
