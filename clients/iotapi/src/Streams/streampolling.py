from collections import OrderedDict, defaultdict
from json import dumps
from .datatypes import TextType, LimitedTextType, SmallIntegerType, HugeIntegerType, FloatType, DecimalType,\
    BooleanType, DateType, TimeType, TimestampType, INetType, UUIDType, IntervalType
from .streams import IOTStream
from .streamscontext import Streams_Context
from Settings.iotlogger import add_log
from Settings.mapiconnection import mapi_get_database_streams
from Utilities.customthreading import PeriodicalThread

Switcher = [{'types': ['clob', 'url'], 'class': TextType},
            {'types': ['char', 'varchar'], 'class': LimitedTextType},
            {'types': ['tinyint', 'smallint', 'int', 'bigint'], 'class': SmallIntegerType},
            {'types': ['real', 'double'], 'class': FloatType},
            {'types': ['decimal'], 'class': DecimalType},
            {'types': ['boolean'], 'class': BooleanType},
            {'types': ['date'], 'class': DateType},
            {'types': ['time', 'timetz'], 'class': TimeType},
            {'types': ['timestamp', 'timestamptz'], 'class': TimestampType},
            {'types': ['sec_interval', 'month_interval'], 'class': IntervalType},
            {'types': ['inet'], 'class': INetType},
            {'types': ['uuid'], 'class': UUIDType}]


def polling_add_hugeint_type():
    Switcher.append({'types': ['hugeint'], 'class': HugeIntegerType})


def init_stream_polling_thread(interval, connection):
    PeriodicalThread(interval=interval, worker_func=stream_polling, argument=[connection]).start()


def stream_polling(arguments):
    # for tables: [0] -> id, [1] -> schema, [2] -> name
    # for columns: [0] -> table_id, [1] -> name, [2] -> type, [3] -> type_digits, [4] -> type_scale,
    # [5] -> default_value, [6] -> is_null

    retained_streams = []
    new_streams = {}

    try:
        tables, columns = mapi_get_database_streams(arguments[0])  # TODO check whenever stream's columns are updated
        grouped_columns = defaultdict(list)  # group the columns to the respective tables
        for entry in columns:
            grouped_columns[entry[0]].append(entry)
    except BaseException as ex:
        add_log(50, ex)
        return

    current_streams = Streams_Context.get_existing_streams()

    for entry in tables:
        try:
            next_concatenated_name = Streams_Context.get_context_entry_name(entry[1], entry[2])
            if next_concatenated_name not in current_streams:
                retrieved_columns = grouped_columns[entry[0]]
                built_columns = OrderedDict()  # dictionary of name -> data_types
                errors = OrderedDict()
                for column in retrieved_columns:
                    if column[5] is not None:
                        default_value = column[5][1:-1]
                    else:
                        default_value = None
                    kwargs_dic = {'name': column[1], 'type': column[2], 'default': default_value, 'nullable': column[6],
                                  'digits': column[3], 'scale': column[4]}

                    valid_type = False
                    for variable in Switcher:  # allocate the proper type wrapper
                        if column[2] in variable['types']:
                            built_columns[column[1]] = variable['class'](**kwargs_dic)
                            valid_type = True
                            break
                    if not valid_type:
                        errors[column[1]] = 'Not valid type: ' + column[2]
                        break
                if len(errors) > 0:
                    add_log(40, dumps(errors))
                    continue

                new_streams[next_concatenated_name] = IOTStream(schema_name=entry[1], stream_name=entry[2],
                                                                columns=built_columns)
            else:
                retained_streams.append(next_concatenated_name)
        except BaseException as ex:
            add_log(50, ex)
            continue
    Streams_Context.merge_context(retained_streams, new_streams)
