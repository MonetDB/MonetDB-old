from collections import OrderedDict, defaultdict
from jsonschema import Draft4Validator, FormatChecker
from datatypes import *
from streams import TupleBasedStream, TimeBasedStream, AutoFlushedStream
from streamscontext import Streams_Context
from Settings.iotlogger import add_log
from Settings.mapiconnection import mapi_get_webserver_streams
from Utilities.customthreading import PeriodicalThread
from jsonschemas import UNBOUNDED_TEXT_TYPE, BOUNDED_TEXT_TYPES, SMALL_INTEGERS_TYPES, HUGE_INTEGER_TYPE, \
    FLOATING_POINT_PRECISION_TYPES, DECIMAL_TYPE, DATE_TYPE, TIME_WITHOUT_TIMEZONE_TYPE, \
    TIME_WITH_TIMEZONE_TYPE_INTERNAL, TIMESTAMP_WITHOUT_TIMEZONE_TYPE, TIMESTAMP_WITH_TIMEZONE_TYPE_INTERNAL, \
    BOOLEAN_TYPE, INET_TYPE, URL_TYPE, UUID_TYPE, INET6_TYPE, MAC_TYPE, REGEX_TYPE, ENUM_TYPE

Switcher = [{'types': [UNBOUNDED_TEXT_TYPE], 'class': 'TextType'},
            {'types': BOUNDED_TEXT_TYPES, 'class': 'LimitedTextType'},
            {'types': SMALL_INTEGERS_TYPES, 'class': 'SmallIntegerType'},
            {'types': FLOATING_POINT_PRECISION_TYPES, 'class': 'FloatType'},
            {'types': [DECIMAL_TYPE], 'class': 'DecimalType'},
            {'types': [BOOLEAN_TYPE], 'class': 'BooleanType'},
            {'types': [DATE_TYPE], 'class': 'DateType'},
            {'types': [TIME_WITHOUT_TIMEZONE_TYPE], 'class': 'TimeWithoutTimeZoneType'},
            {'types': [TIME_WITH_TIMEZONE_TYPE_INTERNAL], 'class': 'TimeWithTimeZoneType'},
            {'types': [TIMESTAMP_WITHOUT_TIMEZONE_TYPE], 'class': 'TimestampWithoutTimeZoneType'},
            {'types': [TIMESTAMP_WITH_TIMEZONE_TYPE_INTERNAL], 'class': 'TimestampWithTimeZoneType'},
            {'types': [URL_TYPE], 'class': 'URLType'},
            {'types': [INET_TYPE], 'class': 'INetType'},
            {'types': [UUID_TYPE], 'class': 'UUIDType'}]

INTEGER_TYPES = SMALL_INTEGERS_TYPES
FLOATING_POINT_TYPES = FLOATING_POINT_PRECISION_TYPES + DECIMAL_TYPE
DATETIME_TYPES = [DATE_TYPE, TIME_WITHOUT_TIMEZONE_TYPE, TIME_WITH_TIMEZONE_TYPE_INTERNAL,
                  TIMESTAMP_WITHOUT_TIMEZONE_TYPE, TIMESTAMP_WITH_TIMEZONE_TYPE_INTERNAL]


def polling_add_hugeint_type():
    global INTEGER_TYPES
    Switcher.append({'types': [HUGE_INTEGER_TYPE], 'class': 'HugeIntegerType'})
    INTEGER_TYPES += HUGE_INTEGER_TYPE


def init_stream_polling_thread(interval):
    thread = PeriodicalThread(interval=interval, worker_func=stream_polling)
    thread.start()


def stream_polling():
    streams = OrderedDict()  # dictionary of schema_name + '.' + stream_name -> DataCellStream
    # for tables [0] -> id, [1] -> schema, [2] -> name, [3] -> has_hostname, [4] -> base, [5] -> interval, [6] -> unit
    # FLUSHING_STREAMS = {1: 'TupleBasedStream', 2: 'TimeBasedStream', 3: 'AutoFlushedStream'}  Important!!!
    # SPECIAL_TYPES = {1: 'MACType', 2: 'RegexType', 3: 'EnumType', 4: 'INetSixType'}

    # for columns [0] -> id, [1] -> table_id, [2] -> name, [3] -> type, [4] -> type_digits, [5] -> type_scale,
    # [6] -> default, [7] -> is_null, [8] -> special, [9] -> validation1, [10] -> validation2
    tables, columns = mapi_get_webserver_streams()

    grouped_columns = defaultdict(list)
    for entry in columns:
        grouped_columns[entry[1]].append(entry)

    for entry in tables:
        retrieved_columns = grouped_columns[entry[0]]
        built_columns = {}  # dictionary of name -> data_types
        for column in retrieved_columns:
            kwargs_dic = {'name': column[2], 'default': column[6], 'nullable': column[7]}

            next_switch = column[8]
            if next_switch == 1:  # MACType
                kwargs_dic['type'] = MAC_TYPE
            elif next_switch == 2:  # RegexType
                kwargs_dic['type'] = REGEX_TYPE
                kwargs_dic['regex'] = column[9]
            elif next_switch == 3:  # EnumType
                kwargs_dic['type'] = ENUM_TYPE
                kwargs_dic['values'] = column[9].split(ENUM_TYPE_SEPARATOR)
            elif next_switch == 4:  # INetSixType
                kwargs_dic['type'] = INET6_TYPE
            else:
                next_switch = column[3]
                kwargs_dic['type'] = next_switch
                if next_switch in BOUNDED_TEXT_TYPES:
                    kwargs_dic['limit'] = column[4]
                elif next_switch in INTEGER_TYPES:
                    if column[6] is not None:
                        kwargs_dic['default'] = int(column[6])
                    if column[10] is not None:
                        kwargs_dic['minimum'] = int(column[10])
                    if column[11] is not None:
                        kwargs_dic['maximum'] = int(column[11])
                elif next_switch in FLOATING_POINT_TYPES:
                    if column[6] is not None:
                        kwargs_dic['default'] = float(column[6])
                    if column[10] is not None:
                        kwargs_dic['minimum'] = float(column[10])
                    if column[11] is not None:
                        kwargs_dic['maximum'] = float(column[11])
                    if next_switch == DECIMAL_TYPE:
                        kwargs_dic['precision'] = column[4]
                        kwargs_dic['scale'] = column[5]
                elif next_switch in DATETIME_TYPES:
                    if column[10] is not None:
                        kwargs_dic['minimum'] = column[10]
                    if column[11] is not None:
                        kwargs_dic['maximum'] = column[11]

            for variable in Switcher:  # allocate the proper type wrapper
                if kwargs_dic['type'] in variable['types']:
                    try:
                        reflection_class = globals()[variable['class']]  # import everything from datatypes!!!
                        built_columns[kwargs_dic['name']] = reflection_class(**column)  # pass the json entry as kwargs
                    except BaseException as ex:
                        add_log(50, ex)
                    break

        properties = OrderedDict()
        req_fields = []

        for key, value in built_columns.iteritems():
            value.add_json_schema_entry(properties)  # append new properties entry
            if not value.is_nullable() and value.get_default_value() is None:  # check if it's required or not
                req_fields.append(key)

        json_schema = Draft4Validator({
            "title": "JSON schema to validate inserts in stream " + entry[1] + '.' + entry[2],
            "description": "Validate the inserted properties", "$schema": "http://json-schema.org/draft-04/schema#",
            "id": "http://monetdb.com/schemas/iot_create.json", "type": "array", "minItems": 1,
            "items": {"type": "object", "properties": properties, "required": req_fields, "additionalProperties": False}
        }, format_checker=FormatChecker())

        colums_ids = ','.join(map(lambda x: str(x[0]), retrieved_columns))

        if entry[4] == 1:  # TupleBasedStream
            new_stream = TupleBasedStream(schema_name=entry[1], stream_name=entry[2], columns=built_columns,
                                          validation_schema=json_schema, has_hostname=entry[3], table_id=str(entry[0]),
                                          columns_ids=colums_ids, interval=int(entry[5]))
        elif entry[4] == 1:  # TimeBasedStream
            new_stream = TimeBasedStream(schema_name=entry[1], stream_name=entry[2], columns=built_columns,
                                         validation_schema=json_schema, has_hostname=entry[3], table_id=str(entry[0]),
                                         columns_ids=colums_ids, interval=int(entry[5]), time_unit=entry[6])
        else:  # AutoFlushedStream
            new_stream = AutoFlushedStream(schema_name=entry[1], stream_name=entry[2], columns=built_columns,
                                           validation_schema=json_schema, has_hostname=entry[3], table_id=str(entry[0]),
                                           columns_ids=colums_ids)

        next_concatened_name = Streams_Context.get_context_entry_name(entry[1], entry[2])
        streams[next_concatened_name] = new_stream

    Streams_Context.update_context(streams)
