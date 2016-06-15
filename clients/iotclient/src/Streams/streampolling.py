from collections import OrderedDict, defaultdict
from json import dumps
from jsonschema import Draft4Validator, FormatChecker
from .datatypes import TextType, LimitedTextType, SmallIntegerType, HugeIntegerType, FloatType, DecimalType, DateType,\
    TimeType, TimestampType, IntervalType, BooleanType, INetType, INetSixType, MACType, URLType, UUIDType, RegexType,\
    EnumType, ENUM_TYPE_SEPARATOR
from .jsonschemas import UNBOUNDED_TEXT_TYPE, BOUNDED_TEXT_TYPES, SMALL_INTEGERS_TYPES, HUGE_INTEGER_TYPE,\
    FLOATING_POINT_PRECISION_TYPES, DECIMAL_TYPE, DATE_TYPE, TIME_WITHOUT_TIMEZONE_TYPE, MONTH_INTERVAL_TYPE,\
    TIME_WITH_TIMEZONE_TYPE_INTERNAL, TIMESTAMP_WITHOUT_TIMEZONE_TYPE, TIMESTAMP_WITH_TIMEZONE_TYPE_INTERNAL,\
    BOOLEAN_TYPE, INET_TYPE, URL_TYPE, UUID_TYPE, INET6_TYPE, MAC_TYPE, REGEX_TYPE, ENUM_TYPE, SECOND_INTERVAL_TYPE
from .streams import TupleBasedStream, TimeBasedStream, AutoFlushedStream, IMPLICIT_TIMESTAMP_COLUMN_NAME,\
    HOST_IDENTIFIER_COLUMN_NAME
from .streamscontext import get_streams_context
from Settings.iotlogger import add_log
from Settings.mapiconnection import init_monetdb_connection, mapi_get_database_streams
from Utilities.customthreading import PeriodicalThread

Switcher = [{'types': [UNBOUNDED_TEXT_TYPE], 'class': TextType},
            {'types': BOUNDED_TEXT_TYPES, 'class': LimitedTextType},
            {'types': SMALL_INTEGERS_TYPES, 'class': SmallIntegerType},
            {'types': FLOATING_POINT_PRECISION_TYPES, 'class': FloatType},
            {'types': [DECIMAL_TYPE], 'class': DecimalType},
            {'types': [BOOLEAN_TYPE], 'class': BooleanType},
            {'types': [DATE_TYPE], 'class': DateType},
            {'types': [TIME_WITHOUT_TIMEZONE_TYPE, TIME_WITH_TIMEZONE_TYPE_INTERNAL], 'class': TimeType},
            {'types': [TIMESTAMP_WITHOUT_TIMEZONE_TYPE, TIMESTAMP_WITH_TIMEZONE_TYPE_INTERNAL], 'class': TimestampType},
            {'types': [SECOND_INTERVAL_TYPE, MONTH_INTERVAL_TYPE], 'class': IntervalType},
            {'types': [URL_TYPE], 'class': URLType},
            {'types': [INET_TYPE], 'class': INetType},
            {'types': [UUID_TYPE], 'class': UUIDType},
            {'types': [INET6_TYPE], 'class': INetSixType},
            {'types': [MAC_TYPE], 'class': MACType},
            {'types': [REGEX_TYPE], 'class': RegexType},
            {'types': [ENUM_TYPE], 'class': EnumType}]

INTEGER_AND_INTERVAL_TYPES = SMALL_INTEGERS_TYPES + [SECOND_INTERVAL_TYPE] + [MONTH_INTERVAL_TYPE]
FLOATING_POINT_TYPES = FLOATING_POINT_PRECISION_TYPES + [DECIMAL_TYPE]
DATETIME_TYPES = [DATE_TYPE, TIME_WITHOUT_TIMEZONE_TYPE, TIME_WITH_TIMEZONE_TYPE_INTERNAL,
                  TIMESTAMP_WITHOUT_TIMEZONE_TYPE, TIMESTAMP_WITH_TIMEZONE_TYPE_INTERNAL]

INTERVALS_DICTIONARY = {1: "interval year", 2: "interval year to month", 3: "interval month", 4: "interval day",
                        5: "interval day to hour", 6: "interval day to minute", 7: "interval day to second",
                        8: "interval hour", 9: "interval hour to minute", 10: "interval hour to second",
                        11: "interval minute", 12: "interval minute to second", 13: "interval second"}


def polling_add_hugeint_type():
    global INTEGER_AND_INTERVAL_TYPES
    INTEGER_AND_INTERVAL_TYPES += [HUGE_INTEGER_TYPE]
    Switcher.append({'types': [HUGE_INTEGER_TYPE], 'class': HugeIntegerType})


def init_stream_polling_thread(interval, connection, con_hostname, con_port, con_user, con_password, con_database):
    PeriodicalThread(interval=interval, worker_func=stream_polling,
                     argument=[connection, con_hostname, con_port, con_user, con_password, con_database]).start()


def stream_polling(argument):
    # FLUSHING_STREAMS = {1: 'TupleBasedStream', 2: 'TimeBasedStream', 3: 'AutoFlushedStream'}
    # SPECIAL_TYPES = {1: 'MACType', 2: 'RegexType', 3: 'EnumType', 4: 'INetSixType'}
    # for tables: [0] -> id, [1] -> schema, [2] -> name, [3] -> base, [4] -> interval, [5] -> unit
    # for columns: [0] -> id, [1] -> table_id, [2] -> name, [3] -> type, [4] -> type_digits, [5] -> type_scale,
    # [6] -> default_value, [7] -> is_null, [8] -> special, [9] -> validation1, [10] -> validation2

    retained_streams = []
    new_streams = {}
    context = get_streams_context()

    try:
        tables, columns = mapi_get_database_streams(argument[0])  # TODO check whenever stream's columns are updated
        grouped_columns = defaultdict(list)  # group the columns to the respective tables
        for entry in columns:
            grouped_columns[entry[1]].append(entry)
    except BaseException as ex:
        add_log(50, ex)
        return

    try:
        current_streams = context.get_existing_streams()  # array of concatenated names
        for entry in tables:
            try:
                next_concatenated_name = context.get_context_entry_name(entry[1], entry[2])
                if next_concatenated_name not in current_streams:
                    retrieved_columns = grouped_columns[entry[0]]
                    built_columns = OrderedDict()  # dictionary of name -> data_types
                    errors = OrderedDict()
                    has_timestamp = False
                    has_hostname = False
                    for column in retrieved_columns:
                        if column[2] == IMPLICIT_TIMESTAMP_COLUMN_NAME:
                            has_timestamp = True
                        elif column[2] == HOST_IDENTIFIER_COLUMN_NAME:
                            has_hostname = True
                        else:
                            if column[6] is not None:
                                default_value = column[6][1:-1]
                            else:
                                default_value = None
                            kwargs_dic = {'name': column[2], 'default': default_value, 'nullable': column[7]}
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
                                elif next_switch in INTEGER_AND_INTERVAL_TYPES:
                                    if default_value is not None:
                                        kwargs_dic['default'] = int(default_value)
                                    if column[9] is not None:
                                        kwargs_dic['minimum'] = int(column[9])
                                    if column[10] is not None:
                                        kwargs_dic['maximum'] = int(column[10])
                                    if next_switch in (SECOND_INTERVAL_TYPE, MONTH_INTERVAL_TYPE):
                                        kwargs_dic['type'] = INTERVALS_DICTIONARY.get(column[4])
                                elif next_switch in FLOATING_POINT_TYPES:
                                    if default_value is not None:
                                        kwargs_dic['default'] = float(default_value)
                                    if column[9] is not None:
                                        kwargs_dic['minimum'] = float(column[9])
                                    if column[10] is not None:
                                        kwargs_dic['maximum'] = float(column[10])
                                    if next_switch == DECIMAL_TYPE:
                                        kwargs_dic['precision'] = column[4]
                                        kwargs_dic['scale'] = column[5]
                                elif next_switch in DATETIME_TYPES:
                                    if default_value is not None:
                                        kwargs_dic['default'] = default_value
                                    if column[9] is not None:
                                        kwargs_dic['minimum'] = column[9]
                                    if column[10] is not None:
                                        kwargs_dic['maximum'] = column[10]

                            valid_type = False
                            for variable in Switcher:  # allocate the proper type wrapper
                                if kwargs_dic['type'] in variable['types']:
                                    reflection_class = variable['class']()
                                    built_columns[kwargs_dic['name']] = reflection_class(**kwargs_dic)
                                    valid_type = True
                                    break
                            if not valid_type:
                                errors[kwargs_dic['name']] = 'Not valid type: ' + kwargs_dic['type']
                                break
                    if len(errors) > 0:
                        add_log(40, dumps(errors))
                        continue
                    properties = OrderedDict()
                    req_fields = []

                    for key, value in built_columns.iteritems():
                        value.add_json_schema_entry(properties)  # append new properties entry
                        if not value.is_nullable() and value.get_default_value() is None:  # check if it's required
                            req_fields.append(key)

                    json_schema = Draft4Validator({
                        "title": "JSON schema to validate inserts in stream " + entry[1] + '.' + entry[2],
                        "description": "Validate the inserted properties",
                        "$schema": "http://json-schema.org/draft-04/schema#",
                        "id": "http://monetdb.com/schemas/iot_create.json", "type": "array", "minItems": 1,
                        "items": {"type": "object", "properties": properties, "required": req_fields,
                                  "additionalProperties": False}
                    }, format_checker=FormatChecker())

                    columns_ids = ','.join(map(lambda x: str(x[0]), retrieved_columns))
                    mapi_connection = init_monetdb_connection(argument[1], argument[2], argument[3], argument[4],
                                                              argument[5])

                    if entry[3] == 1:  # TupleBasedStream
                        new_stream = TupleBasedStream(schema_name=entry[1], stream_name=entry[2], columns=built_columns,
                                                      validation_schema=json_schema, has_timestamp=has_timestamp,
                                                      has_hostname=has_hostname, connection=mapi_connection,
                                                      table_id=str(entry[0]), columns_ids=columns_ids,
                                                      interval=int(entry[4]))
                    elif entry[3] == 2:  # TimeBasedStream
                        new_stream = TimeBasedStream(schema_name=entry[1], stream_name=entry[2], columns=built_columns,
                                                     validation_schema=json_schema, has_timestamp=has_timestamp,
                                                     has_hostname=has_hostname, connection=mapi_connection,
                                                     table_id=str(entry[0]), columns_ids=columns_ids,
                                                     interval=int(entry[4]), time_unit=entry[5])
                    else:  # AutoFlushedStream
                        new_stream = AutoFlushedStream(schema_name=entry[1], stream_name=entry[2],
                                                       columns=built_columns, validation_schema=json_schema,
                                                       has_timestamp=has_timestamp, has_hostname=has_hostname,
                                                       connection=mapi_connection, table_id=str(entry[0]),
                                                       columns_ids=columns_ids)
                    new_streams[next_concatenated_name] = new_stream
                else:
                    retained_streams.append(next_concatenated_name)
            except BaseException as ex:
                add_log(50, ex)
                continue
        context.merge_context(retained_streams, new_streams)
    except BaseException as ex:
        context.release_lock()
        add_log(50, ex)
