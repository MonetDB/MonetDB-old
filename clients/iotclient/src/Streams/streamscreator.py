from collections import OrderedDict
from json import dumps
from jsonschema import Draft4Validator, FormatChecker
from .datatypes import TextType, LimitedTextType, SmallIntegerType, FloatType, DecimalType, DateType, EnumType,\
    TimeType, TimestampType, IntervalType, BooleanType, INetType, INetSixType, MACType, URLType, UUIDType, RegexType
from .jsonschemas import UNBOUNDED_TEXT_INPUTS, BOUNDED_TEXT_INPUTS, SMALL_INTEGERS_INPUTS, HUGE_INTEGER_TYPE,\
    FLOATING_POINT_PRECISION_INPUTS, DECIMAL_INPUTS, DATE_TYPE, TIME_INPUTS, TIMESTAMP_INPUTS, INTERVAL_INPUTS,\
    BOOLEAN_INPUTS, INET_TYPE, INET6_TYPE, MAC_TYPE, URL_TYPE, UUID_TYPE, REGEX_TYPE, ENUM_TYPE,\
    TIMED_FLUSH_IDENTIFIER, TUPLE_FLUSH_IDENTIFIER
from .streams import TupleBasedStream, TimeBasedStream, AutoFlushedStream, IMPLICIT_TIMESTAMP_COLUMN_NAME,\
    HOST_IDENTIFIER_COLUMN_NAME
from Settings.mapiconnection import init_monetdb_connection

Switcher = [{'types': UNBOUNDED_TEXT_INPUTS, 'class': TextType},
            {'types': BOUNDED_TEXT_INPUTS, 'class': LimitedTextType},
            {'types': SMALL_INTEGERS_INPUTS, 'class': SmallIntegerType},
            {'types': FLOATING_POINT_PRECISION_INPUTS, 'class': FloatType},
            {'types': DECIMAL_INPUTS, 'class': DecimalType},
            {'types': [DATE_TYPE], 'class': DateType},
            {'types': TIME_INPUTS, 'class': TimeType},
            {'types': TIMESTAMP_INPUTS, 'class': TimestampType},
            {'types': INTERVAL_INPUTS, 'class': IntervalType},
            {'types': BOOLEAN_INPUTS, 'class': BooleanType},
            {'types': [INET_TYPE], 'class': INetType},
            {'types': [INET6_TYPE], 'class': INetSixType},
            {'types': [MAC_TYPE], 'class': MACType},
            {'types': [URL_TYPE], 'class': URLType},
            {'types': [UUID_TYPE], 'class': UUIDType},
            {'types': [REGEX_TYPE], 'class': RegexType},
            {'types': [ENUM_TYPE], 'class': EnumType}]


def creator_add_hugeint_type():
    Switcher.append({'types': [HUGE_INTEGER_TYPE], 'class': 'HugeIntegerType'})


def validate_schema_and_create_stream(schema, con_hostname, con_port, con_user, con_password, con_database):
    validated_columns = OrderedDict()  # dictionary of name -> data_types
    errors = OrderedDict()

    for column in schema['columns']:  # create the data types dictionary
        next_type = column['type']
        next_name = column['name']

        if next_name in validated_columns:
            errors[next_name] = 'The column ' + next_name + ' is duplicated!'
            continue

        if next_name in (IMPLICIT_TIMESTAMP_COLUMN_NAME, HOST_IDENTIFIER_COLUMN_NAME):
            errors[next_name] = 'The column name ' + next_name + ' is reserved!'
            continue

        for entry in Switcher:  # allocate the proper type wrapper
            if next_type in entry['types']:
                try:
                    reflection_class = entry['class']()  # use reflection
                    validated_columns[next_name] = reflection_class(**column)  # pass the json entry as kwargs
                except BaseException as ex:
                    errors[next_name] = ex
                break
    if errors:
        raise Exception(message=dumps(errors))  # dictionary of name ->  error message

    properties = OrderedDict()
    req_fields = []

    for key, value in validated_columns.iteritems():
        value.add_json_schema_entry(properties)  # append new properties entry
        if not value.is_nullable() and value.get_default_value() is None:  # check if it's required or not
            req_fields.append(key)

    json_schema = Draft4Validator({
        "title": "JSON schema to validate inserts in stream " + schema['schema'] + '.' + schema['stream'],
        "description": "Validate the inserted properties", "$schema": "http://json-schema.org/draft-04/schema#",
        "id": "http://monetdb.com/schemas/iot_create.json", "type": "array", "minItems": 1,
        "items": {"type": "object", "properties": properties, "required": req_fields, "additionalProperties": False}
    }, format_checker=FormatChecker())

    flushing_object = schema['flushing']  # check the flush method
    mapi_connection = init_monetdb_connection(con_hostname, con_port, con_user, con_password, con_database)

    if flushing_object['base'] == TIMED_FLUSH_IDENTIFIER:
        res = TimeBasedStream(schema_name=schema['schema'], stream_name=schema['stream'], columns=validated_columns,
                              validation_schema=json_schema, has_timestamp=True, has_hostname=schema['hostname'],
                              connection=mapi_connection, table_id="", columns_ids="",
                              interval=flushing_object['interval'], time_unit=flushing_object['unit'])
    elif flushing_object['base'] == TUPLE_FLUSH_IDENTIFIER:
        res = TupleBasedStream(schema_name=schema['schema'], stream_name=schema['stream'], columns=validated_columns,
                               validation_schema=json_schema, has_timestamp=True, has_hostname=schema['hostname'],
                               connection=mapi_connection, table_id="", columns_ids="",
                               interval=flushing_object['interval'])
    else:
        res = AutoFlushedStream(schema_name=schema['schema'], stream_name=schema['stream'], columns=validated_columns,
                                validation_schema=json_schema, has_timestamp=True, has_hostname=schema['hostname'],
                                connection=mapi_connection, table_id="", columns_ids="")
    return res
