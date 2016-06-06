import collections
import json

from jsonschema import Draft4Validator, FormatChecker
from datatypes import *
from jsonschemas import UNBOUNDED_TEXT_TYPES, BOUNDED_TEXT_TYPES, SMALL_INTEGERS, HUGE_INTEGER_TYPE,\
    FLOATING_POINT_PRECISION_TYPES, DECIMAL_TYPES, DATE_TYPE, TIME_WITHOUT_TIMEZONE_TYPE, BOOLEAN_TYPE, INET_TYPE,\
    TIME_WITH_TIMEZONE_TYPE, TIMESTAMP_WITHOUT_TIMEZONE_TYPE, TIMESTAMP_WITH_TIMEZONE_TYPE, INET6_TYPE, MAC_TYPE,\
    URL_TYPE, UUID_TYPE, REGEX_TYPE, ENUM_TYPE, TIMED_FLUSH_IDENTIFIER, TUPLE_FLUSH_IDENTIFIER
from streams import TupleBasedStream, TimeBasedStream, AutoFlushedStream
from src.Settings.mapiconnection import mapi_create_stream

SWITCHER = [{'types': UNBOUNDED_TEXT_TYPES, 'class': 'TextType'},
            {'types': BOUNDED_TEXT_TYPES, 'class': 'LimitedTextType'},
            {'types': SMALL_INTEGERS, 'class': 'SmallIntegerType'},
            {'types': FLOATING_POINT_PRECISION_TYPES, 'class': 'FloatType'},
            {'types': DECIMAL_TYPES, 'class': 'DecimalType'},
            {'types': DATE_TYPE, 'class': 'DateType'},
            {'types': TIME_WITHOUT_TIMEZONE_TYPE, 'class': 'TimeWithoutTimeZoneType'},
            {'types': TIME_WITH_TIMEZONE_TYPE, 'class': 'TimeWithTimeZoneType'},
            {'types': TIMESTAMP_WITHOUT_TIMEZONE_TYPE, 'class': 'TimestampWithoutTimeZoneType'},
            {'types': TIMESTAMP_WITH_TIMEZONE_TYPE, 'class': 'TimestampWithTimeZoneType'},
            {'types': BOOLEAN_TYPE, 'class': 'BooleanType'},
            {'types': INET_TYPE, 'class': 'INetType'},
            {'types': INET6_TYPE, 'class': 'INetSixType'},
            {'types': MAC_TYPE, 'class': 'MACType'},
            {'types': URL_TYPE, 'class': 'URLType'},
            {'types': UUID_TYPE, 'class': 'UUIDType'},
            {'types': REGEX_TYPE, 'class': 'RegexType'},
            {'types': ENUM_TYPE, 'class': 'EnumType'}]


def creator_add_hugeint_type():
    SWITCHER.append({'types': HUGE_INTEGER_TYPE, 'class': 'HugeIntegerType'})


def validate_schema_and_create_stream(schema):
    validated_columns = collections.OrderedDict()  # dictionary of name -> data_types
    errors = collections.OrderedDict()

    for column in schema['columns']:  # create the data types dictionary
        next_type = column['type']
        next_name = column['name']

        if next_name in validated_columns:
            errors[next_name] = 'The column ' + next_name + ' is duplicated!'
            continue

        for entry in SWITCHER:  # allocate the proper type wrapper
            if next_type in entry['types']:
                try:
                    reflection_class = globals()[entry['class']]  # import everything from datatypes!!!
                    validated_columns[next_name] = reflection_class(**column)  # pass the json entry as kwargs
                except Exception as ex:
                    errors[next_name] = ex
                break
    if errors:
        raise Exception(message=json.dumps(errors))  # dictionary of name ->  error message

    properties = collections.OrderedDict()
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
    if flushing_object['base'] == TIMED_FLUSH_IDENTIFIER:
        res = TimeBasedStream(schema_name=schema['schema'], stream_name=schema['stream'], columns=validated_columns,
                              validation_schema=json_schema, interval=int(flushing_object['interval']),
                              time_unit=flushing_object['unit'])
    elif flushing_object['base'] == TUPLE_FLUSH_IDENTIFIER:
        res = TupleBasedStream(schema_name=schema['schema'], stream_name=schema['stream'], columns=validated_columns,
                               validation_schema=json_schema, interval=int(flushing_object['number']))
    else:
        res = AutoFlushedStream(schema_name=schema['schema'], stream_name=schema['stream'], columns=validated_columns,
                                validation_schema=json_schema)
    mapi_create_stream(res) # send the CREATE STREAM TABLE statement when the create json request is made
    return res
