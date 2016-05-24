import collections

from jsonschema import Draft4Validator, FormatChecker
from datatypes import *
from jsonschemas import UNBOUNDED_TEXT_TYPES, BOUNDED_TEXT_TYPES, SMALL_INTEGERS, HUGE_INTEGER, \
    FLOATING_POINT_PRECISION_TYPES, DECIMAL_TYPES, DATE_TYPE, TIME_TYPE, TIMESTAMP_TYPE, BOOLEAN_TYPE, INET_TYPE, \
    INET6_TYPE, MAC_TYPE, URL_TYPE, UUID_TYPE, REGEX_TYPE, ENUM_TYPE, TIMED_FLUSH_IDENTIFIER, TUPLE_FLUSH_IDENTIFIER
from streams import TupleBasedStream, TimeBasedStream, AutoFlushedStream


class ColumnsValidationException(Exception):
    def __init__(self, error_messages):
        super(ColumnsValidationException, self).__init__()
        self.message = error_messages  # dictionary of name ->  error message

SWITCHER = [{'types': UNBOUNDED_TEXT_TYPES, 'class': 'TextType'},
            {'types': BOUNDED_TEXT_TYPES, 'class': 'LimitedTextType'},
            {'types': SMALL_INTEGERS, 'class': 'SmallIntegerType'},
            {'types': HUGE_INTEGER, 'class': 'HugeIntegerType'},
            {'types': FLOATING_POINT_PRECISION_TYPES, 'class': 'FloatType'},
            {'types': DECIMAL_TYPES, 'class': 'DecimalType'},
            {'types': DATE_TYPE, 'class': 'DateType'},
            {'types': TIME_TYPE, 'class': 'TimeType'},
            {'types': TIMESTAMP_TYPE, 'class': 'TimestampType'},
            {'types': BOOLEAN_TYPE, 'class': 'BooleanType'},
            {'types': INET_TYPE, 'class': 'INetType'},
            {'types': INET6_TYPE, 'class': 'INetSixType'},
            {'types': MAC_TYPE, 'class': 'MACType'},
            {'types': URL_TYPE, 'class': 'URLType'},
            {'types': UUID_TYPE, 'class': 'UUIDType'},
            {'types': REGEX_TYPE, 'class': 'RegexType'},
            {'types': ENUM_TYPE, 'class': 'EnumType'}]


def validate_schema_and_create_stream(schema, created=False):
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
                    new_column = reflection_class(**column)  # pass the json entry as kwargs
                    if 'default' in column:
                        new_column.set_default_value(column['default'])
                    if 'nullable' in column:
                        new_column.set_nullable(column['nullable'])
                    validated_columns[next_name] = new_column
                except Exception as ex:
                    errors[next_name] = ex
                break

    if errors:
        raise ColumnsValidationException(error_messages=errors)

    properties = collections.OrderedDict()
    required_fields = []

    for key, value in validated_columns.iteritems():
        value.add_json_schema_entry(properties)  # append new properties entry
        if not value.is_nullable() and value.get_default_value() is None:  # check if it's required or not
            required_fields.append(key)

    json_schema = Draft4Validator({
        "title": "JSON schema to validate inserts in stream " + schema['schema'] + '.' + schema['stream'],
        "description": "Validate the inserted properties",
        "$schema": "http://json-schema.org/draft-04/schema#",
        "id": "http://monetdb.com/schemas/iot_create.json",
        "type": "array",
        "minItems": 1,
        "items": {"type": "object", "properties": properties, "required": required_fields,
                  "additionalProperties": False}
    }, format_checker=FormatChecker())

    flushing_object = schema['flushing']  # check the flush method
    if flushing_object['base'] == TIMED_FLUSH_IDENTIFIER:
        return TimeBasedStream(schema_name=schema['schema'], stream_name=schema['stream'], columns=validated_columns,
                               validation_schema=json_schema, created=created,
                               interval=int(flushing_object['interval']), time_unit=flushing_object['unit'])
    elif flushing_object['base'] == TUPLE_FLUSH_IDENTIFIER:
        return TupleBasedStream(schema_name=schema['schema'], stream_name=schema['stream'], columns=validated_columns,
                            validation_schema=json_schema, created=created, limit=int(flushing_object['number']))
    return AutoFlushedStream(schema_name=schema['schema'], stream_name=schema['stream'], columns=validated_columns,
                            validation_schema=json_schema, created=created)
