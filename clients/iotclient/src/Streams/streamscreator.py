import collections

from jsonschema import Draft4Validator, FormatChecker
from flushing import TupleBasedFlushing, TimeBasedFlushing
from streams import DataCellStream
from datatypes import *  # the datatypes.py import must be here in order to the reflection work


class ColumnsValidationException(Exception):
    def __init__(self, error_messages):
        super(ColumnsValidationException, self).__init__()
        self.message = error_messages  # dictionary of name ->  error message

SWITCHER = [{'types': ['text', 'string', 'clob', 'character large object'], 'class': 'TextType'},
    {'types': ['uuid'], 'class': 'UUIDType'},
    {'types': ['mac'], 'class': 'MACType'},
    {'types': ['url'], 'class': 'URLType'},
    {'types': ['inet'], 'class': 'INet'},
    {'types': ['inet6'], 'class': 'INetSix'},
    {'types': ['regex'], 'class': 'RegexType'},
    {'types': ['char', 'character', 'varchar', 'character varying'], 'class': 'LimitedTextType'},
    {'types': ['enum'], 'class': 'EnumType'},
    {'types': ['boolean'], 'class': 'BooleanType'},
    {'types': ['tinyint', 'smallint', 'int', 'integer', 'bigint'], 'class': 'SmallIntegerType'},
    {'types': ['hugeint'], 'class': 'HugeIntegerType'},
    {'types': ['real', 'float', 'double'], 'class': 'FloatType'},
    {'types': ['decimal', 'numeric'], 'class': 'DecimalType'},
    {'types': ['date'], 'class': 'DateType'},
    {'types': ['time'], 'class': 'TimeType'},
    {'types': ['timestamp'], 'class': 'TimestampType'}]


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

    flushing_object = schema['flushing']  # check the flush method
    flush_string = flushing_object['base']
    if flush_string == 'time':
        flushing_method = TimeBasedFlushing(interval=int(flushing_object['interval']),
                                            time_unit=flushing_object['unit'])
    else:
        flushing_method = TupleBasedFlushing(limit=int(flushing_object['number']))

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
        "items": {"type": "object", "properties": properties,
                  "required": required_fields, "additionalProperties": False}
    }, format_checker=FormatChecker())

    return DataCellStream(schema_name=schema['schema'], stream_name=schema['stream'], flush_method=flushing_method,
                          columns=validated_columns, validation_schema=json_schema, created=created)
