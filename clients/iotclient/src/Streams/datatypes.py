import datetime
import dateutil
import itertools
import struct
import copy

from abc import ABCMeta, abstractmethod
from dateutil import parser
from src.Streams.jsonschemas import UUID_REG

# TODO later check the byte order https://docs.python.org/2/library/struct.html#byte-order-size-and-alignment
# TODO Also check the consequences of aligment on packing HUGEINTs!
# TODO The null constants might change from system to system due to different CPU's
ALIGNMENT = '<'  # for now is little-endian for Intel CPU's

NIL_STRING = "\200"

INT8_MIN = 0x80
INT16_MIN = 0x8000
INT32_MIN = 0x80000000
INT64_MIN = 0x8000000000000000
INT128_MIN = 0x80000000000000000000000000000000

FLOAT_NAN = struct.unpack('f', '\xff\xff\x7f\xff')[0]
DOUBLE_NAN = struct.unpack('d', '\xff\xff\xff\xff\xff\xff\xef\xff')[0]


class DataValidationException(Exception):
    def __init__(self, errors):
        super(DataValidationException, self).__init__()
        self.message = errors  # dictionary of row_number -> error


class StreamDataType(object):
    """MonetDB's data types for validation base class"""
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self._column_name = kwargs['name']  # name of the column
        self._data_type = kwargs['type']  # SQL name of the type
        self._is_nullable = False  # boolean
        self._default_value = None

    def set_nullable(self, value):  # set if the column is nullable or not
        self._is_nullable = value

    def is_nullable(self):  # check if the column is nullable or not
        return self._is_nullable

    def get_nullable_constant(self):  # get the nullable constant if the column is nullable
        return None

    @abstractmethod
    def set_default_value(self, default_value):  # set the default value representation in the data type
        self._default_value = None

    def get_default_value(self):  # get the default value representation in the data type
        return self._default_value

    @abstractmethod
    def add_json_schema_entry(self, schema):  # add the entry for the stream's corresponding json schema
        pass  # must be done after setting the default value!!!

    def prepare_parameters(self):  # prepare arguments for the binary conversion
        return {}

    @abstractmethod
    def process_next_value(self, entry, counter, parameters, errors):  # process the next entry in binary conversion
        pass

    @abstractmethod
    def pack_parsed_values(self, extracted_values, counter, parameters):
        pass  # pack everything at the end of conversion for 1 write per column

    def process_values(self, json_values):  # json_values is an array of the type
        counter = 0
        parameters = self.prepare_parameters()
        extracted_values = []
        errors = {}

        for entry in json_values:
            counter += 1
            extracted_values.append(self.process_next_value(entry, counter, parameters, errors))

        if errors:
            raise DataValidationException(errors=errors)

        return self.pack_parsed_values(extracted_values, counter, parameters)

    def to_json_representation(self):  # get a json representation of the data type while checking the stream's info
        json_data = {'type': self._data_type, 'nullable': self._is_nullable}
        if self._default_value is not None:
            json_data['default'] = self._default_value
        return json_data

    @abstractmethod
    def get_sql_params(self):
        return []

    def create_stream_sql(self):
        array = [self._column_name, " "]
        array.extend(self.get_sql_params())
        return ''.join(array)


class BaseTextType(StreamDataType):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        super(BaseTextType, self).__init__(**kwargs)

    def get_nullable_constant(self):
        return NIL_STRING

    def prepare_parameters(self):
        return {'lengths_sum': 0}

    def process_next_value(self, entry, counter, parameters, errors):
        parsed_str = str(entry) + '\n'  # for MonetDB's storage
        parameters['lengths_sum'] += len(parsed_str)
        return parsed_str

    def pack_parsed_values(self, extracted_values, counter, parameters):
        string_pack = "".join(extracted_values)
        return struct.pack(ALIGNMENT + str(parameters['lengths_sum']) + 's', string_pack)


class TextType(BaseTextType):
    """Covers: CHAR, VARCHAR, CHARACTER VARYING, TEXT, STRING, CLOB and CHARACTER LARGE OBJECT
        Also Inet, URL and UUID"""

    def __init__(self, **kwargs):
        super(TextType, self).__init__(**kwargs)
        if 'limit' in kwargs:
            self._limit = int(kwargs['limit'])

    def add_json_schema_entry(self, schema):
        dic = {"type": "string"}

        if hasattr(self, '_limit'):  # limit is not used in uri, inet or uuid
            dic['maxLength'] = self._limit
        elif self._data_type == 'url':
            dic['format'] = 'uri'
        elif self._data_type == 'inet':
            dic['format'] = 'ipv4'
        elif self._data_type == 'uuid':
            dic['pattern'] = UUID_REG

        if hasattr(self, '_default_value'):
            dic['default'] = self._default_value

        schema[self._column_name] = dic

    def set_default_value(self, default_value):
        str_value = str(default_value)
        parsed_len = len(str_value)
        if hasattr(self, '_limit') and parsed_len > self._limit:
            raise Exception('The default string is higher than the limit: %d > %d' % (parsed_len, self._limit))
        self._default_value = str_value

    def to_json_representation(self):
        json_value = super(TextType, self).to_json_representation()
        if hasattr(self, '_limit'):
            json_value['limit'] = self._limit
        return json_value

    def get_sql_params(self):
        array = [self._data_type]
        if hasattr(self, '_limit'):
            array.extend(["(", str(self._limit), ")"])
        if self._default_value is not None:
            array.extend([" DEFAULT '", str(self._default_value), "'"])
        if not self._is_nullable:
            array.extend([" NOT NULL"])
        return array


class EnumType(BaseTextType):
    """Covers: enums of strings"""

    def __init__(self, **kwargs):
        super(EnumType, self).__init__(**kwargs)
        self._values = kwargs['values']
        self._max_length = max([len(x) for x in self._values])  # for sql create query

    def add_json_schema_entry(self, schema):
        dic = {"type": "string", "enum": self._values}
        if hasattr(self, '_default_value'):
            dic['default'] = self._default_value
        schema[self._column_name] = dic

    def set_default_value(self, default_value):
        str_value = str(default_value)
        if str_value not in self._values:
            raise Exception('The default value is not in the enumeration!')
        self._default_value = str_value

    def to_json_representation(self):
        json_value = super(EnumType, self).to_json_representation()
        json_value['values'] = self._values
        return json_value

    def get_sql_params(self):
        array = ["char(", str(self._max_length), ")"]
        if self._default_value is not None:
            array.extend([" DEFAULT '", str(self._default_value), "'"])
        if not self._is_nullable:
            array.extend([" NOT NULL"])
        array.extend([" CHECK (", self._column_name, " IN ("])
        array.extend(','.join(map(lambda x: "\'" + x + "\'", self._values)))
        array.extend(["))"])
        return array


class BooleanType(StreamDataType):
    """Covers: BOOLEAN"""

    def __init__(self, **kwargs):
        super(BooleanType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        dic = {"type": "boolean"}
        if hasattr(self, '_default_value'):
            dic['default'] = self._default_value
        schema[self._column_name] = dic

    def set_default_value(self, default_value):
        self._default_value = bool(default_value)

    def get_nullable_constant(self):
        return INT8_MIN

    def process_next_value(self, entry, counter, parameters, errors):
        if bool(entry):
            return 1
        return 0

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(ALIGNMENT + str(counter) + 'b', *extracted_values)

    def get_sql_params(self):
        array = [self._data_type]
        if self._default_value is not None:
            array.extend([" DEFAULT ", str(self._default_value)])
        if not self._is_nullable:
            array.extend([" NOT NULL"])
        return array


class NumberBaseType(StreamDataType):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        super(NumberBaseType, self).__init__(**kwargs)
        if 'minimum' in kwargs:
            self._minimum = self.process_next_value(kwargs['minimum'], 0, {}, {})
        if 'maximum' in kwargs:
            self._maximum = self.process_next_value(kwargs['maximum'], 0, {}, {})
        if hasattr(self, '_minimum') and hasattr(self, '_maximum') and self._minimum > self._maximum:
            raise Exception('The minimum value is higher than the maximum!')

    def add_json_schema_entry(self, schema):
        dic = {}
        if hasattr(self, '_default_value'):
            dic['default'] = self._default_value
        if hasattr(self, '_minimum'):  # we can add numbers to schema
            dic['minimum'] = self._minimum
        if hasattr(self, '_maximum'):
            dic['maximum'] = self._maximum
        schema[self._column_name] = dic

    @abstractmethod
    def process_default_value(self, value):
        return value

    def set_default_value(self, default_value):
        parsed_val = self.process_default_value(default_value)
        if hasattr(self, '_minimum') and not hasattr(self, '_maximum') and parsed_val < self._minimum:
            raise Exception('The default value is less than the minimum: %d < %d' % (parsed_val, self._minimum))
        elif hasattr(self, '_maximum') and not hasattr(self, '_minimum') and parsed_val > self._maximum:
            raise Exception('The default value is higher than the maximum: %d > %d' % (parsed_val, self._maximum))
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and parsed_val < self._minimum:
            raise Exception('The default value is out of range: %d < %d' % (parsed_val, self._minimum))
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and parsed_val > self._maximum:
            raise Exception('The default value is out of range: %d > %d' % (parsed_val, self._maximum))
        self._default_value = parsed_val

    def to_json_representation(self):
        json_value = super(NumberBaseType, self).to_json_representation()
        if hasattr(self, '_minimum'):
            json_value['minimum'] = self._minimum
        if hasattr(self, '_maximum'):
            json_value['maximum'] = self._maximum
        return json_value

    def get_sql_params(self):
        array = [self._data_type]
        if self._default_value is not None:
            array.extend([" DEFAULT ", str(self._default_value)])
        if not self._is_nullable:
            array.extend([" NOT NULL"])
        if hasattr(self, '_minimum') and not hasattr(self, '_maximum'):
            array.extend([" CHECK (", self._column_name, " > ", str(self._minimum), ")"])
        elif hasattr(self, '_maximum') and not hasattr(self, '_minimum'):
            array.extend([" CHECK (", self._column_name, " < ", str(self._maximum), ")"])
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum'):
            array.extend([" CHECK (", self._column_name, " BETWEEN ", str(self._minimum),
                          " AND ", str(self._maximum), ")"])
        return array


class SmallIntegerType(NumberBaseType):
    """Covers: TINYINT, SMALLINT, INT[EGER], BIGINT"""

    def __init__(self, **kwargs):
        super(SmallIntegerType, self).__init__(**kwargs)
        this_type = kwargs['type']
        self._pack_sym = {'tinyint': 'b', 'smallint': 'h', 'int': 'i', 'integer': 'i', 'bigint': 'q'}.get(this_type)
        self._nullable_constant = {'tinyint': INT8_MIN, 'smallint': INT16_MIN, 'int': INT32_MIN, 'integer': INT32_MIN,
                                   'bigint': INT64_MIN}.get(this_type)

    def add_json_schema_entry(self, schema):
        super(SmallIntegerType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'integer'

    def get_nullable_constant(self):
        return self._nullable_constant

    def process_default_value(self, value):
        return int(value)

    def process_next_value(self, entry, counter, parameters, errors):
        return int(entry)

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(ALIGNMENT + str(counter) + self._pack_sym, extracted_values)

max_int64 = 0xFFFFFFFFFFFFFFFF


class HugeIntegerType(NumberBaseType):
    """Covers: HUGEINT"""

    def __init__(self, **kwargs):
        super(HugeIntegerType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(HugeIntegerType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'integer'

    def get_nullable_constant(self):
        return INT128_MIN

    def process_default_value(self, value):
        return int(value)

    def process_next_value(self, entry, counter, parameters, errors):
        return [entry & max_int64, (entry >> 64) & max_int64]

    def pack_parsed_values(self, extracted_values, counter, parameters):
        concat_array = list(itertools.chain(*extracted_values))
        return struct.pack(ALIGNMENT + str(counter << 1) + 'Q', *concat_array)


class FloatType(NumberBaseType):
    """Covers: REAL, FLOAT and DOUBLE"""

    def __init__(self, **kwargs):
        super(FloatType, self).__init__(**kwargs)
        this_type = kwargs['type']
        self._pack_sym = {'real': 'f', 'float': 'd', 'double': 'd'}.get(this_type)
        self._nullable_constant = {'real': FLOAT_NAN, 'float': DOUBLE_NAN, 'double': DOUBLE_NAN}.get(this_type)

    def add_json_schema_entry(self, schema):
        super(FloatType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'number'

    def get_nullable_constant(self):
        return self._nullable_constant

    def process_default_value(self, value):
        return float(value)

    def process_next_value(self, entry, counter, parameters, errors):
        return float(entry)

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(ALIGNMENT + str(counter) + self._pack_sym, *extracted_values)


class DecimalType(NumberBaseType):  # TODO finish this class, how to serialize these values
    """Covers: DECIMAL and NUMERIC"""

    def __init__(self, **kwargs):
        super(DecimalType, self).__init__(**kwargs)
        if 'precision' in kwargs:
            self._precision = int(kwargs['precision'])
        else:
            self._precision = 18
        if 'scale' in kwargs:
            self._scale = int(kwargs['scale'])
        else:
            self._scale = 0

    def add_json_schema_entry(self, schema):
        super(DecimalType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'number'

    def get_nullable_constant(self):
        return 0

    def process_default_value(self, value):
        return float(value)

    def process_next_value(self, entry, counter, parameters, errors):
        # precision, scale = precision_and_scale(entry)
        return float(entry)

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(ALIGNMENT + str(counter) + 'd', *extracted_values)

    def to_json_representation(self):
        json_value = super(DecimalType, self).to_json_representation()
        json_value['precision'] = self._precision
        json_value['scale'] = self._scale
        return json_value


class BaseDateTimeType(StreamDataType):  # The validation of time variables can't be done on the schema
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        super(BaseDateTimeType, self).__init__(**kwargs)
        if 'minimum' in kwargs:
            self._minimum_text = kwargs['minimum']  # to show on json representation
            self._minimum = self.parse_entry(kwargs['minimum'])
        if 'maximum' in kwargs:
            self._maximum_text = kwargs['maximum']  # to show on json representation
            self._maximum = self.parse_entry(kwargs['maximum'])
        if hasattr(self, '_minimum') and hasattr(self, '_maximum') and self._minimum > self._maximum:
            raise Exception('The minimum value is higher than the maximum!')

    def add_json_schema_entry(self, schema):
        dic = {}
        if hasattr(self, '_default_value'):
            dic['default'] = str(self._default_value)
        schema[self._column_name] = dic

    def set_default_value(self, default_value):
        parsed_val = self.parse_entry(default_value)  # Process the default value as others
        if hasattr(self, '_minimum') and not hasattr(self, '_maximum') and parsed_val < self._minimum:
            raise Exception('The default value is less than the minimum: %s < %s' % (default_value, self._minimum_text))
        elif hasattr(self, '_maximum') and not hasattr(self, '_minimum') and parsed_val > self._maximum:
            raise Exception('The default value is higher than the maximum: %s > %s'
                            % (default_value, self._maximum_text))
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and parsed_val < self._minimum:
            raise Exception('The default value is out of range: %s < %s' % (default_value, self._minimum_text))
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and parsed_val > self._maximum:
            raise Exception('The default value is out of range: %s > %s' % (default_value, self._maximum_text))
        self._default_value = parsed_val

    def to_json_representation(self):
        json_value = super(BaseDateTimeType, self).to_json_representation()
        if hasattr(self, '_minimum'):
            json_value['minimum'] = self._minimum_text
        if hasattr(self, '_maximum'):
            json_value['maximum'] = self._maximum_text
        return json_value

    def get_sql_params(self):
        array = [self._data_type]
        if not self._is_nullable:
            array.extend([" NOT NULL"])
        return array

    @abstractmethod
    def parse_entry(self, entry):
        return 0

    @abstractmethod
    def pack_next_value(self, parsed, counter, parameters, errors):
        return

    def process_next_value(self, entry, counter, parameters, errors):
        parsed = self.parse_entry(entry)

        if hasattr(self, '_minimum') and not hasattr(self, '_maximum') and parsed < self._minimum:
            errors[counter] = 'The value is higher than the minimum!'
        elif hasattr(self, '_maximum') and not hasattr(self, '_minimum') and parsed > self._maximum:
            errors[counter] = 'The value is higher than the maximum!'
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and \
                (parsed < self._minimum or parsed > self._maximum):
            errors[counter] = 'The value is out of range!'
        return self.pack_next_value(parsed, counter, parameters, errors)


class DateType(BaseDateTimeType):  # Stored as an uint with the number of days since day 0 of month 1 (Jan) from year 0
    """Covers: DATE"""

    def __init__(self, **kwargs):
        super(DateType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(DateType, self).add_json_schema_entry(schema)
        entry = schema[self._column_name]
        entry['type'] = 'string'
        entry['format'] = 'date'

    def get_nullable_constant(self):
        return INT32_MIN  # Checked from MonetDB's source code

    def parse_entry(self, entry):
        return datetime.datetime.strptime(str(entry), "%Y-%m-%d")

    def pack_next_value(self, parsed, counter, parameters, errors):
        day0 = copy.deepcopy(parsed).replace(year=1, month=1, day=1)
        return (parsed - day0).days + 366  # the mindate in python is 1, but for the representation is 0, so why the add

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(ALIGNMENT + str(counter) + 'I', *extracted_values)


class TimeType(BaseDateTimeType):  # Stored as an uint with the number of milliseconds since hour 00:00:00
    """Covers: TIME"""

    def __init__(self, **kwargs):
        super(TimeType, self).__init__(**kwargs)

    def get_nullable_constant(self):
        return INT32_MIN  # Checked from MonetDB's source code

    def add_json_schema_entry(self, schema):
        super(TimeType, self).add_json_schema_entry(schema)
        entry = schema[self._column_name]
        entry['type'] = 'string'
        entry['format'] = 'time'

    def parse_entry(self, entry):
        return datetime.datetime.strptime(str(entry), "%H:%M:%S")

    def pack_next_value(self, parsed, counter, parameters, errors):
        hour0 = copy.deepcopy(parsed).replace(hour=0, minute=0, second=0, microsecond=0)
        return (parsed - hour0).seconds * 1000

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(ALIGNMENT + str(counter) + 'I', *extracted_values)


class TimestampType(BaseDateTimeType):  # it's represented with the two integers from time and date
    """Covers: TIMESTAMP"""

    def __init__(self, **kwargs):
        super(TimestampType, self).__init__(**kwargs)

    def get_nullable_constant(self):
        return [0, INT32_MIN]  # Checked from MonetDB's source code

    def add_json_schema_entry(self, schema):
        super(TimestampType, self).add_json_schema_entry(schema)
        entry = schema[self._column_name]
        entry['type'] = 'string'
        entry['format'] = 'date-time'

    def parse_entry(self, entry):
        return dateutil.parser.parse(entry)

    def pack_next_value(self, parsed, counter, parameters, errors):
        hour0 = copy.deepcopy(parsed).replace(hour=0, minute=0, second=0, microsecond=0)
        day0 = copy.deepcopy(parsed).replace(year=1, month=1, day=1)

        days = (parsed - day0).days + 366  # the mindate in python is 1, but for the representation is 0, so why the add
        milliseconds = (parsed - hour0).seconds * 1000
        return [milliseconds, days]

    def pack_parsed_values(self, extracted_values, counter, parameters):
        concat_array = list(itertools.chain(*extracted_values))
        return struct.pack(ALIGNMENT + str(counter << 1) + 'I', *concat_array)
