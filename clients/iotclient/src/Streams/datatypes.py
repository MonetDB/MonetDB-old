import copy
import datetime
import dateutil
import itertools
import math
import re
import struct


from abc import ABCMeta, abstractmethod
from dateutil import parser
from jsonschemas import UUID_REGEX, MAC_ADDRESS_REGEX, TIME_REGEX

# Later check the byte order https://docs.python.org/2/library/struct.html#byte-order-size-and-alignment
# Also check the consequences of aligment on packing HUGEINTs!
# The null constants might change from system to system due to different CPU's
ALIGNMENT = '<'  # for now is little-endian for Intel CPU's

NIL_STRING = "\200"

INT8_MIN = 0x80
INT16_MIN = 0x8000
INT32_MIN = 0x80000000
INT64_MIN = 0x8000000000000000
INT64_MAX = 0xFFFFFFFFFFFFFFFF
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

    def add_json_schema_entry(self, schema):  # add the entry for the stream's corresponding json schema
        dic = {}  # must be done after setting the default value!!!
        if hasattr(self, '_default_value'):
            dic['default'] = self._default_value
        schema[self._column_name] = dic

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
        json_data = {'name': self._column_name, 'type': self._data_type, 'nullable': self._is_nullable}
        if self._default_value is not None:
            json_data['default'] = self._default_value
        return json_data

    def process_sql_parameters(self, array):  # get other possible parameters such as a limit, minimum and maximum
        pass

    def create_stream_sql(self):  # get column creation statement on SQL
        array = [self._column_name, " ", self._data_type]
        self.process_sql_parameters(array)  # add extra parameters to the SQL statement
        if self._default_value is not None:
            array.extend([" DEFAULT '", str(self._default_value), "'"])
        if not self._is_nullable:
            array.append(" NOT NULL")
        return ''.join(array)


class TextType(StreamDataType):
    """Covers: TEXT, STRING, CLOB and CHARACTER LARGE OBJECT"""

    def __init__(self, **kwargs):
        super(TextType, self).__init__(**kwargs)

    def get_nullable_constant(self):
        return NIL_STRING

    def set_default_value(self, default_value):
        self._default_value = default_value

    def add_json_schema_entry(self, schema):
        super(TextType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'string'

    def prepare_parameters(self):
        return {'lengths_sum': 0}

    def process_next_value(self, entry, counter, parameters, errors):
        parsed_str = str(entry) + '\n'  # for MonetDB's storage
        parameters['lengths_sum'] += len(parsed_str)
        return parsed_str

    def pack_parsed_values(self, extracted_values, counter, parameters):
        string_pack = "".join(extracted_values)
        return struct.pack(ALIGNMENT + str(parameters['lengths_sum']) + 's', string_pack)


class UUIDType(TextType):
    """Covers: UUID"""

    def __init__(self, **kwargs):
        super(UUIDType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(UUIDType, self).add_json_schema_entry(schema)
        schema[self._column_name]['pattern'] = UUID_REGEX

    def prepare_parameters(self):
        return {}

    def process_next_value(self, entry, counter, parameters, errors):
        return str(entry) + '\n'

    def pack_parsed_values(self, extracted_values, counter, parameters):
        string_pack = "".join(extracted_values)
        return struct.pack(ALIGNMENT + str(37 * counter) + 's', string_pack)


class MACType(TextType):
    """Covers: MAC addresses"""

    def __init__(self, **kwargs):
        super(MACType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(MACType, self).add_json_schema_entry(schema)
        schema[self._column_name]['pattern'] = MAC_ADDRESS_REGEX

    def prepare_parameters(self):
        return {}

    def process_next_value(self, entry, counter, parameters, errors):
        return str(entry) + '\n'

    def pack_parsed_values(self, extracted_values, counter, parameters):
        string_pack = "".join(extracted_values)
        return struct.pack(ALIGNMENT + str(18 * counter) + 's', string_pack)

    def process_sql_parameters(self, array):
        array[2] = 'char(17)'  # A MAC Address has 17 characters


class URLType(TextType):
    """Covers: URL"""

    def __init__(self, **kwargs):
        super(URLType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(URLType, self).add_json_schema_entry(schema)
        schema[self._column_name]['format'] = 'uri'


class INet(TextType):
    """Covers: Inet"""

    def __init__(self, **kwargs):
        super(INet, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(INet, self).add_json_schema_entry(schema)
        schema[self._column_name]['format'] = 'ipv4'


class INetSix(TextType):
    """Covers: Inet6"""

    def __init__(self, **kwargs):
        super(INetSix, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(INetSix, self).add_json_schema_entry(schema)
        schema[self._column_name]['format'] = 'ipv6'

    # http://stackoverflow.com/questions/166132/maximum-length-of-the-textual-representation-of-an-ipv6-address
    def process_sql_parameters(self, array):
        array[2] = 'char(45)'


class RegexType(TextType):
    """Covers: Regex"""

    def __init__(self, **kwargs):
        super(TextType, self).__init__(**kwargs)
        self._regex = re.compile(kwargs['regex'])

    def set_default_value(self, default_value):
        if self._regex.match(default_value) is None:
            raise Exception('The default value does not match with the regular expression!')
        self._default_value = default_value

    def add_json_schema_entry(self, schema):
        super(RegexType, self).add_json_schema_entry(schema)
        schema[self._column_name]['pattern'] = self._regex.pattern

    def to_json_representation(self):
        json_value = super(RegexType, self).to_json_representation()
        json_value['regex'] = self._regex.pattern
        return json_value

    def process_sql_parameters(self, array):
        array[2] = 'string'  # Store as string


class LimitedTextType(TextType):
    """Covers: CHAR, CHARACTER, VARCHAR, CHARACTER VARYING"""

    def __init__(self, **kwargs):
        super(LimitedTextType, self).__init__(**kwargs)
        self._limit = kwargs['limit']

    def add_json_schema_entry(self, schema):
        super(LimitedTextType, self).add_json_schema_entry(schema)
        schema[self._column_name]['maxLength'] = self._limit

    def set_default_value(self, default_value):
        str_value = str(default_value)
        parsed_len = len(str_value)
        if parsed_len > self._limit:
            raise Exception('The default string\'s length is longer than the limit: %d > %d!'
                            % (parsed_len, self._limit))
        self._default_value = str_value

    def to_json_representation(self):
        json_value = super(LimitedTextType, self).to_json_representation()
        json_value['limit'] = self._limit
        return json_value

    def process_sql_parameters(self, array):
        array[2] += ''.join(["(", str(self._limit), ")"])  # add the limit restriction after the type declaration


class EnumType(TextType):
    """Covers: Enum of strings"""

    def __init__(self, **kwargs):
        super(EnumType, self).__init__(**kwargs)
        self._values = kwargs['values']

    def add_json_schema_entry(self, schema):
        super(EnumType, self).add_json_schema_entry(schema)
        schema[self._column_name]['enum'] = self._values

    def set_default_value(self, default_value):
        str_value = str(default_value)
        if str_value not in self._values:
            raise Exception('The default value is not present in the enumeration!')
        self._default_value = str_value

    def to_json_representation(self):
        json_value = super(EnumType, self).to_json_representation()
        json_value['values'] = self._values
        return json_value

    def create_stream_sql(self):
        max_length = max([len(y) for y in self._values])
        array = [self._column_name, " char(", str(max_length), ")"]
        if self._default_value is not None:
            array.extend([" DEFAULT '", str(self._default_value), "'"])
        if not self._is_nullable:
            array.append(" NOT NULL")
        array.extend([" CHECK (", self._column_name, " IN (", ','.join(map(lambda x: "\'" + x + "\'", self._values)),
                      "))"])
        return ''.join(array)


class BooleanType(StreamDataType):
    """Covers: BOOLEAN"""

    def __init__(self, **kwargs):
        super(BooleanType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(BooleanType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'boolean'

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


class NumberBaseType(StreamDataType):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        super(NumberBaseType, self).__init__(**kwargs)
        if 'minimum' in kwargs:
            self._minimum = kwargs['minimum']
        if 'maximum' in kwargs:
            self._maximum = kwargs['maximum']
        if hasattr(self, '_minimum') and hasattr(self, '_maximum') and self._minimum > self._maximum:
            raise Exception('The minimum value is higher than the maximum!')

    def add_json_schema_entry(self, schema):
        super(NumberBaseType, self).add_json_schema_entry(schema)
        if hasattr(self, '_minimum'):  # we can add numbers to schema
            schema[self._column_name]['minimum'] = self._minimum
        if hasattr(self, '_maximum'):
            schema[self._column_name]['maximum'] = self._maximum

    def set_default_value(self, default_value):
        if hasattr(self, '_minimum') and not hasattr(self, '_maximum') and default_value < self._minimum:
            raise Exception('The default value is less than the minimum: %s < %s!' % (default_value, self._minimum))
        elif hasattr(self, '_maximum') and not hasattr(self, '_minimum') and default_value > self._maximum:
            raise Exception('The default value is higher than the maximum: %s > %s!' % (default_value, self._maximum))
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and default_value < self._minimum:
            raise Exception('The default value is out of range: %s < %s!' % (default_value, self._minimum))
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and default_value > self._maximum:
            raise Exception('The default value is out of range: %s > %s!' % (default_value, self._maximum))
        self._default_value = default_value

    def to_json_representation(self):
        json_value = super(NumberBaseType, self).to_json_representation()
        if hasattr(self, '_minimum'):
            json_value['minimum'] = self._minimum
        if hasattr(self, '_maximum'):
            json_value['maximum'] = self._maximum
        return json_value

    def create_stream_sql(self):
        string = super(NumberBaseType, self).create_stream_sql()

        if hasattr(self, '_minimum') and not hasattr(self, '_maximum'):
            return string + ''.join([" CHECK (", self._column_name, " > ", str(self._minimum), ")"])
        elif hasattr(self, '_maximum') and not hasattr(self, '_minimum'):
            return string + ''.join([" CHECK (", self._column_name, " < ", str(self._maximum), ")"])
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum'):
            return string + ''.join([" CHECK (", self._column_name, " BETWEEN ", str(self._minimum),
                                     " AND ", str(self._maximum), ")"])
        return string


class SmallIntegerType(NumberBaseType):
    """Covers: TINYINT, SMALLINT, INT[EGER], BIGINT"""

    def __init__(self, **kwargs):
        super(SmallIntegerType, self).__init__(**kwargs)
        self._pack_sym = {'tinyint': 'b', 'smallint': 'h', 'int': 'i', 'integer': 'i', 'bigint': 'q'} \
            .get(kwargs['type'])
        self._nullable_constant = {'tinyint': INT8_MIN, 'smallint': INT16_MIN, 'int': INT32_MIN, 'integer': INT32_MIN,
                                   'bigint': INT64_MIN}.get(kwargs['type'])

    def add_json_schema_entry(self, schema):
        super(SmallIntegerType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'integer'

    def get_nullable_constant(self):
        return self._nullable_constant

    def process_next_value(self, entry, counter, parameters, errors):
        return int(entry)

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(ALIGNMENT + str(counter) + self._pack_sym, *extracted_values)


class HugeIntegerType(NumberBaseType):
    """Covers: HUGEINT"""

    def __init__(self, **kwargs):
        super(HugeIntegerType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(HugeIntegerType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'integer'

    def get_nullable_constant(self):
        return INT128_MIN

    def process_next_value(self, entry, counter, parameters, errors):
        return [entry & INT64_MAX, (entry >> 64) & INT64_MAX]

    def pack_parsed_values(self, extracted_values, counter, parameters):
        extracted_values = list(itertools.chain(*extracted_values))
        return struct.pack(ALIGNMENT + str(counter << 1) + 'Q', *extracted_values)


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

    def process_next_value(self, entry, counter, parameters, errors):
        return float(entry)

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(ALIGNMENT + str(counter) + self._pack_sym, *extracted_values)


class DecimalType(NumberBaseType):
    """Covers: DECIMAL and NUMERIC"""

    def __init__(self, **kwargs):
        super(DecimalType, self).__init__(**kwargs)
        if 'precision' in kwargs:
            self._precision = kwargs['precision']
        else:
            self._precision = 18
        if 'scale' in kwargs:
            self._scale = kwargs['scale']
        else:
            self._scale = 0

        if self._scale > self._precision:
            raise Exception('The scale must be between 0 and the precision!')

        if self._default_value is not None:
            self.check_value_precision(self._default_value, 'default')
        if hasattr(self, '_minimum'):
            self.check_value_precision(self._minimum, 'minimum')
        if hasattr(self, '_maximum'):
            self.check_value_precision(self._maximum, 'maximum')

        if self._precision <= 2:  # calculate the number of bytes to use according to the precision
            self._pack_sym = 'b'
        elif 2 < self._precision <= 4:
            self._pack_sym = 'h'
        elif 4 < self._precision <= 8:
            self._pack_sym = 'i'
        elif 8 < self._precision <= 18:
            self._pack_sym = 'q'
        elif 18 < self._precision <= 38:
            self._pack_sym = 'Q'

        self._nullable_constant = {'b': INT8_MIN, 'h': INT16_MIN, 'i': INT32_MIN, 'q': INT64_MIN, 'Q': INT128_MIN} \
            .get(self._pack_sym)

    def check_value_precision(self, value, text):
        number_digits = int(math.ceil(math.log10(abs(value))))
        if number_digits > self._precision:
            raise Exception('Too many digits on %s value: %s > %s!' % (text, number_digits, self._precision))

    def add_json_schema_entry(self, schema):
        super(DecimalType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'number'

    def get_nullable_constant(self):
        return self._nullable_constant

    def process_next_value(self, entry, counter, parameters, errors):
        self.check_value_precision(entry, 'entry')
        parsed_value = int(entry)
        if self._pack_sym != 'Q':
            return parsed_value
        else:
            return [parsed_value & INT64_MAX, (parsed_value >> 64) & INT64_MAX]

    def pack_parsed_values(self, extracted_values, counter, parameters):
        if self._pack_sym == 'Q':
            extracted_values = list(itertools.chain(*extracted_values))
            counter <<= 1  # duplicate the counter for packing
        return struct.pack(ALIGNMENT + str(counter) + self._pack_sym, *extracted_values)

    def to_json_representation(self):
        json_value = super(DecimalType, self).to_json_representation()
        json_value['precision'] = self._precision
        json_value['scale'] = self._scale
        return json_value

    def process_sql_parameters(self, array):  # add the precision and scale
        array.append(''.join(["(", str(self._precision), ",", str(self._scale), ")"]))


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

    def get_nullable_constant(self):
        return "0"

    @abstractmethod
    def parse_entry(self, entry):
        pass

    def set_default_value(self, default_value):
        parsed_val = self.parse_entry(default_value)  # Process the default value as the others
        if hasattr(self, '_minimum') and not hasattr(self, '_maximum') and parsed_val < self._minimum:
            raise Exception('The default value is less than the minimum: %s < %s!'
                            % (default_value, self._minimum_text))
        elif hasattr(self, '_maximum') and not hasattr(self, '_minimum') and parsed_val > self._maximum:
            raise Exception('The default value is higher than the maximum: %s > %s!'
                            % (default_value, self._maximum_text))
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and parsed_val < self._minimum:
            raise Exception('The default value is out of range: %s < %s!' % (default_value, self._minimum_text))
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and parsed_val > self._maximum:
            raise Exception('The default value is out of range: %s > %s!' % (default_value, self._maximum_text))
        self._default_value = default_value

    @abstractmethod
    def pack_next_value(self, parsed, counter, parameters, errors):
        pass

    def process_next_value(self, entry, counter, parameters, errors):
        if entry == self.get_nullable_constant():  # have to do this trick due to Python's datetime limitations
            return self.pack_next_value(None, counter, parameters, errors)
        parsed = self.parse_entry(entry)
        if hasattr(self, '_minimum') and not hasattr(self, '_maximum') and parsed < self._minimum:
            errors[counter] = 'The value is higher than the minimum: %s < %s!' % (parsed, self._minimum_text)
        elif hasattr(self, '_maximum') and not hasattr(self, '_minimum') and parsed > self._maximum:
            errors[counter] = 'The value is higher than the maximum: %s > %s!' % (parsed, self._maximum_text)
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and parsed < self._minimum:
            errors[counter] = 'The value is out of range: %s < %s!' % (parsed, self._minimum_text)
        elif hasattr(self, '_maximum') and hasattr(self, '_minimum') and parsed > self._maximum:
            errors[counter] = 'The value is out of range: %s > %s!' % (parsed, self._maximum_text)
        return self.pack_next_value(parsed, counter, parameters, errors)

    def to_json_representation(self):
        json_value = super(BaseDateTimeType, self).to_json_representation()
        if hasattr(self, '_minimum'):
            json_value['minimum'] = self._minimum_text
        if hasattr(self, '_maximum'):
            json_value['maximum'] = self._maximum_text
        return json_value


class DateType(BaseDateTimeType):  # Stored as an uint with the number of days since day 1 of month 1 (Jan) from year 0
    """Covers: DATE"""

    def __init__(self, **kwargs):
        super(DateType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(DateType, self).add_json_schema_entry(schema)
        schema[self._column_name]['format'] = 'date'

    def parse_entry(self, entry):
        return datetime.datetime.strptime(str(entry), "%Y-%m-%d")

    def pack_next_value(self, parsed, counter, parameters, errors):
        if parsed is None:
            return INT32_MIN
        day0 = copy.deepcopy(parsed).replace(year=1, month=1, day=1)
        # the minyear in python is 1, but for the representation is 0, so why the add
        return int((parsed - day0).days) + 366

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(ALIGNMENT + str(counter) + 'I', *extracted_values)


class TimeType(BaseDateTimeType):  # Stored as an uint with the number of milliseconds since hour 00:00:00
    """Covers: TIME"""

    def __init__(self, **kwargs):
        super(TimeType, self).__init__(**kwargs)
        if 'timezone' in kwargs:
            self._has_timezone = kwargs['timezone']
        else:
            self._has_timezone = True

    def add_json_schema_entry(self, schema):
        super(TimeType, self).add_json_schema_entry(schema)
        schema[self._column_name]['pattern'] = TIME_REGEX

    def parse_entry(self, entry):
        parsed_time = datetime.datetime.strptime(str(entry), "%H:%M:%S.%f")
        if not self._has_timezone:
            parsed_time = parsed_time.replace(tzinfo=None)
        return parsed_time

    def pack_next_value(self, parsed, counter, parameters, errors):
        if parsed is None:
            return INT32_MIN
        hour0 = copy.deepcopy(parsed).replace(hour=0, minute=0, second=0, microsecond=0)
        delta = parsed - hour0
        return int(delta.total_seconds()) * 1000 + int(delta.microseconds) / 1000

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(ALIGNMENT + str(counter) + 'I', *extracted_values)


class TimestampType(BaseDateTimeType):  # it's represented with the two integers from time and date
    """Covers: TIMESTAMP"""

    def __init__(self, **kwargs):
        super(TimestampType, self).__init__(**kwargs)
        if 'timezone' in kwargs:
            self._has_timezone = kwargs['timezone']
        else:
            self._has_timezone = True

    def add_json_schema_entry(self, schema):
        super(TimestampType, self).add_json_schema_entry(schema)
        schema[self._column_name]['format'] = 'date-time'

    def parse_entry(self, entry):
        parsed_timestamp = dateutil.parser.parse(entry)
        if not self._has_timezone:
            parsed_timestamp = parsed_timestamp.replace(tzinfo=None)
        return parsed_timestamp

    def pack_next_value(self, parsed, counter, parameters, errors):
        if parsed is None:
            return [0, INT32_MIN]
        hour0 = copy.deepcopy(parsed).replace(hour=0, minute=0, second=0, microsecond=0)
        day0 = copy.deepcopy(parsed).replace(year=1, month=1, day=1)
        days = int((parsed - day0).days) + 366
        delta = parsed - hour0
        milliseconds = int(delta.total_seconds()) * 1000 + int(delta.microseconds) / 1000
        return [milliseconds, days]

    def pack_parsed_values(self, extracted_values, counter, parameters):
        concat_array = list(itertools.chain(*extracted_values))
        return struct.pack(ALIGNMENT + str(counter << 1) + 'I', *concat_array)
