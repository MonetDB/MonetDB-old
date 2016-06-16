import struct

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta
from dateutil import parser
from itertools import chain
from json import dumps
from math import ceil, log10
from re import compile, split
from jsonschemas import UUID_REGEX, MAC_ADDRESS_REGEX, TIME_REGEX, IPV4_REGEX, TIME_WITH_TIMEZONE_TYPE_INTERNAL,\
    TIME_WITH_TIMEZONE_TYPE_EXTERNAL, TIMESTAMP_WITH_TIMEZONE_TYPE_INTERNAL, TIMESTAMP_WITH_TIMEZONE_TYPE_EXTERNAL

# The null constants might change from system to system due to different CPU's limits
LITTLE_ENDIAN_ALIGNMENT = '<'  # for now it is little-endian

NIL_STRING = "\200"
NIL_UUID = "00000000-0000-0000-0000-000000000000"

INT8_MIN = -128
INT16_MIN = -32768
INT32_MIN = -2147483648
INT64_MIN = -9223372036854775808
FLOAT_NAN = struct.unpack('f', '\xff\xff\x7f\xff')[0]
DOUBLE_NAN = struct.unpack('d', '\xff\xff\xff\xff\xff\xff\xef\xff')[0]

ENUM_TYPE_SEPARATOR = '\r'


class StreamDataType:
    """MonetDB's data types for validation base class"""
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self._column_name = kwargs['name']  # name of the column
        self._data_type = kwargs['type']  # SQL name of the type
        self._is_nullable = kwargs.get('nullable', True)  # boolean
        if 'default' in kwargs and kwargs['default'] is not None:
            self._default_value = self.process_default_value(kwargs['default'])
        else:
            self._default_value = None

    def is_nullable(self):  # check if the column is nullable or not
        return self._is_nullable

    @abstractmethod
    def get_nullable_constant(self):  # get the nullable constant if the column is nullable
        return None

    def process_default_value(self, default_value):  # process the default value representation in the data type
        return default_value

    def get_default_value(self):  # get the default value representation in the data type
        return self._default_value

    def add_json_schema_entry(self, schema):  # add the entry for the stream's corresponding json schema
        dic = {}  # must be done after setting the default value!!!
        if self._default_value is not None:
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
            raise Exception(errors=dumps(errors))  # dictionary of row_number -> error
        return self.pack_parsed_values(extracted_values, counter, parameters)

    def to_json_representation(self):  # get a json representation of the data type while checking the stream's info
        return OrderedDict((('name', self._column_name), ('type', self._data_type),
                            ('default', self._default_value), ('nullable', self._is_nullable)))

    def process_sql_parameters(self, array):  # get other possible parameters such as a limit, minimum and maximum
        pass

    def create_stream_sql(self):  # get column creation statement on SQL
        array = [self._column_name, " ", self._data_type]
        if self._default_value is not None:
            array.extend([" DEFAULT '", str(self._default_value), "'"])
        if not self._is_nullable:
            array.append(" NOT NULL")
        self.process_sql_parameters(array)  # add extra parameters to the SQL statement
        return ''.join(array)

    def get_extra_sql_statement(self):  # data to iot.webservervalidation
        return ",NULL,NULL,NULL"


class TextType(StreamDataType):
    """Covers: TEXT, STRING, CLOB and CHARACTER LARGE OBJECT"""

    def __init__(self, **kwargs):
        super(TextType, self).__init__(**kwargs)

    def get_nullable_constant(self):
        return NIL_STRING

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
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(parameters['lengths_sum']) + 's', string_pack)


class MACType(TextType):
    """Covers: MAC addresses"""

    def __init__(self, **kwargs):
        super(MACType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(MACType, self).add_json_schema_entry(schema)
        schema[self._column_name]['pattern'] = MAC_ADDRESS_REGEX

    def process_next_value(self, entry, counter, parameters, errors):
        return str(entry) + '\n'

    def pack_parsed_values(self, extracted_values, counter, parameters):
        string_pack = "".join(extracted_values)
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(18 * counter) + 's', string_pack)

    def process_sql_parameters(self, array):
        array[2] = 'char(17)'  # A MAC Address has 17 characters

    def get_extra_sql_statement(self):
        return ",1,NULL,NULL"


class URLType(TextType):
    """Covers: URL"""

    def __init__(self, **kwargs):
        super(URLType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(URLType, self).add_json_schema_entry(schema)
        schema[self._column_name]['format'] = 'uri'


class RegexType(TextType):
    """Covers: Regex"""

    def __init__(self, **kwargs):
        self._regex = compile(kwargs['regex'])
        self._regex_text = kwargs['regex']
        super(RegexType, self).__init__(**kwargs)

    def process_default_value(self, default_value):
        if self._regex.match(default_value) is None:
            raise Exception('The default value does not match with the regular expression!')
        return default_value

    def add_json_schema_entry(self, schema):
        super(RegexType, self).add_json_schema_entry(schema)
        schema[self._column_name]['pattern'] = self._regex.pattern

    def to_json_representation(self):
        json_value = super(RegexType, self).to_json_representation()
        json_value['regex'] = self._regex_text
        return json_value

    def process_sql_parameters(self, array):
        array[2] = 'string'  # Store as string

    def get_extra_sql_statement(self):
        return ",2,'" + self._regex_text + "',NULL"


class LimitedTextType(TextType):
    """Covers: CHAR, CHARACTER, VARCHAR, CHARACTER VARYING"""

    def __init__(self, **kwargs):
        self._limit = kwargs['limit']
        super(LimitedTextType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(LimitedTextType, self).add_json_schema_entry(schema)
        schema[self._column_name]['maxLength'] = self._limit

    def process_default_value(self, default_value):
        str_value = str(default_value)
        parsed_len = len(str_value)
        if parsed_len > self._limit:
            raise Exception('The default string\'s length is longer than the limit: %d > %d!'
                            % (parsed_len, self._limit))
        return default_value

    def to_json_representation(self):
        json_value = super(LimitedTextType, self).to_json_representation()
        json_value['limit'] = self._limit
        return json_value

    def process_sql_parameters(self, array):
        array[2] += ''.join(["(", str(self._limit), ")"])  # add the limit restriction after the type declaration


class EnumType(TextType):
    """Covers: Enum of strings"""

    def __init__(self, **kwargs):
        self._values = kwargs['values']
        super(EnumType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(EnumType, self).add_json_schema_entry(schema)
        schema[self._column_name]['enum'] = self._values

    def process_default_value(self, default_value):
        str_value = str(default_value)
        if str_value not in self._values:
            raise Exception('The default value is not present in the enumeration!')
        return default_value

    def to_json_representation(self):
        json_value = super(EnumType, self).to_json_representation()
        json_value['values'] = self._values
        return json_value

    def process_sql_parameters(self, array):
        array[2] = 'char(' + str(max([len(y) for y in self._values])) + ')'  # char with max length of enum values

    def get_extra_sql_statement(self):
        return ",3,'" + ENUM_TYPE_SEPARATOR.join(self._values) + "',NULL"


class INetSixType(TextType):
    """Covers: Inet6"""

    def __init__(self, **kwargs):
        super(INetSixType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(INetSixType, self).add_json_schema_entry(schema)
        schema[self._column_name]['format'] = 'ipv6'

    # http://stackoverflow.com/questions/166132/maximum-length-of-the-textual-representation-of-an-ipv6-address
    def process_sql_parameters(self, array):
        array[2] = 'char(45)'

    def get_extra_sql_statement(self):
        return ",4,NULL,NULL"


class INetType(StreamDataType):
    """Covers: Inet"""

    def __init__(self, **kwargs):
        super(INetType, self).__init__(**kwargs)

    def get_nullable_constant(self):
        return "0"  # has to trick because it is impossible to get a null value from a valid IPv4 address in MonetDB

    def add_json_schema_entry(self, schema):
        super(INetType, self).add_json_schema_entry(schema)
        schema[self._column_name]['pattern'] = IPV4_REGEX

    def process_next_value(self, entry, counter, parameters, errors):
        array = [0, 0, 0, 0, 0, 0, 0, 0]  # according to MonetDB's source code

        if entry == self.get_nullable_constant():
            array[7] = 1
        else:
            components = split(r'[./]+', entry)
            for i in xrange(4):
                array[i] = int(components[i])
            if len(components) > 4:  # if it has a mask add it to the array
                array[4] = int(components[4])
            else:
                array[4] = 32
        return array

    def pack_parsed_values(self, extracted_values, counter, parameters):
        extracted_values = list(chain(*extracted_values))
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter << 3) + 'B', *extracted_values)  # arrays of 8 uchars


class UUIDType(StreamDataType):
    """Covers: UUID"""

    def __init__(self, **kwargs):
        super(UUIDType, self).__init__(**kwargs)

    def get_nullable_constant(self):
        return NIL_UUID

    def add_json_schema_entry(self, schema):
        super(UUIDType, self).add_json_schema_entry(schema)
        schema[self._column_name]['pattern'] = UUID_REGEX

    def process_next_value(self, entry, counter, parameters, errors):
        array = 16 * [0]
        j = 0
        s = 0

        for i in xrange(16):
            if j in (8, 12, 16, 20):  # do nothing with the dashes
                s += 1
            next_char = ord(entry[s])
            if 48 <= next_char <= 57:  # between '0' and '9'
                array[i] = next_char - 48
            elif 97 <= next_char <= 102:  # between 'a' and 'f'
                array[i] = next_char - 87
            elif 65 <= next_char <= 70:  # between 'A' and 'F'
                array[i] = next_char - 55
            s += 1
            j += 1
            array[i] <<= 4
            next_char = ord(entry[s])
            if 48 <= next_char <= 57:  # between '0' and '9'
                array[i] |= next_char - 48
            elif 97 <= next_char <= 102:  # between 'a' and 'f'
                array[i] |= next_char - 87
            elif 65 <= next_char <= 70:  # between 'A' and 'F'
                array[i] |= next_char - 55
            s += 1
            j += 1
        return array

    def pack_parsed_values(self, extracted_values, counter, parameters):
        extracted_values = list(chain(*extracted_values))
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter << 4) + 'B', *extracted_values)  # arrays of 16 uchars


class BooleanType(StreamDataType):
    """Covers: BOOL[EAN]"""

    def __init__(self, **kwargs):
        super(BooleanType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(BooleanType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'boolean'

    def process_default_value(self, default_value):
        if bool(default_value):
            return 1
        else:
            return 0

    def get_nullable_constant(self):
        return INT8_MIN

    def process_next_value(self, entry, counter, parameters, errors):
        if entry == self.get_nullable_constant():
            return entry
        if bool(entry):
            return 1
        return 0

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter) + 'b', *extracted_values)


class NumberBaseType(StreamDataType):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self._minimum = kwargs.get('minimum', None)
        self._maximum = kwargs.get('maximum', None)
        if self._minimum is not None and self._maximum is not None and self._minimum > self._maximum:
            raise Exception('The minimum value is higher than the maximum!')
        super(NumberBaseType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(NumberBaseType, self).add_json_schema_entry(schema)
        if self._minimum is not None:  # we can add numbers to schema
            schema[self._column_name]['minimum'] = self._minimum
        if self._maximum is not None:
            schema[self._column_name]['maximum'] = self._maximum

    def process_default_value(self, default_value):
        if self._minimum is not None and self._maximum is None and default_value < self._minimum:
            raise Exception('The default value is less than the minimum: %s < %s!' % (default_value, self._minimum))
        elif self._minimum is None and self._maximum is not None and default_value > self._maximum:
            raise Exception('The default value is higher than the maximum: %s > %s!' % (default_value, self._maximum))
        elif self._minimum is not None and self._maximum is not None and default_value < self._minimum:
            raise Exception('The default value is out of range: %s < %s!' % (default_value, self._minimum))
        elif self._minimum is not None and self._maximum is not None and default_value > self._maximum:
            raise Exception('The default value is out of range: %s > %s!' % (default_value, self._maximum))
        return default_value

    def to_json_representation(self):
        json_value = super(NumberBaseType, self).to_json_representation()
        if self._minimum is not None:
            json_value['minimum'] = self._minimum
        if self._maximum is not None:
            json_value['maximum'] = self._maximum
        return json_value

    def get_extra_sql_statement(self):
        res_str = [",NULL,"]
        if self._minimum is not None:
            res_str.append("'" + str(self._minimum) + "',")
        else:
            res_str.append("NULL,")
        if self._maximum is not None:
            res_str.append("'" + str(self._maximum) + "'")
        else:
            res_str.append("NULL")
        return ''.join(res_str)


class SmallIntegerType(NumberBaseType):
    """Covers: TINYINT, SMALLINT, INT[EGER], WRD, BIGINT"""

    def __init__(self, **kwargs):
        self._pack_sym = {'tinyint': 'b', 'smallint': 'h', 'int': 'i', 'integer': 'i', 'bigint': 'q'}\
            .get(kwargs['type'])
        self._nullable_constant = {'tinyint': INT8_MIN, 'smallint': INT16_MIN, 'int': INT32_MIN, 'integer': INT32_MIN,
                                   'bigint': INT64_MIN}.get(kwargs['type'])
        super(SmallIntegerType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(SmallIntegerType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'integer'

    def get_nullable_constant(self):
        return self._nullable_constant

    def process_next_value(self, entry, counter, parameters, errors):
        return entry

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter) + self._pack_sym, *extracted_values)


class HugeIntegerType(NumberBaseType):
    """Covers: HUGEINT"""

    def __init__(self, **kwargs):
        super(HugeIntegerType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(HugeIntegerType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'integer'

    def get_nullable_constant(self):
        return 0x80000000000000000000000000000000

    def process_next_value(self, entry, counter, parameters, errors):
        return [entry & 0xFFFFFFFFFFFFFFFF, (entry >> 64) & 0xFFFFFFFFFFFFFFFF]

    def pack_parsed_values(self, extracted_values, counter, parameters):
        extracted_values = list(chain(*extracted_values))
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter << 1) + 'Q', *extracted_values)


class FloatType(NumberBaseType):
    """Covers: REAL, FLOAT and DOUBLE"""

    def __init__(self, **kwargs):
        this_type = kwargs['type']
        self._pack_sym = {'real': 'f', 'float': 'd', 'double': 'd', 'double precision': 'd'}.get(this_type)
        self._nullable_constant = {'real': FLOAT_NAN, 'float': DOUBLE_NAN, 'double': DOUBLE_NAN,
                                   'double precision': DOUBLE_NAN}.get(this_type)
        super(FloatType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(FloatType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'number'

    def get_nullable_constant(self):
        return self._nullable_constant

    def process_next_value(self, entry, counter, parameters, errors):
        return entry

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter) + self._pack_sym, *extracted_values)


class DecimalType(NumberBaseType):
    """Covers: DECIMAL and NUMERIC"""

    def __init__(self, **kwargs):
        self._precision = kwargs.get('precision', 18)
        self._scale = kwargs.get('scale', 3)
        if self._scale > self._precision:
            raise Exception('The scale must be between 0 and the precision!')

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

        self._nullable_constant = {'b': INT8_MIN, 'h': INT16_MIN, 'i': INT32_MIN, 'q': INT64_MIN,
                                   'Q': 0x80000000000000000000000000000000}.get(self._pack_sym)

        super(DecimalType, self).__init__(**kwargs)
        if self._default_value is not None:
            self.check_value_precision(self._default_value, 'default')
        if self._minimum is not None:
            self.check_value_precision(self._minimum, 'minimum')
        if self._maximum is not None:
            self.check_value_precision(self._maximum, 'maximum')

    def check_value_precision(self, value, text):
        if value != self._nullable_constant:
            number_digits = int(ceil(log10(abs(value))))
            if number_digits > self._precision:
                raise Exception('Too many digits on %s: %s > %s!' % (text, number_digits, self._precision))

    def add_json_schema_entry(self, schema):
        super(DecimalType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'number'

    def get_nullable_constant(self):
        return self._nullable_constant

    def process_next_value(self, entry, counter, parameters, errors):
        self.check_value_precision(entry, 'entry')
        if self._pack_sym != 'Q':
            return int(entry)
        else:
            var = int(entry)
            return [var & 0xFFFFFFFFFFFFFFFF, (var >> 64) & 0xFFFFFFFFFFFFFFFF]

    def pack_parsed_values(self, extracted_values, counter, parameters):
        if self._pack_sym == 'Q':
            extracted_values = list(chain(*extracted_values))
            counter <<= 1  # duplicate the counter for packing
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter) + self._pack_sym, *extracted_values)

    def to_json_representation(self):
        json_value = super(DecimalType, self).to_json_representation()
        json_value['precision'] = self._precision
        json_value['scale'] = self._scale
        return json_value

    def process_sql_parameters(self, array):  # add the precision and scale
        array[2] += ''.join(["(", str(self._precision), ",", str(self._scale), ")"])


class BaseDateTimeType(StreamDataType):  # The validation of time variables can't be done on the schema
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        if 'minimum' in kwargs:
            self._minimum_text = kwargs['minimum']  # to show on json representation
            self._minimum = self.parse_entry(kwargs['minimum'])
        else:
            self._minimum = None
        if 'maximum' in kwargs:
            self._maximum_text = kwargs['maximum']  # to show on json representation
            self._maximum = self.parse_entry(kwargs['maximum'])
        else:
            self._maximum = None
        if self._minimum is not None and self._maximum is not None and self._minimum > self._maximum:
            raise Exception('The minimum value is higher than the maximum!')
        super(BaseDateTimeType, self).__init__(**kwargs)

    def get_nullable_constant(self):  # had to add this to bypass python's datetime limits
        return "0"

    @abstractmethod
    def parse_entry(self, entry):  # convert the string to datetime instance
        pass

    @abstractmethod
    def pack_next_value(self, parsed, counter, parameters, errors):
        pass

    def process_default_value(self, default_value):
        parsed_val = self.parse_entry(default_value)  # Process the default value as the others
        if self._minimum is not None and self._maximum is None and parsed_val < self._minimum:
            raise Exception('The default value is less than the minimum: %s < %s!'
                            % (default_value, self._minimum_text))
        elif self._minimum is None and self._maximum is not None and parsed_val > self._maximum:
            raise Exception('The default value is higher than the maximum: %s > %s!'
                            % (default_value, self._maximum_text))
        elif self._minimum is not None and self._maximum is not None and parsed_val < self._minimum:
            raise Exception('The default value is out of range: %s < %s!' % (default_value, self._minimum_text))
        elif self._minimum is not None and self._maximum is not None and parsed_val > self._maximum:
            raise Exception('The default value is out of range: %s > %s!' % (default_value, self._maximum_text))
        return default_value

    def process_next_value(self, entry, counter, parameters, errors):
        if entry == self.get_nullable_constant():  # have to do this trick due to Python datetime limitations
            return self.pack_next_value(None, counter, parameters, errors)
        parsed = self.parse_entry(entry)
        if self._minimum is not None and self._maximum is None and parsed < self._minimum:
            errors[counter] = 'The value is higher than the minimum: %s < %s!' % (parsed, self._minimum_text)
        elif self._minimum is None and self._maximum is not None and parsed > self._maximum:
            errors[counter] = 'The value is higher than the maximum: %s > %s!' % (parsed, self._maximum_text)
        elif self._minimum is not None and self._maximum is not None and parsed < self._minimum:
            errors[counter] = 'The value is out of range: %s < %s!' % (parsed, self._minimum_text)
        elif self._minimum is not None and self._maximum is not None and parsed > self._maximum:
            errors[counter] = 'The value is out of range: %s > %s!' % (parsed, self._maximum_text)
        return self.pack_next_value(parsed, counter, parameters, errors)

    def to_json_representation(self):
        json_value = super(BaseDateTimeType, self).to_json_representation()
        if self._minimum is not None:
            json_value['minimum'] = self._minimum_text
        if self._maximum is not None:
            json_value['maximum'] = self._maximum_text
        return json_value

    def get_extra_sql_statement(self):
        res_str = [",NULL,"]
        if self._minimum is not None:
            res_str.append("'" + self._minimum_text + "',")
        else:
            res_str.append("NULL,")
        if self._maximum is not None:
            res_str.append("'" + self._maximum_text + "'")
        else:
            res_str.append("NULL")
        return ''.join(res_str)


class DateType(BaseDateTimeType):  # Stored as an uint with the number of days since day 1 of month 1 (Jan) from year 0
    """Covers: DATE"""

    def __init__(self, **kwargs):
        super(DateType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(DateType, self).add_json_schema_entry(schema)
        schema[self._column_name]['format'] = 'date'

    def parse_entry(self, entry):
        return datetime.strptime(entry, "%Y-%m-%d")

    def pack_next_value(self, parsed, counter, parameters, errors):
        if parsed is None:
            return 0x80000000
        day0 = deepcopy(parsed).replace(year=1, month=1, day=1)
        # the minyear in python is 1, but for the representation is 0, so why the add
        return int((parsed - day0).days) + 366

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter) + 'I', *extracted_values)


class TimeType(BaseDateTimeType):  # Stored as an uint with the number of milliseconds since hour 00:00:00
    """Covers: TIME"""

    def __init__(self, **kwargs):
        super(TimeType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(TimeType, self).add_json_schema_entry(schema)
        schema[self._column_name]['pattern'] = TIME_REGEX

    def parse_entry(self, entry):
        parsed = parser.parse(entry)
        string = parsed.strftime("%z")
        delta = timedelta(hours=int(string[1:3]), minutes=int(string[3:5]))
        if string[0] == '-':
            delta = -delta
        parsed = parsed.replace(tzinfo=None) - delta
        return parsed

    def pack_next_value(self, parsed, counter, parameters, errors):
        if parsed is None:
            return 0x80000000
        hour0 = deepcopy(parsed).replace(hour=0, minute=0, second=0, microsecond=0)
        delta = parsed - hour0
        return int(delta.total_seconds()) * 1000 + int(delta.microseconds) / 1000

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter) + 'I', *extracted_values)

    def to_json_representation(self):
        json_value = super(TimeType, self).to_json_representation()
        if self._data_type == TIME_WITH_TIMEZONE_TYPE_INTERNAL:
            json_value['type'] = TIME_WITH_TIMEZONE_TYPE_EXTERNAL
        return json_value


class TimestampType(BaseDateTimeType):  # It is represented with the two integers from time and date
    """Covers: TIMESTAMP"""

    def __init__(self, **kwargs):
        super(TimestampType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(TimestampType, self).add_json_schema_entry(schema)
        schema[self._column_name]['format'] = 'date-time'

    def parse_entry(self, entry):
        parsed = parser.parse(entry)
        string = parsed.strftime("%z")
        delta = timedelta(hours=int(string[1:3]), minutes=int(string[3:5]))
        if string[0] == '-':
            delta = -delta
        parsed = parsed.replace(tzinfo=None) - delta
        return parsed

    def pack_next_value(self, parsed, counter, parameters, errors):
        if parsed is None:
            return [0, 0x80000000]
        hour0 = deepcopy(parsed).replace(hour=0, minute=0, second=0, microsecond=0)
        day0 = deepcopy(parsed).replace(year=1, month=1, day=1)
        days = int((parsed - day0).days) + 366
        delta = parsed - hour0
        milliseconds = int(delta.total_seconds()) * 1000 + int(delta.microseconds) / 1000
        return [milliseconds, days]

    def pack_parsed_values(self, extracted_values, counter, parameters):
        concat_array = list(chain(*extracted_values))
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter << 1) + 'I', *concat_array)

    def to_json_representation(self):
        json_value = super(TimestampType, self).to_json_representation()
        if self._data_type == TIMESTAMP_WITH_TIMEZONE_TYPE_INTERNAL:
            json_value['type'] = TIMESTAMP_WITH_TIMEZONE_TYPE_EXTERNAL
        return json_value


class IntervalType(NumberBaseType):
    """Covers: INTERVAL"""

    def __init__(self, **kwargs):
        interval = kwargs['type'][9:].split(" to ")[-1]
        self._multiplier = {'second': 1000, 'minute': 60000, 'hour': 3600000, 'day': 86400000, 'month': 1, 'year': 12}\
            .get(interval)
        self._nullable_constant = {'second': INT64_MIN, 'minute': INT64_MIN, 'hour': INT64_MIN, 'day': INT64_MIN,
                                   'month': INT32_MIN, 'year': INT32_MIN}.get(interval)
        self._pack_sym = {'second': 'q', 'minute': 'q', 'hour': 'q', 'day': 'q', 'month': 'i', 'year': 'i'}\
            .get(interval)
        super(IntervalType, self).__init__(**kwargs)

    def add_json_schema_entry(self, schema):
        super(IntervalType, self).add_json_schema_entry(schema)
        schema[self._column_name]['type'] = 'integer'

    def get_nullable_constant(self):
        return self._nullable_constant

    def process_next_value(self, entry, counter, parameters, errors):
        if entry == self.get_nullable_constant():
            return entry
        return entry * self._multiplier

    def pack_parsed_values(self, extracted_values, counter, parameters):
        return struct.pack(LITTLE_ENDIAN_ALIGNMENT + str(counter) + self._pack_sym, *extracted_values)
