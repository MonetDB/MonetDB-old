import struct
from abc import ABCMeta, abstractmethod
from datetime import date, time, datetime
from os import SEEK_END

from dateutil.relativedelta import relativedelta

LITTLE_ENDIAN_ALIGNMENT = '<'  # for now is little-endian for Intel CPU's

NIL_STRING = "\200\n"  # added newline for performance
NIL_UUID = "00000000-0000-0000-0000-000000000000"

INT8_MIN = -128
INT16_MIN = -32768
INT32_MIN = -2147483648
INT64_MIN = -9223372036854775808
FLOAT_NAN = struct.unpack('f', '\xff\xff\x7f\xff')[0]
DOUBLE_NAN = struct.unpack('d', '\xff\xff\xff\xff\xff\xff\xef\xff')[0]


class StreamDataType(object):
    """MonetDB's data types for reading base class"""
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self._column_name = kwargs['name']  # name of the column
        self._data_type = kwargs['type']  # SQL name of the type
        self._default_value = kwargs['default']  # default value text
        self._is_nullable = kwargs['nullable']  # is nullable

    def get_column_name(self):
        return self._column_name

    def is_file_mode_binary(self):
        return True

    @abstractmethod
    def calculate_file_size(self, fp):
        pass

    @abstractmethod
    def skip_tuples(self, fp, offset):
        pass

    @abstractmethod
    def read_next_batch(self, fp, limit):
        pass

    def read_next_tuples(self, file_name, offset, read_size):
        open_string = 'r'
        if not self.is_file_mode_binary():
            open_string += 'u'
        file_pointer = open(file_name, open_string)

        if offset > 0:
            self.skip_tuples(file_pointer, offset)

        results = self.read_next_batch(file_pointer, read_size)
        file_pointer.close()
        return results

    def to_json_representation(self):  # get a json representation of the data type while checking the stream's info
        return (('name', self._column_name), ('type', self._data_type),
                ('default', self._default_value), ('nullable', self._is_nullable))


class TextType(StreamDataType):
    """Covers: CLOB and URL"""

    def __init__(self, **kwargs):
        super(TextType, self).__init__(**kwargs)
        self._nullable_constant = NIL_STRING

    def is_file_mode_binary(self):
        return False

    def calculate_file_size(self, fp):
        return sum(1 for _ in fp)

    def skip_tuples(self, fp, offset):
        for _ in xrange(offset):
            next(fp)

    def read_next_batch(self, fp, limit):
        array = []
        for _ in xrange(limit):
            next_line = next(fp)
            if next_line == self._nullable_constant:
                array.append(None)
            else:
                array.append(next_line[:-1])  # remove newline
        return array


class LimitedTextType(TextType):
    """Covers: CHAR and VARCHAR"""

    def __init__(self, **kwargs):
        super(LimitedTextType, self).__init__(**kwargs)
        self._limit = kwargs['digits']

    def to_json_representation(self):
        return super(LimitedTextType, self).to_json_representation() + (('limit', self._limit),)


class INetType(StreamDataType):
    """Covers: Inet"""

    def __init__(self, **kwargs):
        super(INetType, self).__init__(**kwargs)

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell() >> 3  # a inet has a size of 8 bytes

    def skip_tuples(self, fp, offset):
        fp.seek(offset << 3)

    def read_next_batch(self, fp, limit):
        results = []
        read_size = limit << 3
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(read_size) + 'B', fp.read(read_size))
        iterator = iter(array)

        for _ in xrange(limit):
            next_ipv4 = [next(iterator) for _ in xrange(8)]
            if next_ipv4[7] == 1:  # check nil value
                results.append(None)
            else:
                parsed_ip = '.'.join([str(next_ipv4[0]), str(next_ipv4[1]), str(next_ipv4[2]), str(next_ipv4[3])])
                results.append(parsed_ip + '/' + str(next_ipv4[4]))
        return results


class UUIDType(StreamDataType):
    """Covers: UUID"""

    def __init__(self, **kwargs):
        super(UUIDType, self).__init__(**kwargs)

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell() >> 4  # a inet has a size of 16 bytes

    def skip_tuples(self, fp, offset):
        fp.seek(offset << 4)

    def read_next_batch(self, fp, limit):
        results = []
        read_size = limit << 4
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(read_size) + 'B', fp.read(read_size))
        iterator = iter(array)

        for _ in xrange(limit):
            next_uuid = []
            for i in xrange(20):
                if i in (4, 7, 10, 13):
                    next_uuid.append("-")
                else:
                    next_uuid.append("%02x" % next(iterator))
            built_uuid = ''.join(next_uuid)
            if built_uuid == NIL_UUID:
                built_uuid = None
            results.append(built_uuid)
        return results


class BooleanType(StreamDataType):
    """Covers: BOOLEAN"""

    def __init__(self, **kwargs):
        super(BooleanType, self).__init__(**kwargs)

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell()  # a boolean has a size of 1 byte

    def skip_tuples(self, fp, offset):
        fp.seek(offset)

    def read_next_batch(self, fp, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + 'b', fp.read(limit))
        return map(lambda x: None if x == INT8_MIN else bool(x), array)


class SmallIntegerType(StreamDataType):
    """Covers: TINYINT, SMALLINT, INTEGER, BIGINT"""

    def __init__(self, **kwargs):
        super(SmallIntegerType, self).__init__(**kwargs)
        self._pack_sym = {'tinyint': 'b', 'smallint': 'h', 'int': 'i', 'integer': 'i', 'bigint': 'q'} \
            .get(self._data_type)
        self._size = {'tinyint': 0, 'smallint': 1, 'int': 2, 'integer': 2, 'bigint': 3}.get(self._data_type)
        self._nullable_constant = {'tinyint': INT8_MIN, 'smallint': INT16_MIN, 'int': INT32_MIN, 'integer': INT32_MIN,
                                   'bigint': INT64_MIN}.get(self._data_type)

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell() >> self._size

    def skip_tuples(self, fp, offset):
        fp.seek(offset << self._size)

    def read_next_batch(self, fp, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + self._pack_sym, fp.read(limit << self._size))
        return map(lambda x: None if x == self._nullable_constant else x, array)


class HugeIntegerType(StreamDataType):
    """Covers: HUGEINT"""

    def __init__(self, **kwargs):
        super(HugeIntegerType, self).__init__(**kwargs)

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell() >> 4

    def skip_tuples(self, fp, offset):
        fp.seek(offset << 4)

    def read_next_batch(self, fp, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit << 1) + 'Q', fp.read(limit << 4))
        results = []
        iterator = iter(array)  # has to iterate two values at once, so use iterator
        for value in iterator:
            second_value = next(iterator)
            next_huge = (second_value << 64) + value
            if next_huge == 0x80000000000000000000000000000000:
                results.append(None)
            else:
                results.append(next_huge)
        return results


class FloatType(StreamDataType):
    """Covers: REAL, DOUBLE"""

    def __init__(self, **kwargs):
        super(FloatType, self).__init__(**kwargs)
        self._pack_sym = {'real': 'f', 'float': 'd', 'double': 'd'}.get(self._data_type)
        self._size = {'real': 2, 'float': 3, 'double': 3}.get(self._data_type)
        self._nullable_constant = {'real': FLOAT_NAN, 'float': DOUBLE_NAN, 'double': DOUBLE_NAN}.get(self._data_type)

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell() >> self._size

    def skip_tuples(self, fp, offset):
        fp.seek(offset << self._size)

    def read_next_batch(self, fp, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + self._pack_sym, fp.read(limit << self._size))
        return map(lambda x: None if x == self._nullable_constant else x, array)


class DecimalType(StreamDataType):
    """Covers: DECIMAL"""

    def __init__(self, **kwargs):
        super(DecimalType, self).__init__(**kwargs)
        self._precision = kwargs['digits']
        self._scale = kwargs['scale']

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
        self._size = {'b': 0, 'h': 1, 'i': 2, 'q': 3, 'Q': 4}.get(self._pack_sym)

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell() >> self._size

    def skip_tuples(self, fp, offset):
        fp.seek(offset << self._size)

    def read_next_batch(self, fp, limit):
        if self._pack_sym != 'Q':
            array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + self._pack_sym, fp.read(limit << self._size))
            return map(lambda x: None if x == self._nullable_constant else float(x), array)
        else:
            array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit << 1) + self._pack_sym,
                                  fp.read(limit << self._size))
            results = []
            iterator = iter(array)  # has to iterate two values at once, so use iterator
            for value in iterator:
                second_value = next(iterator)
                next_huge_decimal = (second_value << 64) + value
                if next_huge_decimal == self._nullable_constant:
                    results.append(None)
                else:
                    results.append(float(next_huge_decimal))
            return results

    def to_json_representation(self):
        return super(DecimalType, self).to_json_representation() + \
               (('precision', self._precision), ('scale', self._scale),)

year_delta = relativedelta(years=1)


class DateType(StreamDataType):  # Stored as an uint with the number of days since day 1 of month 1 (Jan) from year 0
    """Covers: DATE"""

    def __init__(self, **kwargs):
        super(DateType, self).__init__(**kwargs)

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell() >> 2

    def skip_tuples(self, fp, offset):
        fp.seek(offset << 2)

    def read_next_batch(self, fp, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + 'I', fp.read(limit << 2))
        results = []
        for value in array:
            if value == 0x80000000:
                results.append(None)
            else:  # dates in python start on year 1, so we must subtract one year
                results.append((date.fromordinal(value) - year_delta).isoformat())
        return results


class TimeType(StreamDataType):  # Stored as an uint with the number of milliseconds since hour 00:00:00
    """Covers: TIME"""

    def __init__(self, **kwargs):
        super(TimeType, self).__init__(**kwargs)
        if self._data_type == 'timetz':
            self._data_type = 'time with time zone'

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell() >> 2

    def skip_tuples(self, fp, offset):
        fp.seek(offset << 2)

    def read_next_batch(self, fp, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + 'I', fp.read(limit << 2))
        results = []
        for value in array:
            if value == 0x80000000:
                results.append(None)
            else:
                hour = value / 3600000
                aux1 = hour * 3600000
                minute = (value - aux1) / 60000
                aux2 = minute * 60000
                second = (value - aux1 - aux2) / 1000
                millisecond = value - aux1 - aux2 - (second * 1000)
                results.append(time(hour=hour, minute=minute, second=second, microsecond=millisecond * 1000)
                               .isoformat())
        return results


class TimestampType(StreamDataType):  # It is represented with the two integers from time and date
    """Covers: TIMESTAMP"""

    def __init__(self, **kwargs):
        super(TimestampType, self).__init__(**kwargs)
        if self._data_type == 'timestamptz':
            self._data_type = 'timestamp with time zone'

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell() >> 3

    def skip_tuples(self, fp, offset):
        fp.seek(offset << 3)

    def read_next_batch(self, fp, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit << 1) + 'I', fp.read(limit << 3))
        results = []
        iterator = iter(array)  # has to iterate two values at once, so use iterator

        for value in iterator:
            second_value = next(iterator)
            if value == 0 and second_value == 0x80000000:
                results.append(None)
            else:  # dates in python start on year 1, so we must subtract one year
                read_date = date.fromordinal(second_value) - year_delta
                hour = value / 3600000
                aux1 = hour * 3600000
                minute = (value - aux1) / 60000
                aux2 = minute * 60000
                second = (value - aux1 - aux2) / 1000
                millisecond = value - aux1 - aux2 - (second * 1000)
                results.append(datetime.combine(read_date, time(hour=hour, minute=minute, second=second,
                                                                microsecond=millisecond)).isoformat())
        return results

INTERVALS_DICTIONARY = {1: "interval year", 2: "interval year to month", 3: "interval month", 4: "interval day",
                        5: "interval day to hour", 6: "interval day to minute", 7: "interval day to second",
                        8: "interval hour", 9: "interval hour to minute", 10: "interval hour to second",
                        11: "interval minute", 12: "interval minute to second", 13: "interval second"}


class IntervalType(StreamDataType):
    """Covers: INTERVAL"""

    def __init__(self, **kwargs):
        super(IntervalType, self).__init__(**kwargs)
        if kwargs['digits'] < 4:
            self._pack_sym = 'i'
            self._nullable_constant = INT32_MIN
            self._size = 2
        else:
            self._pack_sym = 'q'
            self._nullable_constant = INT64_MIN
            self._size = 3
        self._data_type = INTERVALS_DICTIONARY.get(kwargs['digits'])

    def calculate_file_size(self, fp):
        fp.seek(0, SEEK_END)
        return fp.tell() >> self._size

    def skip_tuples(self, fp, offset):
        fp.seek(offset << self._size)

    def read_next_batch(self, fp, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + self._pack_sym, fp.read(limit << self._size))
        return map(lambda x: None if x == self._nullable_constant else int(x), array)
