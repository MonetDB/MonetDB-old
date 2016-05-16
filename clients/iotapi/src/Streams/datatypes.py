import struct
from datetime import date, time, datetime

from abc import ABCMeta, abstractmethod
from dateutil.relativedelta import relativedelta

ALIGNMENT = '<'  # for now is little-endian for Intel CPU's

NIL_STRING = "\200\n"  # added newline for performance

INT8_MIN = 0x80
INT16_MIN = 0x8000
INT32_MIN = 0x80000000
INT64_MIN = 0x8000000000000000
INT64_MAX = 0xFFFFFFFFFFFFFFFF
INT128_MIN = 0x80000000000000000000000000000000

FLOAT_NAN = struct.unpack('f', '\xff\xff\x7f\xff')[0]
DOUBLE_NAN = struct.unpack('d', '\xff\xff\xff\xff\xff\xff\xef\xff')[0]


class StreamDataType(object):
    """MonetDB's data types for validation base class"""
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self._column_name = kwargs['name']  # name of the column
        self._data_type = kwargs['type']  # SQL name of the type
        # self._location = kwargs['location'] + '.tail'  # Location of the file

    def is_file_mode_binary(self):
        return True

    @abstractmethod
    def read_next_batch(self, file_pointer, count):
        return []


class TextType(StreamDataType):
    """Covers: CHAR, VARCHAR, CLOB"""

    def __init__(self, **kwargs):
        super(TextType, self).__init__(**kwargs)
        self._nullable_constant = NIL_STRING

    def is_file_mode_binary(self):
        return False

    def read_next_batch(self, file_pointer, count):
        array = file_pointer.readlines()
        return map(lambda x: None if x == self._nullable_constant else x[:-1], array)


class BooleanType(StreamDataType):
    """Covers: BOOLEAN"""

    def __init__(self, **kwargs):
        super(BooleanType, self).__init__(**kwargs)
        self._nullable_constant = INT8_MIN

    def read_next_batch(self, file_pointer, count):
        array = struct.unpack(ALIGNMENT + str(count) + 'b', file_pointer.read(count))
        return map(lambda x: None if x == self._nullable_constant else bool(x), array)


class SmallIntegerType(StreamDataType):
    """Covers: TINYINT, SMALLINT, INTEGER, BIGINT"""

    def __init__(self, **kwargs):
        super(SmallIntegerType, self).__init__(**kwargs)
        self._pack_sym = {'tinyint': 'b', 'smallint': 'h', 'int': 'i', 'integer': 'i', 'bigint': 'q'} \
            .get(self._data_type)
        self._size = struct.calcsize(self._pack_sym)
        self._nullable_constant = {'tinyint': INT8_MIN, 'smallint': INT16_MIN, 'int': INT32_MIN, 'integer': INT32_MIN,
                                   'bigint': INT64_MIN}.get(self._data_type)

    def read_next_batch(self, file_pointer, count):
        array = struct.unpack(ALIGNMENT + str(count) + self._pack_sym, file_pointer.read(count * self._size))
        return map(lambda x: None if x == self._nullable_constant else int(x), array)


class HugeIntegerType(StreamDataType):
    """Covers: HUGEINT"""

    def __init__(self, **kwargs):
        super(HugeIntegerType, self).__init__(**kwargs)
        self._nullable_constant = INT128_MIN

    def read_next_batch(self, file_pointer, count):  # [entry & INT64_MAX, (entry >> 64) & INT64_MAX]
        array = struct.unpack(ALIGNMENT + str(count << 1) + 'Q', file_pointer.read(count << 3))
        results = []
        iterator = iter(array)  # has to iterate two values at once, so use iterator
        for value in iterator:
            next_huge = value + (next(iterator) << 64)
            if next_huge == self._nullable_constant:
                results.append(None)
            else:
                results.append(int(next_huge))
        return results


class FloatType(StreamDataType):
    """Covers: REAL, DOUBLE"""

    def __init__(self, **kwargs):
        super(FloatType, self).__init__(**kwargs)
        self._pack_sym = {'real': 'f', 'float': 'd', 'double': 'd'}.get(self._data_type)
        self._size = struct.calcsize(self._pack_sym)
        self._nullable_constant = {'real': FLOAT_NAN, 'float': DOUBLE_NAN, 'double': DOUBLE_NAN}.get(self._data_type)

    def read_next_batch(self, file_pointer, count):
        array = struct.unpack(ALIGNMENT + str(count) + self._pack_sym, file_pointer.read(count * self._size))
        return map(lambda x: None if x == self._nullable_constant else float(x), array)


class DecimalType(StreamDataType):
    """Covers: DECIMAL"""

    def __init__(self, **kwargs):
        super(DecimalType, self).__init__(**kwargs)

        self._pack_sym = {'1': 'b', '2': 'h', '4': 'i', '8': 'q', '16': 'Q'}.get(kwargs['typewidth'])
        self._nullable_constant = {'1': INT8_MIN, '2': INT16_MIN, '4': INT32_MIN, '8': INT64_MIN, '16': INT128_MIN} \
            .get(kwargs['typewidth'])
        self._size = struct.calcsize(self._pack_sym)
        if self._pack_sym == 'Q':
            self._size <<= 1  # has to read two values at once

    def read_next_batch(self, file_pointer, count):
        array = struct.unpack(ALIGNMENT + str(count) + self._pack_sym, file_pointer.read(count * self._size))
        if self._pack_sym != 'Q':
            return map(lambda x: None if x == self._nullable_constant else float(x), array)

        results = []
        iterator = iter(array)  # has to iterate two values at once, so use iterator
        for value in iterator:
            next_huge_decimal = value + (next(iterator) << 64)
            if next_huge_decimal == self._nullable_constant:
                results.append(None)
            else:
                results.append(next_huge_decimal)
        return results


class DateType(StreamDataType):  # Stored as an uint with the number of days since day 1 of month 1 (Jan) from year 0
    """Covers: DATE"""

    def __init__(self, **kwargs):
        super(DateType, self).__init__(**kwargs)
        self._nullable_constant = INT32_MIN

    def read_next_batch(self, file_pointer, count):
        array = struct.unpack(ALIGNMENT + str(count) + 'I', file_pointer.read(count << 2))
        results = []
        for value in array:
            if value == self._nullable_constant:
                results.append(None)
            else:  # dates in python start on year 1, so we must subtract one year
                results.append((date.fromordinal(value) - relativedelta(years=1)).isoformat())
        return results


class TimeType(StreamDataType):  # Stored as an uint with the number of milliseconds since hour 00:00:00
    """Covers: TIME"""

    def __init__(self, **kwargs):
        super(TimeType, self).__init__(**kwargs)
        self._nullable_constant = INT32_MIN

    def read_next_batch(self, file_pointer, count):
        array = struct.unpack(ALIGNMENT + str(count) + 'I', file_pointer.read(count << 2))
        results = []
        for value in array:
            if value == self._nullable_constant:
                results.append(None)
            else:
                div1, milliseconds = divmod(value, 1000)
                div2, second = divmod(div1, 60)
                hour, minute = divmod(div2, 60)
                results.append(time(hour=hour, minute=minute, second=second, microsecond=milliseconds * 1000)
                               .isoformat())
        return results


class TimestampType(StreamDataType):  # it's represented with the two integers from time and date
    """Covers: TIMESTAMP"""

    def __init__(self, **kwargs):
        super(TimestampType, self).__init__(**kwargs)

    def read_next_batch(self, file_pointer, count):
        array = struct.unpack(ALIGNMENT + str(count << 1) + 'I', file_pointer.read(count << 3))
        results = []
        iterator = iter(array)  # has to iterate two values at once, so use iterator

        for value in iterator:
            second_value = next(iterator)
            if value == INT32_MIN and second_value == 0:
                results.append(None)
            else:  # dates in python start on year 1, so we must subtract one year
                read_date = date.fromordinal(value) - relativedelta(years=1)
                div1, milliseconds = divmod(second_value, 1000)
                div2, second = divmod(div1, 60)
                hour, minute = divmod(div2, 60)
                results.append(datetime.combine(read_date, time(hour=hour, minute=minute, second=second,
                                                                microsecond=milliseconds * 1000)).isoformat())
        return results
