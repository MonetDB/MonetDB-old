import struct
from abc import ABCMeta, abstractmethod
from datetime import date, time, datetime

from dateutil.relativedelta import relativedelta

LITTLE_ENDIAN_ALIGNMENT = '<'  # for now is little-endian for Intel CPU's

NIL_STRING = "\200\n"  # added newline for performance
NIL_UUID = "00000000-0000-0000-0000-000000000000"

INT8_MIN = 0x80
INT16_MIN = 0x8000
INT32_MIN = 0x80000000
INT64_MIN = 0x8000000000000000
INT64_MAX = 0xFFFFFFFFFFFFFFFF
INT128_MIN = 0x80000000000000000000000000000000

FLOAT_NAN = struct.unpack('f', '\xff\xff\x7f\xff')[0]
DOUBLE_NAN = struct.unpack('d', '\xff\xff\xff\xff\xff\xff\xef\xff')[0]


class StreamDataType(object):
    """MonetDB's data types for reading base class"""
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self._column_name = kwargs['name']  # name of the column
        self._data_type = kwargs['type']  # SQL name of the type

    def is_file_mode_binary(self):
        return True

    @abstractmethod
    def skip_tuples(self, file_pointer, offset):
        pass

    @abstractmethod
    def read_next_batch(self, file_pointer, limit):
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


class TextType(StreamDataType):
    """Covers: CHAR, VARCHAR, CLOB and URL"""

    def __init__(self, **kwargs):
        super(TextType, self).__init__(**kwargs)
        self._nullable_constant = NIL_STRING

    def is_file_mode_binary(self):
        return False

    def skip_tuples(self, file_pointer, offset):
        for _ in xrange(offset):
            next(file_pointer)

    def read_next_batch(self, file_pointer, limit):
        array = []
        for _ in xrange(limit):
            next_line = next(file_pointer)
            if next_line == self._nullable_constant:
                array.append(None)
            else:
                array.append(next_line[:-1])  # remove newline
        return array


class INetType(StreamDataType):
    """Covers: Inet"""

    def __init__(self, **kwargs):
        super(INetType, self).__init__(**kwargs)

    def skip_tuples(self, file_pointer, offset):
        file_pointer.seek(offset << 3)

    def read_next_batch(self, file_pointer, limit):
        results = []
        read_size = limit << 3
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(read_size) + 'B', file_pointer.read(read_size))
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

    def skip_tuples(self, file_pointer, offset):
        file_pointer.seek(offset << 4)

    def read_next_batch(self, file_pointer, limit):
        results = []
        read_size = limit << 4
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(read_size) + 'B', file_pointer.read(read_size))
        iterator = iter(array)

        for _ in xrange(limit):
            next_uuid = ''.join(map(lambda x: "%02x" % x, [next(iterator) for _ in xrange(16)]))
            next_uuid = ''.join([next_uuid[:8], '-', next_uuid[8:12], '-', next_uuid[12:16], '-', next_uuid[16:20],
                                 '-', next_uuid[20:]])
            if next_uuid == NIL_UUID:
                next_uuid = None
            results.append(next_uuid)
        return results


class BooleanType(StreamDataType):
    """Covers: BOOLEAN"""

    def __init__(self, **kwargs):
        super(BooleanType, self).__init__(**kwargs)
        self._nullable_constant = INT8_MIN

    def skip_tuples(self, file_pointer, offset):
        file_pointer.seek(offset)

    def read_next_batch(self, file_pointer, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + 'b', file_pointer.read(limit))
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

    def skip_tuples(self, file_pointer, offset):
        file_pointer.seek(offset * self._size)

    def read_next_batch(self, file_pointer, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + self._pack_sym,
                              file_pointer.read(limit * self._size))
        return map(lambda x: None if x == self._nullable_constant else int(x), array)


class HugeIntegerType(StreamDataType):
    """Covers: HUGEINT"""

    def __init__(self, **kwargs):
        super(HugeIntegerType, self).__init__(**kwargs)
        self._nullable_constant = INT128_MIN

    def skip_tuples(self, file_pointer, offset):
        file_pointer.seek(offset << 4)

    def read_next_batch(self, file_pointer, limit):  # [entry & INT64_MAX, (entry >> 64) & INT64_MAX]
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit << 1) + 'Q', file_pointer.read(limit << 4))
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

    def skip_tuples(self, file_pointer, offset):
        file_pointer.seek(offset * self._size)

    def read_next_batch(self, file_pointer, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + self._pack_sym,
                              file_pointer.read(limit * self._size))
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

    def skip_tuples(self, file_pointer, offset):
        file_pointer.seek(offset * self._size)

    def read_next_batch(self, file_pointer, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + self._pack_sym,
                              file_pointer.read(limit * self._size))
        if self._pack_sym != 'Q':
            return map(lambda x: None if x == self._nullable_constant else float(x), array)
        else:
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

    def skip_tuples(self, file_pointer, offset):
        file_pointer.seek(offset << 2)

    def read_next_batch(self, file_pointer, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + 'I', file_pointer.read(limit << 2))
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

    def skip_tuples(self, file_pointer, offset):
        file_pointer.seek(offset << 2)

    def read_next_batch(self, file_pointer, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit) + 'I', file_pointer.read(limit << 2))
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

    def skip_tuples(self, file_pointer, offset):
        file_pointer.seek(offset << 3)

    def read_next_batch(self, file_pointer, limit):
        array = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + str(limit << 1) + 'I', file_pointer.read(limit << 3))
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
