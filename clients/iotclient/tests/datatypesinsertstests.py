import os
import random
import uuid
import unittest

from abc import ABCMeta, abstractmethod
from faker import Factory
from pytz import timezone


def load_src(name, fpath):
    import os
    import imp
    p = fpath if os.path.isabs(fpath) \
        else os.path.join(os.path.dirname(__file__), fpath)
    return imp.load_source(name, p)

load_src("jsonschemas", "../src/Streams/jsonschemas.py")
load_src("datatypes", "../src/Streams/datatypes.py")

from datatypes import TextType, URLType, INetType, UUIDType, BooleanType, SmallIntegerType,\
    HugeIntegerType, FloatType, DecimalType, DateType, TimeWithoutTimeZoneType, TimeWithTimeZoneType,\
    TimestampWithoutTimeZoneType, TimestampWithTimeZoneType

faker = Factory.create()


class DataTypesTest(unittest.TestCase):
    __metaclass__ = ABCMeta

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(DataTypesTest, self).__init__()
        self._mapi_connection = mapi_connection
        self._number_inserts = number_inserts
        self._data_type = self.get_data_type()
        self._serializer = self.get_serializer()
        self._temp_path = os.path.join(temp_path, self._data_type.replace(' ', '_'))

    @abstractmethod
    def get_data_type(self):
        return 'none'

    @abstractmethod
    def get_serializer(self):
        pass

    @abstractmethod
    def get_next_batch(self, number_inserts):
        pass

    @abstractmethod
    def convert_batch_to_sql(self, batch):
        pass

    def runTest(self):
        try:
            self._mapi_connection.execute("DROP TABLE sql_insert")
            self._mapi_connection.execute("DROP TABLE binary_insert")
        except:
            pass

        self._mapi_connection.execute("CREATE TABLE sql_insert (val " + self.get_data_type() + ")")
        self._mapi_connection.execute("CREATE TABLE binary_insert (val " + self.get_data_type() + ")")

        # make SQL inserts
        next_batch = self.get_next_batch(self._number_inserts)
        self._mapi_connection.execute("INSERT INTO sql_insert VALUES " +
                                      ','.join("(" + v + ")" for v in self.convert_batch_to_sql(next_batch)))

        # make binary inserts
        binary_array = self._serializer.process_values(next_batch)
        with open(self._temp_path, 'w+b') as fp:
            fp.write(binary_array)
            fp.flush()
        self._mapi_connection.execute("COPY BINARY INTO binary_insert FROM ('" + self._temp_path + "')")

        cursor = self._mapi_connection.cursor()
        cursor.execute("SELECT val FROM sql_insert")
        sql_inserts = cursor.fetchall()
        cursor.execute("SELECT val FROM binary_insert")
        binary_inserts = cursor.fetchall()

        self._mapi_connection.execute("DROP TABLE sql_insert")
        self._mapi_connection.execute("DROP TABLE binary_insert")

        try:
            os.remove(self._temp_path)
        except:
            pass
        self.assertListEqual(sql_inserts, binary_inserts)  # the lists must be equal!!


class BaseStringText(DataTypesTest):
    __metaclass__ = ABCMeta

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(BaseStringText, self).__init__(mapi_connection, number_inserts, temp_path)

    def convert_batch_to_sql(self, batch):
        return ["'" + val + "'" for val in batch]


class TextTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(TextTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'clob'

    def get_serializer(self):
        return TextType(**{'name': 'val', 'type': 'clob'})

    def get_next_batch(self, number_inserts):
        return [faker.sentence() for _ in xrange(number_inserts)]


class URLTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(URLTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'url'

    def get_serializer(self):
        return URLType(**{'name': 'val', 'type': 'url'})

    def get_next_batch(self, number_inserts):
        return [faker.uri() for _ in xrange(number_inserts)]

    #@unittest.skip("Data conversion problem")
    #def runTest(self):
    #    super(URLTest, self).runTest()


class INetTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(INetTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'inet'

    def get_serializer(self):
        return INetType(**{'name': 'val', 'type': 'inet'})

    def get_next_batch(self, number_inserts):
        return [faker.ipv4() for _ in xrange(number_inserts)]


class UUIDTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(UUIDTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'uuid'

    def get_serializer(self):
        return UUIDType(**{'name': 'val', 'type': 'uuid'})

    def get_next_batch(self, number_inserts):
        return [str(uuid.uuid4()) for _ in xrange(number_inserts)]


class BooleanTest(DataTypesTest):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(BooleanTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'boolean'

    def get_serializer(self):
        return BooleanType(**{'name': 'val', 'type': 'boolean'})

    def get_next_batch(self, number_inserts):
        return [faker.boolean() for _ in xrange(number_inserts)]

    def convert_batch_to_sql(self, batch):
        return ["1" if val else "0" for val in batch]


class BaseIntegerTest(DataTypesTest):
    __metaclass__ = ABCMeta

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(BaseIntegerTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_serializer(self):
        return SmallIntegerType(**{'name': 'val', 'type': self.get_data_type()})

    def convert_batch_to_sql(self, batch):
        return [str(val) for val in batch]


class TinyIntegerTest(BaseIntegerTest):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(TinyIntegerTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'tinyint'

    def get_next_batch(self, number_inserts):
        return [random.randint(-127, 127) for _ in xrange(number_inserts)]


class SmallIntegerTest(BaseIntegerTest):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(SmallIntegerTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'smallint'

    def get_next_batch(self, number_inserts):
        return [random.randint(-32768, 32767) for _ in xrange(number_inserts)]


class IntegerTest(BaseIntegerTest):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(IntegerTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'integer'

    def get_next_batch(self, number_inserts):
        return [random.randint(-2147483648, 2147483647) for _ in xrange(number_inserts)]


class BigIntegerTest(BaseIntegerTest):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(BigIntegerTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'bigint'

    def get_next_batch(self, number_inserts):
        return [random.randint(-9223372036854775808, 9223372036854775807) for _ in xrange(number_inserts)]


class HugeIntegerTest(BaseIntegerTest):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(HugeIntegerTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'hugeint'

    def get_serializer(self):
        return HugeIntegerType(**{'name': 'val', 'type': 'hugeint'})

    def get_next_batch(self, number_inserts):
        return [random.randint(0x8000000000000000000000000000000,
                               0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF) for _ in xrange(number_inserts)]


class RealPointTest(BaseIntegerTest):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(RealPointTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'real'

    def get_serializer(self):
        return FloatType(**{'name': 'val', 'type': 'real'})

    def get_next_batch(self, number_inserts):
        return [random.uniform(1.40e-45, 3.40e+38) for _ in xrange(number_inserts)]


class FloatPointTest(BaseIntegerTest):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(FloatPointTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'float'

    def get_serializer(self):
        return FloatType(**{'name': 'val', 'type': 'float'})

    def get_next_batch(self, number_inserts):
        return [random.uniform(4.94e-324, 1.79e+308) for _ in xrange(number_inserts)]

    @unittest.skip("Float conversion problem")
    def runTest(self):
        super(FloatPointTest, self).runTest()


class DecimalTest(BaseIntegerTest):

    def __init__(self, mapi_connection, number_inserts, temp_path, precision, scale):
        self._precision = precision
        self._scale = scale
        super(DecimalTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return "decimal (" + str(self._precision) + "," + str(self._scale) + ")"

    def get_serializer(self):
        return DecimalType(**{'name': 'val', 'type': 'decimal', 'precision': self._precision, 'scale': self._scale})

    def get_next_batch(self, number_inserts):
        range_start = -1 * ((10 ** (self._precision - 1)) - 1)
        range_end = (10 ** (self._precision - 1)) - 1
        return [random.randint(range_start, range_end) for _ in xrange(number_inserts)]

    @unittest.skip("Float conversion problem")
    def runTest(self):
        super(DecimalTest, self).runTest()


class DateTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(DateTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'date'

    def get_serializer(self):
        return DateType(**{'name': 'val', 'type': 'date'})

    def get_next_batch(self, number_inserts):
        return [faker.date() for _ in xrange(number_inserts)]


class TimeWithoutTimezoneTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(TimeWithoutTimezoneTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'time'

    def get_serializer(self):
        return TimeWithoutTimeZoneType(**{'name': 'val', 'type': 'time'})

    def get_next_batch(self, number_inserts):
        return [faker.date_time().time().strftime('%H:%M:%S.%f%z') for _ in xrange(number_inserts)]


class TimeWithTimezoneTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(TimeWithTimezoneTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'time with time zone'

    def get_serializer(self):
        return TimeWithTimeZoneType(**{'name': 'val', 'type': 'time with time zone'})

    def get_next_batch(self, number_inserts):
        return [faker.date_time(tzinfo=timezone(faker.timezone())).strftime('%H:%M:%S.%f%z')
                for _ in xrange(number_inserts)]


class TimestampWithoutTimezoneTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(TimestampWithoutTimezoneTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'timestamp'

    def get_serializer(self):
        return TimestampWithoutTimeZoneType(**{'name': 'val', 'type': 'timestamp'})

    def get_next_batch(self, number_inserts):
        return [faker.iso8601(tzinfo=None) for _ in xrange(number_inserts)]


class TimestampWithTimezoneTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(TimestampWithTimezoneTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'timestamp with time zone'

    def get_serializer(self):
        return TimestampWithTimeZoneType(**{'name': 'val', 'type': 'timestamp with time zone'})

    def get_next_batch(self, number_inserts):
        return [faker.iso8601(tzinfo=timezone(faker.timezone())) for _ in xrange(number_inserts)]


class NullablesTest(unittest.TestCase):
    __metaclass__ = ABCMeta

    def __init__(self, mapi_connection, temp_path):
        super(NullablesTest, self).__init__()
        self._mapi_connection = mapi_connection
        self._data_type = self.get_data_type()
        self._temp_path = os.path.join(temp_path, self._data_type.replace(' ', '_'))
        self._data_type = self.get_data_type()

    @abstractmethod
    def get_data_type(self):
        return 'none'

    @abstractmethod
    def get_null_value(self):
        return 'none'

    def runTest(self):
        try:
            self._mapi_connection.execute("DROP TABLE sql_insert")
            self._mapi_connection.execute("DROP TABLE binary_insert")
        except:
            pass

        self._mapi_connection.execute("CREATE TABLE sql_insert (val " + self.get_data_type() + ")")
        self._mapi_connection.execute("CREATE TABLE binary_insert (val " + self.get_data_type() + ")")

        # make the null value sql insert
        self._mapi_connection.execute("INSERT INTO sql_insert VALUES (null)")

        # make the null value binary insert
        with open(self._temp_path, 'w+b') as fp:
            fp.write(self.get_null_value())
            fp.flush()
        self._mapi_connection.execute("COPY BINARY INTO binary_insert FROM ('" + self._temp_path + "')")

        cursor = self._mapi_connection.cursor()
        cursor.execute("SELECT val FROM sql_insert")
        sql_inserts = cursor.fetchall()
        cursor.execute("SELECT val FROM binary_insert")
        binary_inserts = cursor.fetchall()

        self._mapi_connection.execute("DROP TABLE sql_insert")
        self._mapi_connection.execute("DROP TABLE binary_insert")

        try:
            os.remove(self._temp_path)
        except:
            pass
        self.assertListEqual(sql_inserts, binary_inserts)  # the lists must be equal!!


class NullableTextTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableTextTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'clob'

    def get_null_value(self):
        serializer = TextType(**{'name': 'val', 'type': 'clob'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableURLTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableURLTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'url'

    def get_null_value(self):
        serializer = URLType(**{'name': 'val', 'type': 'url'})
        return serializer.process_values([serializer.get_nullable_constant()])

    #@unittest.skip("Data conversion problem")
    #def runTest(self):
    #    super(NullableURLTest, self).runTest()


class NullableINetTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableINetTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'inet'

    def get_null_value(self):
        serializer = INetType(**{'name': 'val', 'type': 'inet'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableUUIDTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableUUIDTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'uuid'

    def get_null_value(self):
        serializer = UUIDType(**{'name': 'val', 'type': 'uuid'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableBooleanTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableBooleanTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'boolean'

    def get_null_value(self):
        serializer = BooleanType(**{'name': 'val', 'type': 'boolean'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableTinyIntegerTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableTinyIntegerTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'tinyint'

    def get_null_value(self):
        serializer = SmallIntegerType(**{'name': 'val', 'type': 'tinyint'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableSmallIntegerTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableSmallIntegerTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'smallint'

    def get_null_value(self):
        serializer = SmallIntegerType(**{'name': 'val', 'type': 'smallint'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableIntegerTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableIntegerTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'integer'

    def get_null_value(self):
        serializer = SmallIntegerType(**{'name': 'val', 'type': 'integer'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableBigIntegerTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableBigIntegerTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'bigint'

    def get_null_value(self):
        serializer = SmallIntegerType(**{'name': 'val', 'type': 'bigint'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableHugeIntegerTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableHugeIntegerTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'hugeint'

    def get_null_value(self):
        serializer = HugeIntegerType(**{'name': 'val', 'type': 'hugeint'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableRealPointTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableRealPointTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'real'

    def get_null_value(self):
        serializer = FloatType(**{'name': 'val', 'type': 'real'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableFloatPointTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableFloatPointTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'float'

    def get_null_value(self):
        serializer = FloatType(**{'name': 'val', 'type': 'float'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableDecimalTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path, precision, scale):
        self._precision = precision
        self._scale = scale
        super(NullableDecimalTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return "decimal (" + str(self._precision) + "," + str(self._scale) + ")"

    def get_null_value(self):
        serial = DecimalType(**{'name': 'val', 'type': 'decimal', 'precision': self._precision, 'scale': self._scale})
        return serial.process_values([serial.get_nullable_constant()])


class NullableDateTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableDateTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'date'

    def get_null_value(self):
        serializer = DateType(**{'name': 'val', 'type': 'date'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableTimeWithoutTimezoneTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableTimeWithoutTimezoneTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'time'

    def get_null_value(self):
        serializer = TimeWithoutTimeZoneType(**{'name': 'val', 'type': 'time', 'timezone': False})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableTimeWithTimezoneTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableTimeWithTimezoneTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'time with time zone'

    def get_null_value(self):
        serializer = TimeWithTimeZoneType(**{'name': 'val', 'type': 'time with time zone'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableTimestampWithoutTimezoneTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableTimestampWithoutTimezoneTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'timestamp'

    def get_null_value(self):
        serializer = TimestampWithoutTimeZoneType(**{'name': 'val', 'type': 'timestamp'})
        return serializer.process_values([serializer.get_nullable_constant()])


class NullableTimestampWithTimezoneTest(NullablesTest):

    def __init__(self, mapi_connection, temp_path):
        super(NullableTimestampWithTimezoneTest, self).__init__(mapi_connection, temp_path)

    def get_data_type(self):
        return 'timestamp with time zone'

    def get_null_value(self):
        serializer = TimestampWithTimeZoneType(**{'name': 'val', 'type': 'timestamp with time zone'})
        return serializer.process_values([serializer.get_nullable_constant()])
