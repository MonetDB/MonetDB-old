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

from datatypes import TextType, URLType, INetType, UUIDType, BooleanType, SmallIntegerType, \
    HugeIntegerType, FloatType, DecimalType, DateType, TimeType, TimestampType

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
            self._mapi_connection.execute("DROP TABLE sql_insert;")
            self._mapi_connection.execute("DROP TABLE binary_insert;")
        except:
            pass

        self._mapi_connection.execute("CREATE TABLE sql_insert (val " + self.get_data_type() + ");")
        self._mapi_connection.execute("CREATE TABLE binary_insert (val " + self.get_data_type() + ");")

        # make SQL inserts
        next_batch = self.get_next_batch(self._number_inserts)
        self._mapi_connection.execute("INSERT INTO sql_insert VALUES " +
                                      ','.join("(" + v + ")" for v in self.convert_batch_to_sql(next_batch)) + ";")

        # make binary inserts
        binary_array = self._serializer.process_values(next_batch)
        with open(self._temp_path, 'w+b') as fp:
            fp.write(binary_array)
            fp.flush()
        self._mapi_connection.execute("COPY BINARY INTO binary_insert FROM ('" + self._temp_path + "');")

        cursor = self._mapi_connection.cursor()
        cursor.execute("SELECT val FROM sql_insert;")
        sql_inserts = cursor.fetchall()
        cursor.execute("SELECT val FROM binary_insert;")
        binary_inserts = cursor.fetchall()

        self._mapi_connection.execute("DROP TABLE sql_insert;")
        self._mapi_connection.execute("DROP TABLE binary_insert;")

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

    @unittest.skip("Data conversion problem")
    def runTest(self):
        super(URLTest, self).runTest()


class INetTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(INetTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'inet'

    def get_serializer(self):
        return INetType(**{'name': 'val', 'type': 'inet'})

    def get_next_batch(self, number_inserts):
        return [faker.ipv4() for _ in xrange(number_inserts)]

    @unittest.skip("Data conversion problem")
    def runTest(self):
        super(INetTest, self).runTest()


class UUIDTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(UUIDTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'uuid'

    def get_serializer(self):
        return UUIDType(**{'name': 'val', 'type': 'uuid'})

    def get_next_batch(self, number_inserts):
        return [str(uuid.uuid4()) for _ in xrange(number_inserts)]

    @unittest.skip("Data conversion problem")
    def runTest(self):
        super(UUIDTest, self).runTest()

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
        return TimeType(**{'name': 'val', 'type': 'time', 'timezone': False})

    def get_next_batch(self, number_inserts):
        return [faker.date_time().time().strftime('%H:%M:%S.%f%z') for _ in xrange(number_inserts)]


class TimeWithTimezoneTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(TimeWithTimezoneTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'time WITH TIME ZONE'

    def get_serializer(self):
        return TimeType(**{'name': 'val', 'type': 'time', 'timezone': True})

    def get_next_batch(self, number_inserts):
        return [faker.date_time(tzinfo=timezone(faker.timezone())).strftime('%H:%M:%S.%f%z')
                for _ in xrange(number_inserts)]


class TimestampWithoutTimezoneTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(TimestampWithoutTimezoneTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'timestamp'

    def get_serializer(self):
        return TimestampType(**{'name': 'val', 'type': 'timestamp', 'timezone': False})

    def get_next_batch(self, number_inserts):
        return [faker.iso8601(tzinfo=None) for _ in xrange(number_inserts)]


class TimestampWithTimezoneTest(BaseStringText):

    def __init__(self, mapi_connection, number_inserts, temp_path):
        super(TimestampWithTimezoneTest, self).__init__(mapi_connection, number_inserts, temp_path)

    def get_data_type(self):
        return 'timestamp WITH TIME ZONE'

    def get_serializer(self):
        return TimestampType(**{'name': 'val', 'type': 'timestamp', 'timezone': True})

    def get_next_batch(self, number_inserts):
        return [faker.iso8601(tzinfo=timezone(faker.timezone())) for _ in xrange(number_inserts)]
