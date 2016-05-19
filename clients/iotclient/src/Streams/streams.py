import os
import struct
from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict

from Settings.filesystem import get_baskets_base_location, get_host_identifier
from Settings.iotlogger import add_log
from Settings.mapiconnection import mapi_create_stream, mapi_flush_baskets
from Utilities.filecreator import create_file_if_not_exists
from Utilities.readwritelock import RWLock

from Utilities.customthreading import PeriodicalThread
from datatypes import TimestampType, TextType, DataValidationException, LITTLE_ENDIAN_ALIGNMENT

IMPLICIT_TIMESTAMP_COLUMN_NAME = 'implicit_timestamp'
Timestamps_Handler = TimestampType(name=IMPLICIT_TIMESTAMP_COLUMN_NAME, type="timestamp")  # timestamp
Extra_columns_SQL = [Timestamps_Handler.create_stream_sql()]  # array for SQL creation

HOST_IDENTIFIER_COLUMN_NAME = 'host_identifier'
Hostname_Bin_Value = None
BASKETS_COUNT_FILE = 'count'


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def init_streams_hosts():
    global Hostname_Bin_Value

    host_identifier = get_host_identifier()
    if host_identifier is not None:
        hosts_handler = TextType(name=HOST_IDENTIFIER_COLUMN_NAME, type="text")  # host_identifier
        Hostname_Bin_Value = hosts_handler.process_values([host_identifier])
        Extra_columns_SQL.append(hosts_handler.create_stream_sql())


class StreamException(Exception):
    """Exception fired when the validation of a stream insert fails"""

    def __init__(self, message):
        self.message = message  # dictionary of column -> list of error messages


class BaseIOTStream(object):
    """Representation of a stream for validation"""
    __metaclass__ = ABCMeta

    def __init__(self, schema_name, stream_name, columns, validation_schema, created=False):
        self._schema_name = schema_name  # name of the schema
        self._stream_name = stream_name  # name of the stream
        self._tuples_in_per_basket = 0  # for efficiency
        self._columns = columns  # dictionary of name -> data_types
        self._validation_schema = validation_schema  # json validation schema for the inserts
        self._monitor = RWLock()  # baskets lock to protect files (the server is multi-threaded)
        self._base_path = os.path.join(get_baskets_base_location(), schema_name, stream_name)

        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)
            self._baskets_counter = 1
        else:
            dirs = [int(name) for name in os.listdir(self._base_path) if represents_int(name)]
            if dirs:
                for elem in dirs:  # for each directory found, flush it
                    dir_path = os.path.join(self._base_path, str(elem))
                    mapi_flush_baskets(self._schema_name, self._stream_name, dir_path)
                self._baskets_counter = max(dirs) + 1  # increment current basket number
            else:
                self._baskets_counter = 1
        self._current_base_path = os.path.join(self._base_path, str(self._baskets_counter))
        os.makedirs(self._current_base_path)

        for key in self._columns.keys():  # create files for the columns, timestamp and hostname
            create_file_if_not_exists(os.path.join(self._current_base_path, key))
        create_file_if_not_exists(os.path.join(self._current_base_path, IMPLICIT_TIMESTAMP_COLUMN_NAME))
        if Hostname_Bin_Value is not None:
            create_file_if_not_exists(os.path.join(self._current_base_path, HOST_IDENTIFIER_COLUMN_NAME))

        if created:  # when the stream is reloaded from the config file, the create SQL statement is not sent
            sql_array = [column.create_stream_sql() for column in self._columns.values()]
            mapi_create_stream(self._schema_name, self._stream_name, ', '.join(sql_array + Extra_columns_SQL))

    def get_schema_name(self):
        return self._schema_name

    def get_stream_name(self):
        return self._stream_name

    def start_stream(self):
        add_log(20, 'Started stream %s.%s' % (self._schema_name, self._stream_name))

    def stop_stream(self):
        self._monitor.acquire_write()
        try:
            self.flush_baskets(last=True)
        except:
            pass
        self._monitor.release()
        add_log(20, 'Stopped stream %s.%s' % (self._schema_name, self._stream_name))

    @abstractmethod
    def get_flushing_dictionary(self):
        return {}

    def get_data_dictionary(self, include_number_tuples=False):
        self._monitor.acquire_read()
        dic = OrderedDict({'schema': self._schema_name, 'stream': self._stream_name,
                           'flushing': self.get_flushing_dictionary(),
                           'columns': [value.to_json_representation() for value in self._columns.values()]})

        #  when writing the data to config file, we don't serialize the number of tuples inserted on the baskets
        if include_number_tuples:
            dic['tuples_inserted_per_basket'] = self._tuples_in_per_basket
        self._monitor.release()
        return dic

    def flush_baskets(self, last=False):  # the monitor has to be acquired in write mode before running this method!!!
        # write the tuple count in the basket
        basket_counter_file_pointer = open(os.path.join(self._current_base_path, BASKETS_COUNT_FILE), "w+")
        basket_counter_file_pointer.write(struct.pack(LITTLE_ENDIAN_ALIGNMENT + "1i", self._tuples_in_per_basket))
        basket_counter_file_pointer.flush()
        basket_counter_file_pointer.close()
        mapi_flush_baskets(self._schema_name, self._stream_name, self._current_base_path)

        if not last:  # when stopping the stream, we don't want to continue to create more baskets files
            self._tuples_in_per_basket = 0
            self._baskets_counter += 1
            self._current_base_path = os.path.join(self._base_path, str(self._baskets_counter))
            os.makedirs(self._current_base_path)

            for key in self._columns.keys():
                create_file_if_not_exists(os.path.join(self._current_base_path, key))
            create_file_if_not_exists(os.path.join(self._current_base_path, IMPLICIT_TIMESTAMP_COLUMN_NAME))
            if Hostname_Bin_Value is not None:
                create_file_if_not_exists(os.path.join(self._current_base_path, HOST_IDENTIFIER_COLUMN_NAME))

    def validate_and_insert(self, new_data, timestamp):
        self._validation_schema.validate(new_data)  # validate the stream's schema first

        batch_errors = OrderedDict()  # dictionary of column_name -> array of errors
        column_names = self._columns.keys()
        tuple_counter = 0

        for entry in new_data:
            tuple_counter += 1
            absent_columns = [x for x in column_names if x not in entry.keys()]  # for all columns missing
            for value in absent_columns:
                data_type = self._columns[value]  # get the corresponding data type
                default_value = data_type.get_default_value()
                if default_value is not None:  # set the default value
                    entry[value] = default_value
                elif data_type.is_nullable():  # or the null constant
                    entry[value] = data_type.get_nullable_constant()
                else:
                    batch_errors[value].append('Problem while parsing this column in tuple: ' + str(tuple_counter))

        if batch_errors:
            raise StreamException(message=batch_errors)

        transposed_data = defaultdict(list)  # transpose the inserts to benefit the MonetDB's column storage
        for entry in new_data:
            for key, value in entry.iteritems():
                transposed_data[key].append(value)  # dictionary of column_name -> array of values to insert

        for key, values in transposed_data.iteritems():
            data_type = self._columns[key]  # get the correspondent column
            try:
                transposed_data[key] = data_type.process_values(values)  # convert into binary
            except DataValidationException as ex:
                batch_errors[key] = ex

        if batch_errors:
            raise StreamException(message=batch_errors)

        # prepare variables outside the lock for more parallelism
        total_tuples = len(new_data)

        timestamp_bin_value = Timestamps_Handler.process_values([timestamp])
        timestamps_binary_array = ''.join([timestamp_bin_value for _ in xrange(total_tuples)])

        if Hostname_Bin_Value is not None:  # write the host name if applicable
            hosts_binary_array = ''.join([Hostname_Bin_Value for _ in xrange(total_tuples)])

        self._monitor.acquire_write()
        try:
            for key, inserts in transposed_data.iteritems():  # now write the binary data
                # open basket in binary mode and append the new entries
                basket_fp = open(os.path.join(self._current_base_path, key), 'ab')
                basket_fp.write(inserts)
                basket_fp.flush()
                basket_fp.close()

            # write the implicit timestamp
            time_basket_fp = open(os.path.join(self._current_base_path, IMPLICIT_TIMESTAMP_COLUMN_NAME), 'ab')
            time_basket_fp.write(timestamps_binary_array)
            time_basket_fp.flush()
            time_basket_fp.close()

            if Hostname_Bin_Value is not None:  # the variable never changes
                hosts_basket_fp = open(os.path.join(self._current_base_path, HOST_IDENTIFIER_COLUMN_NAME), 'ab')
                hosts_basket_fp.write(hosts_binary_array)
                hosts_basket_fp.flush()
                hosts_basket_fp.close()

            self._tuples_in_per_basket += total_tuples
        except BaseException as ex:
            self._monitor.release()
            add_log(50, ex)
        else:
            self._monitor.release()
            add_log(20, 'Inserted %d tuples to stream %s.%s' % (total_tuples, self._schema_name, self._stream_name))


class TupleBasedStream(BaseIOTStream):
    """Stream with tuple based flushing"""

    def __init__(self, schema_name, stream_name, columns, validation_schema, created, limit):
        super(TupleBasedStream, self).__init__(schema_name, stream_name, columns, validation_schema, created)
        self._limit = limit

    def get_flushing_dictionary(self):
        return {'base': 'tuple', 'number': self._limit}

    def validate_and_insert(self, new_data, timestamp):
        super(TupleBasedStream, self).validate_and_insert(new_data, timestamp)
        self._monitor.acquire_write()
        try:
            if self._tuples_in_per_basket >= self._limit:
                self.flush_baskets(last=False)
        except BaseException as ex:
            self._monitor.release()
            add_log(50, ex)
        else:
            self._monitor.release()
            add_log(20, 'Flushed stream %s.%s baskets' % (self._schema_name, self._stream_name))


class TimeBasedStream(BaseIOTStream):
    """Stream with time based flushing"""

    def __init__(self, schema_name, stream_name, columns, validation_schema, created, interval, time_unit):
        super(TimeBasedStream, self).__init__(schema_name, stream_name, columns, validation_schema, created)
        self._time_unit = time_unit
        self._interval = interval
        if time_unit == "s":
            calc_time = interval
        elif time_unit == "m":
            calc_time = interval * 60
        else:
            calc_time = interval * 3600
        self._local_thread = PeriodicalThread(calc_time, self.time_based_flush)

    def get_flushing_dictionary(self):
        return {'base': 'time', 'unit': self._time_unit, 'interval': self._interval}

    def time_based_flush(self):
        self._monitor.acquire_write()
        try:
            if self._tuples_in_per_basket > 0:
                self.flush_baskets(last=False)
        except BaseException as ex:
            self._monitor.release()
            add_log(50, ex)
        else:
            self._monitor.release()
            add_log(20, 'Flushed stream %s.%s baskets' % (self._schema_name, self._stream_name))

    def start_stream(self):
        self._local_thread.start()  # start the time based flush on another thread
        super(TimeBasedStream, self).start_stream()

    def stop_stream(self):
        self._local_thread.stop() # stop the time flushing thread
        super(TimeBasedStream, self).stop_stream()
