import os

from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
from json import dumps
from .datatypes import TimestampWithTimeZoneType, TextType
from Settings.filesystem import get_baskets_location
from Settings.iotlogger import add_log
from Settings.mapiconnection import close_monetdb_connection, mapi_flush_baskets
from Utilities.filecreator import create_file_if_not_exists
from Utilities.readwritelock import RWLock
from Utilities.customthreading import PeriodicalThread

IMPLICIT_TIMESTAMP_COLUMN_NAME = 'implicit_timestamp'
Implicit_Timestamp_Handler = TimestampWithTimeZoneType(name=IMPLICIT_TIMESTAMP_COLUMN_NAME, type="timestamp")
Implicit_Timestamp_SQL = Implicit_Timestamp_Handler.create_stream_sql()

HOST_IDENTIFIER_COLUMN_NAME = 'hostname_identifier'
Hostname_Binary_Value = None
Hostname_SQL = ""


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def init_streams_hosts(host_identifier):
    global Hostname_Binary_Value, Hostname_SQL

    hosts_handler = TextType(name=HOST_IDENTIFIER_COLUMN_NAME, type="string")
    Hostname_Binary_Value = hosts_handler.process_values([host_identifier])
    Hostname_SQL = hosts_handler.create_stream_sql()


class BaseIOTStream:
    """Representation of a stream for validation"""
    __metaclass__ = ABCMeta

    def __init__(self, schema_name, stream_name, columns, validation_schema, has_timestamp, has_hostname, connection,
                 table_id="", columns_ids=""):
        self._schema_name = schema_name  # name of the schema
        self._stream_name = stream_name  # name of the stream
        self._tuples_in_per_basket = 0  # for efficiency
        self._columns = columns  # dictionary of name -> data_types
        self._validation_schema = validation_schema  # json validation schema for the inserts
        self._has_timestamp = has_timestamp  # add timestamp column or not
        self._has_hostname = has_hostname  # add hostname column or not
        self._connection = connection  # to be init later
        self._table_id = table_id  # for delete statement on table iot.webserverstreams
        self._columns_ids = columns_ids  # for delete statement on table iot.webservercolumns
        self._baskets_lock = RWLock()  # baskets lock to protect files (the server is multi-threaded)
        self._base_path = os.path.join(get_baskets_location(), schema_name, stream_name)

        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)
            self._baskets_counter = 1
        else:
            dirs = [int(name) for name in os.listdir(self._base_path) if represents_int(name)]
            if dirs:
                for elem in dirs:  # for each directory found, flush it
                    dir_path = os.path.join(self._base_path, str(elem))
                    mapi_flush_baskets(self._connection, self._schema_name, self._stream_name, dir_path)
                self._baskets_counter = max(dirs) + 1  # increment current basket number
            else:
                self._baskets_counter = 1
        self._current_base_path = os.path.join(self._base_path, str(self._baskets_counter))
        os.makedirs(self._current_base_path)

        for key in self._columns.keys():  # create files for the columns, timestamp and hostname
            create_file_if_not_exists(os.path.join(self._current_base_path, key))
        if self._has_timestamp:
            create_file_if_not_exists(os.path.join(self._current_base_path, IMPLICIT_TIMESTAMP_COLUMN_NAME))
        if self._has_hostname:
            create_file_if_not_exists(os.path.join(self._current_base_path, HOST_IDENTIFIER_COLUMN_NAME))

    def get_schema_name(self):
        return self._schema_name

    def get_stream_name(self):
        return self._stream_name

    def get_sql_create_statement(self):  # for CREATE STREAM TABLE statement
        base_sql = ','.join([column.create_stream_sql() for column in self._columns.values()])
        if self._has_timestamp:
            base_sql += ',' + Implicit_Timestamp_SQL
        if self._has_hostname:
            base_sql += ',' + Hostname_SQL
        return base_sql

    def get_columns_extra_sql_statements(self):  # for iot.webservercolumns table
        return OrderedDict({key: value.get_extra_sql_statement() for (key, value) in self._columns.iteritems()})

    def set_delete_ids(self, table_id, columns_ids):  # for the delete statement
        self._table_id = table_id
        self._columns_ids = columns_ids

    def get_table_id(self):  # for the delete statement
        return self._table_id

    def get_columns_ids(self):  # for the delete statement
        return self._columns_ids

    @abstractmethod
    def get_webserverstreams_sql_statement(self):  # insert for iot.webserverflushing table
        return ""

    def start_stream(self):
        add_log(20, 'Started stream %s.%s' % (self._schema_name, self._stream_name))

    def stop_stream(self):
        self._baskets_lock.acquire_write()
        try:
            self.flush_baskets(last=True)
            close_monetdb_connection(self._connection)
        except BaseException as ex:
            add_log(50, ex)
        self._baskets_lock.release()
        add_log(20, 'Stopped stream %s.%s' % (self._schema_name, self._stream_name))

    @abstractmethod
    def get_flushing_dictionary(self):  # for information about the stream
        return {}

    def get_data_dictionary(self):
        dic = OrderedDict((('schema', self._schema_name), ('stream', self._stream_name),
                           ('has_timestamp', self._has_timestamp), ('has_hostname', self._has_hostname),
                           ('flushing', self.get_flushing_dictionary())))
        flushing = dic['flushing']
        self._baskets_lock.acquire_read()
        flushing['tuples_inserted_per_basket'] = self._tuples_in_per_basket
        self._baskets_lock.release()
        dic['columns'] = [value.to_json_representation() for value in self._columns.values()]
        return dic

    def flush_baskets(self, last=False):  # the monitor has to be acquired in write mode before running this method!!!
        # write the tuple count in the basket
        mapi_flush_baskets(self._connection, self._schema_name, self._stream_name, self._current_base_path)

        if not last:  # when stopping the stream, we don't want to continue to create more baskets files
            self._tuples_in_per_basket = 0
            self._baskets_counter += 1
            self._current_base_path = os.path.join(self._base_path, str(self._baskets_counter))
            os.makedirs(self._current_base_path)

            for key in self._columns.keys():
                create_file_if_not_exists(os.path.join(self._current_base_path, key))
            if self._has_timestamp:
                create_file_if_not_exists(os.path.join(self._current_base_path, IMPLICIT_TIMESTAMP_COLUMN_NAME))
            if self._has_hostname:
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
            raise Exception(message=dumps(batch_errors))  # dictionary of column -> list of error messages

        transposed_data = defaultdict(list)  # transpose the inserts to benefit the MonetDB's column storage
        for entry in new_data:
            for key, value in entry.iteritems():
                transposed_data[key].append(value)  # dictionary of column_name -> array of values to insert

        for key, values in transposed_data.iteritems():
            data_type = self._columns[key]  # get the correspondent column
            try:
                transposed_data[key] = data_type.process_values(values)  # convert into binary
            except BaseException as ex:
                batch_errors[key] = ex

        if batch_errors:
            raise Exception(message=dumps(batch_errors))  # dictionary of column -> list of error messages

        # prepare variables outside the lock for more parallelism
        total_tuples = len(new_data)

        if self._has_timestamp:
            timestamp_bin_value = Implicit_Timestamp_Handler.process_values([timestamp])
            timestamps_binary_array = ''.join([timestamp_bin_value for _ in xrange(total_tuples)])

        if self._has_hostname:  # write the host name if applicable
            hosts_binary_array = ''.join([Hostname_Binary_Value for _ in xrange(total_tuples)])

        self._baskets_lock.acquire_write()
        try:
            for key, inserts in transposed_data.iteritems():  # now write the binary data
                # open basket in binary mode and append the new entries
                basket_fp = open(os.path.join(self._current_base_path, key), 'ab')
                basket_fp.write(inserts)
                basket_fp.flush()
                basket_fp.close()

            if self._has_timestamp:  # write the implicit timestamp
                time_basket_fp = open(os.path.join(self._current_base_path, IMPLICIT_TIMESTAMP_COLUMN_NAME), 'ab')
                time_basket_fp.write(timestamps_binary_array)
                time_basket_fp.flush()
                time_basket_fp.close()

            if self._has_hostname:  # the variable never changes
                hosts_basket_fp = open(os.path.join(self._current_base_path, HOST_IDENTIFIER_COLUMN_NAME), 'ab')
                hosts_basket_fp.write(hosts_binary_array)
                hosts_basket_fp.flush()
                hosts_basket_fp.close()

            self._tuples_in_per_basket += total_tuples
        except BaseException as ex:
            self._baskets_lock.release()
            add_log(50, ex)
        else:
            self._baskets_lock.release()
            add_log(20, 'Inserted %d tuples to stream %s.%s' % (total_tuples, self._schema_name, self._stream_name))


class TupleBasedStream(BaseIOTStream):
    """Stream with tuple based flushing"""

    def __init__(self, schema_name, stream_name, columns, validation_schema, has_timestamp, has_hostname, connection,
                 table_id, columns_ids, interval):
        super(TupleBasedStream, self).__init__(schema_name, stream_name, columns, validation_schema, has_timestamp,
                                               has_hostname, connection, table_id, columns_ids)
        self._interval = interval

    def get_flushing_dictionary(self):
        return OrderedDict((('base', 'tuple'), ('interval', self._interval)))

    def get_webserverstreams_sql_statement(self):  # insert for iot.webserverflushing table
        return ''.join([",1,", str(self._interval), ",NULL"])

    def validate_and_insert(self, new_data, timestamp):
        super(TupleBasedStream, self).validate_and_insert(new_data, timestamp)
        flag = False
        self._baskets_lock.acquire_write()
        try:
            if self._tuples_in_per_basket >= self._interval:
                self.flush_baskets(last=False)
                flag = True
        except BaseException as ex:
            self._baskets_lock.release()
            add_log(50, ex)
        else:
            self._baskets_lock.release()
            if flag:
                add_log(20, 'Flushed stream %s.%s baskets' % (self._schema_name, self._stream_name))


class TimeBasedStream(BaseIOTStream):
    """Stream with time based flushing"""

    def __init__(self, schema_name, stream_name, columns, validation_schema, has_timestamp, has_hostname, connection,
                 table_id, columns_ids, interval, time_unit):
        super(TimeBasedStream, self).__init__(schema_name, stream_name, columns, validation_schema, has_timestamp,
                                              has_hostname, connection, table_id, columns_ids)
        self._time_unit = time_unit
        self._interval = interval
        if time_unit == "s":
            calc_time = interval
        elif time_unit == "m":
            calc_time = interval * 60
        else:
            calc_time = interval * 3600
        self._local_thread = PeriodicalThread(interval=calc_time, worker_func=self.time_based_flush)

    def get_flushing_dictionary(self):
        return OrderedDict((('base', 'time'), ('interval', self._interval), ('unit', self._time_unit)))

    def get_webserverstreams_sql_statement(self):  # insert for iot.webserverflushing table
        return ''.join([",2,", str(self._interval), ",'", self._time_unit, "'"])

    def time_based_flush(self):
        flag = False
        self._baskets_lock.acquire_write()
        try:
            if self._tuples_in_per_basket > 0:  # flush only when there are tuples in the baskets
                self.flush_baskets(last=False)
                flag = True
        except BaseException as ex:
            self._baskets_lock.release()
            add_log(50, ex)
        else:
            self._baskets_lock.release()
            if flag:
                add_log(20, 'Flushed stream %s.%s baskets' % (self._schema_name, self._stream_name))

    def start_stream(self):
        self._local_thread.start()  # start the time based flush on another thread
        super(TimeBasedStream, self).start_stream()

    def stop_stream(self):
        self._local_thread.stop()  # stop the time flushing thread
        super(TimeBasedStream, self).stop_stream()


class AutoFlushedStream(BaseIOTStream):
    """Stream with flush every time a new batch is inserted"""

    def __init__(self, schema_name, stream_name, columns, validation_schema, has_timestamp, has_hostname, connection,
                 table_id, columns_ids):
        super(AutoFlushedStream, self).__init__(schema_name, stream_name, columns, validation_schema, has_timestamp,
                                                has_hostname, connection, table_id, columns_ids)

    def get_flushing_dictionary(self):
        return OrderedDict({'base': 'auto'})

    def get_webserverstreams_sql_statement(self):  # insert for iot.webserverflushing table
        return ",3,NULL,NULL"

    def validate_and_insert(self, new_data, timestamp):
        super(AutoFlushedStream, self).validate_and_insert(new_data, timestamp)
        self._baskets_lock.acquire_write()
        try:
            self.flush_baskets(last=False)  # there will be always items on the baskets
        except BaseException as ex:
            self._baskets_lock.release()
            add_log(50, ex)
        else:
            self._baskets_lock.release()
            add_log(20, 'Flushed stream %s.%s baskets' % (self._schema_name, self._stream_name))
