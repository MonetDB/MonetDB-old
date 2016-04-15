import os
from collections import defaultdict

import src.Settings.filesystem
from datatypes import TimestampType, DataValidationException
from flushing import TimeBasedFlushing, TupleBasedFlushing
from src.Settings.mapiconnection import mapi_create_stream, mapi_flush_baskets
from src.Utilities.filecreator import create_file_if_not_exists, get_hidden_file_name
from src.Utilities.readwritelock import RWLock

BASKETS_BASE_DIRECTORY = "baskets"
IMP_TIMESTAMP_COLUMN_NAME = 'implicit_timestamp'


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class StreamException(Exception):
    """Exception fired when the validation of a stream insert fails"""

    def __init__(self, messages):
        self.messages = messages  # dictionary of column -> list of error messages


class DataCellStream(object):
    """Representation of the stream for validation"""

    def __init__(self, schema_name, stream_name, flush_method, columns, validation_schema):
        self._schema_name = schema_name  # name of the schema
        self._stream_name = stream_name  # name of the stream
        self._tuples_in_per_basket = 0  # for efficiency
        self._flush_method = flush_method  # instance of StreamFlushingMethod
        self._columns = columns  # dictionary of name -> data_types
        self._timestamps_handler = TimestampType(name="implicit_timestamp", type="timestamp")  # implicit timestamp
        self._validation_schema = validation_schema  # json validation schema for the inserts
        self._monitor = RWLock()  # baskets lock to protect files (the server is multi-threaded)
        self._base_path = os.path.join(src.Settings.filesystem.filesystem_location, BASKETS_BASE_DIRECTORY,
                                       schema_name, stream_name)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)
            self._baskets_counter = 1
        else:
            dirs = [int(name) for name in os.listdir(self._base_path) if represents_int(name)]
            if dirs:
                for elem in dirs:  # for each directory found, flush it
                    dir_path = os.path.join(self._base_path, str(elem))
                    mapi_flush_baskets(self._schema_name, self._stream_name, dir_path)
                self._baskets_counter = max(dirs) + 1  # the current basket number will be the next one
            else:
                self._baskets_counter = 1
        self._current_base_path = os.path.join(self._base_path, str(self._baskets_counter))
        os.makedirs(self._current_base_path)
        for key in self._columns.keys():  # create the files for the columns and timestamp
            create_file_if_not_exists(os.path.join(self._current_base_path, key), hidden=True)
        create_file_if_not_exists(os.path.join(self._current_base_path, IMP_TIMESTAMP_COLUMN_NAME), hidden=True)

    def start_stream(self):
        column_string = ','.join([column.create_stream_sql() for column in self._columns.values()])
        mapi_create_stream(self._schema_name, self._stream_name, column_string)
        if isinstance(self._flush_method, TimeBasedFlushing):  # start the time based flush on another thread
            self._flush_method.init_local_thread(self)  # pass the time_based_flush method

    def stop_stream(self):
        if isinstance(self._flush_method, TimeBasedFlushing):
            self._flush_method.stop_local_thread()  # stop the time flushing thread
        self.time_based_flush(last=True)  # flush the baskets while deleting (also works for tuple based flushing)

    def get_data_dictionary(self):
        self._monitor.acquire_read()
        dic = {'schema': self._schema_name, 'name': self._stream_name,
               'tuples_in': self._tuples_in_per_basket, 'flush': self._flush_method.get_dictionary_info(),
               'columns': {key: value.to_json_representation() for key, value in self._columns.iteritems()}}
        self._monitor.release()
        return dic

    def flush_baskets(self, last=False):  # the monitor has to be acquired in write mode before running this method!!!
        mapi_flush_baskets(self._schema_name, self._stream_name, self._current_base_path)
        if not last:  # when stopping the stream, we don't want to continue to create more baskets files
            self._tuples_in_per_basket = 0
            self._baskets_counter += 1
            self._current_base_path = os.path.join(self._base_path, str(self._baskets_counter))
            os.makedirs(self._current_base_path)
            for key in self._columns.keys():
                create_file_if_not_exists(os.path.join(self._current_base_path, key), hidden=True)
            create_file_if_not_exists(os.path.join(self._current_base_path, IMP_TIMESTAMP_COLUMN_NAME), hidden=True)

    def time_based_flush(self, last=False):
        self._monitor.acquire_write()
        if self._tuples_in_per_basket > 0:
            self.flush_baskets(last)
        self._monitor.release()

    def validate_and_insert(self, new_data, timestamp):
        self._validation_schema.validate(new_data)  # validate the stream's schema first

        batch_errors = {}  # dictionary of column_name -> array of errors
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
            raise StreamException(messages=batch_errors)

        transposed_data = defaultdict(list)  # transpose the inserts to benefit the MonetDB's column storage
        for entry in new_data:
            for key, value in entry.iteritems():
                transposed_data[key].append(value)  # dictionary of column_name -> array of values to insert

        for key, values in transposed_data.iteritems():
            data_type = self._columns[key]  # get the correspondent column
            try:
                transposed_data[key] = data_type.process_values(values)  # convert into binary
            except DataValidationException as ex:
                batch_errors[key] = ex.message

        if batch_errors:
            raise StreamException(messages=batch_errors)

        # prepare variables outside the lock for more parallelism
        total_tuples = len(new_data)
        bin_value = self._timestamps_handler.process_values([timestamp])
        timestamps_binary_array = ''.join([bin_value for _ in xrange(total_tuples)])
        is_flushing_tuple_based = isinstance(self._flush_method, TupleBasedFlushing)
        # supposing that the flushing method never changes we can do this outside the lock

        self._monitor.acquire_write()

        for key, inserts in transposed_data.iteritems():  # now write the binary data
            # open basket in binary mode and append the new entries
            basket_fp = open(get_hidden_file_name(os.path.join(self._current_base_path, key)), 'ab')
            basket_fp.write(inserts)
            basket_fp.flush()
            basket_fp.close()

        # write the implicit timestamp
        time_basket_fp = open(get_hidden_file_name(os.path.join(self._current_base_path,
                                                                IMP_TIMESTAMP_COLUMN_NAME)), 'ab')
        time_basket_fp.write(timestamps_binary_array)
        time_basket_fp.flush()
        time_basket_fp.close()

        self._tuples_in_per_basket += total_tuples
        if is_flushing_tuple_based and self._tuples_in_per_basket >= self._flush_method.limit:
            self.flush_baskets(last=False)

        self._monitor.release()
