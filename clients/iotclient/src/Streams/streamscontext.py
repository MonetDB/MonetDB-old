import json

from streamscreator import *
from Settings.filesystem import get_configfile_location
from Utilities.readwritelock import RWLock
from jsonschemas import CONFIG_FILE_SCHEMA
from jsonschema import Draft4Validator, FormatChecker

Config_File_Location = None
Config_File_Validator = None


def init_streams_context():
    global Config_File_Location, Config_File_Validator
    Config_File_Location = get_configfile_location()
    Config_File_Validator = Draft4Validator(CONFIG_FILE_SCHEMA, format_checker=FormatChecker())


class IOTStreamsException(Exception):
    pass


class IOTStreams:
    @classmethod
    def get_context_entry_name(cls, schema_name, stream_name):
        return schema_name + '.' + stream_name

    def __init__(self):
        self._locker = RWLock()
        self._context = collections.OrderedDict()  # dictionary of schema_name + '.' + stream_name -> DataCellStream
        self.reload_config_file()

    def reload_config_file(self):  # the write lock must be set before running this method!!! (except on the beginning)
        with open(get_configfile_location(), 'r') as infile:  # read the config searching for existing streams
            data = json.load(infile)  # if the configuration file is invalid, then the context is left untouched
            Config_File_Validator.validate(data)

        for value in self._context.values():  # stop the current streams
            value.stop_stream()
        self._context = collections.OrderedDict()

        stream_dic = collections.OrderedDict()
        for entry in data:
            next_stream = validate_schema_and_create_stream(entry, created=False)
            next_name = IOTStreams.get_context_entry_name(next_stream.get_schema_name(), next_stream.get_stream_name())
            stream_dic[next_name] = next_stream
        self._context = stream_dic  # dictionary of schema_name + '.' + stream_name -> DataCellStream

        for value in self._context.values():  # start the new streams
            value.start_stream()

    def update_config_file(self):  # the write lock must be set before running this method!!!
        data = [value.get_data_dictionary(include_number_tuples=False) for value in self._context.values()]
        with open(get_configfile_location(), 'w') as outfile:  # re-write the whole config file
            json.dump(data, outfile)

    def add_new_stream(self, validating_schema):
        concat_name = IOTStreams.get_context_entry_name(validating_schema['schema'], validating_schema['stream'])
        self._locker.acquire_write()
        if concat_name in self._context:
            self._locker.release()
            raise IOTStreamsException('The stream ' + validating_schema['stream'] + ' in schema ' +
                                      validating_schema['schema'] + ' already exists!')
        try:
            new_stream = validate_schema_and_create_stream(validating_schema, created=True)
            self._context[concat_name] = new_stream
            self.update_config_file()  # update config file after adding a new stream
            new_stream.start_stream()
        except:
            self._locker.release()
            raise
        self._locker.release()

    def delete_existing_stream(self, validating_schema):
        concat_name = IOTStreams.get_context_entry_name(validating_schema['schema'], validating_schema['stream'])
        self._locker.acquire_write()
        if concat_name not in self._context:
            self._locker.release()
            raise IOTStreamsException('The stream ' + validating_schema['stream'] + ' in schema ' +
                                      validating_schema['schema'] + ' does not exist!')
        try:
            old_stream = self._context[concat_name]
            del self._context[concat_name]
            self.update_config_file()
            old_stream.stop_stream()
        except:
            self._locker.release()
            raise
        self._locker.release()

    def get_existing_stream(self, schema_name, stream_name):
        concat_name = IOTStreams.get_context_entry_name(schema_name, stream_name)
        self._locker.acquire_read()
        if concat_name not in self._context:
            self._locker.release()
            raise IOTStreamsException('The stream ' + stream_name + ' in schema ' + schema_name + ' does not exist!')
        res = self._context[concat_name]
        self._locker.release()
        return res

    def get_streams_data(self):
        self._locker.acquire_read()
        res = [value.get_data_dictionary(include_number_tuples=True) for value in self._context.values()]
        self._locker.release()
        return res
