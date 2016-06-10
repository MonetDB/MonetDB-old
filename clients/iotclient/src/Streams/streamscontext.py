import collections

from Utilities.readwritelock import RWLock
from streamscreator import validate_schema_and_create_stream


class IOTStreams(object):
    @classmethod
    def get_context_entry_name(cls, schema_name, stream_name):
        return schema_name + '.' + stream_name

    def __init__(self):
        self._locker = RWLock()
        self._context = collections.OrderedDict()  # dictionary of schema_name + '.' + stream_name -> DataCellStream

    def add_new_stream(self, validating_schema):
        concat_name = IOTStreams.get_context_entry_name(validating_schema['schema'], validating_schema['stream'])
        self._locker.acquire_write()
        if concat_name in self._context:
            self._locker.release()
            raise Exception('The stream ' + validating_schema['stream'] + ' in schema ' + validating_schema['schema'] +
                            ' already exists!')
        try:
            new_stream = validate_schema_and_create_stream(validating_schema)
            new_stream.start_stream()
            self._context[concat_name] = new_stream
        except:
            self._locker.release()
            raise
        self._locker.release()

    def delete_existing_stream(self, validating_schema):
        concat_name = IOTStreams.get_context_entry_name(validating_schema['schema'], validating_schema['stream'])
        self._locker.acquire_write()
        if concat_name not in self._context:
            self._locker.release()
            raise Exception('The stream ' + validating_schema['stream'] + ' in schema ' + validating_schema['schema'] +
                            ' does not exist!')
        try:
            old_stream = self._context[concat_name]
            del self._context[concat_name]
            old_stream.stop_stream()
        except:
            self._locker.release()
            raise
        self._locker.release()

    def get_existing_streams(self):  # To use with next method!!
        self._locker.acquire_write()
        return list(self._context.keys())

    def merge_context(self, retained_streams, new_streams):  # To use with above method!!
        try:
            removed_streams = {key: value for (key, value) in self._context if key not in retained_streams}
            for key, value in removed_streams.iteritems():
                del self._context[key]
                value.stop_stream()
            self._context.update(new_streams)
            for value in new_streams.values():
                value.start_stream()
        except:
            self._locker.release()
            raise
        self._locker.release()

    def get_existing_stream(self, schema_name, stream_name):
        concat_name = IOTStreams.get_context_entry_name(schema_name, stream_name)
        self._locker.acquire_read()
        if concat_name not in self._context:
            self._locker.release()
            raise Exception('The stream ' + stream_name + ' in schema ' + schema_name + ' does not exist!')
        res = self._context[concat_name]
        self._locker.release()
        return res

    def get_streams_data(self):
        self._locker.acquire_read()
        res = {'streams_count': len(self._context),
               'streams_listing': [value.get_data_dictionary() for value in self._context.values()]}
        self._locker.release()
        return res

Streams_Context = IOTStreams()
