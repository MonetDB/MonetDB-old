import collections


class IOTStreams(object):
    """Stream's context"""

    @classmethod
    def get_context_entry_name(cls, schema_name, stream_name):
        return schema_name + '.' + stream_name

    def __init__(self):
        self._context = collections.OrderedDict()  # dictionary of schema_name + '.' + stream_name -> DataCellStream

    def is_stream_in_context(self, concatenated_name):
        return concatenated_name in self._context

    def add_stream(self, concatenated_name, stream):
        self._context[concatenated_name] = stream

    def delete_existing_stream(self, schema_name, stream_name):
        concat_name = IOTStreams.get_context_entry_name(schema_name, stream_name)
        del self._context[concat_name]


Streams_context = IOTStreams()
