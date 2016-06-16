from collections import OrderedDict
from Utilities.readwritelock import RWLock
from WebSockets.websockets import unsubscribe_removed_streams


class IOTStreams:
    """Streams context"""

    @classmethod
    def get_context_entry_name(cls, schema_name, stream_name):
        return schema_name + '.' + stream_name

    def __init__(self):
        self._context = {}  # dictionary of schema_name + '.' + stream_name -> IOTStream
        self._locker = RWLock()

    def get_existing_streams(self):
        self._locker.acquire_read()
        res = list(self._context.keys())
        self._locker.release()
        return res

    def merge_context(self, retained_streams, new_streams):
        self._locker.acquire_write()
        removed_streams = [key for key in self._context.keys() if key not in retained_streams]
        for k in removed_streams:
            del self._context[k]
        self._context.update(new_streams)
        self._locker.release()
        unsubscribe_removed_streams(removed_streams)

    def get_existing_stream(self, concatenated_name):
        self._locker.acquire_read()
        if concatenated_name not in self._context:
            self._locker.release()
            raise Exception(concatenated_name + ' is inexistent in the context!')
        res = self._context[concatenated_name]
        self._locker.release()
        return res

    def get_streams_data(self):
        self._locker.acquire_read()
        res = (('streams_count', len(self._context)),
               ('streams_listing', [OrderedDict(value.get_data_dictionary()) for value in self._context.values()]))
        self._locker.release()
        return res

Streams_Context = IOTStreams()
