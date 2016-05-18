import collections

from Utilities.readwritelock import RWLock
from WebSockets.websockets import desubscribe_removed_streams


class IOTStreams(object):
    """Streams context"""

    @classmethod
    def get_context_entry_name(cls, schema_name, stream_name):
        return schema_name + '.' + stream_name

    def __init__(self):
        self._context = collections.OrderedDict()  # dictionary of schema_name + '.' + stream_name -> DataCellStream
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
        desubscribe_removed_streams(removed_streams)

    def get_existing_stream(self, concatenated_name):
        self._locker.acquire_read()
        if concatenated_name not in self._context:
            self._locker.release()
            raise Exception(concatenated_name + ' is inexistent in the context!')
        res = self._context[concatenated_name]
        self._locker.release()
        return res

Streams_context = IOTStreams()
