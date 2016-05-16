import os
from Settings.filesystem import get_baskets_base_location
from WebSockets.websockets import notify_clients
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.Streams.streamscontext import IOTStreams


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class StreamBasketsHandler(FileSystemEventHandler):
    """File watcher of a stream"""

    def __init__(self, stream):
        super(StreamBasketsHandler, self).__init__()
        self._stream = stream

    def on_created(self, event):  # whenever a basket directory is created, notify
        if isinstance(event, 'DirCreatedEvent'):
            basket_number = int(os.path.basename(os.path.normpath(event.src_path)))
            self._stream.baskets.append(basket_number)
            notify_clients(self._stream.schema_name, self._stream.stream_name)


class DataCellStream(object):
    """Representation of a stream"""

    def __init__(self, schema_name, stream_name, columns):
        self.schema_name = schema_name  # name of the schema
        self.stream_name = stream_name  # name of the stream
        self._columns = columns  # dictionary of name -> data_types
        self._base_path = os.path.join(get_baskets_base_location(), schema_name, stream_name)
        self.baskets = [int(name) for name in os.listdir(self._base_path) if represents_int(name)]
        self._observer = Observer()
        self._observer.schedule(StreamBasketsHandler(stream=self), self._base_path, recursive=False)
        self._observer.start()

    def read_tuples(self, basket_number, limit, offset):
        if basket_number not in self.baskets:
            concatenated_name = IOTStreams.get_context_entry_name(self.schema_name, self.stream_name)
            raise Exception('Stream ' + concatenated_name + ' does not contain basket ' + str(basket_number))

        results = []
        base_read_path = os.path.join(self._base_path, str(basket_number))
        for key, column in self._columns.iteritems():
            next_file_name = os.path.join(base_read_path, key)
            open_string = 'r'
            if not column.is_file_mode_binary():
                open_string += 'u'
            file_pointer = open(next_file_name, open_string)
            results.append(column.read_next_batch(file_pointer, 100))

        packed_results = zip(*results)  # TODO check if this is viable, it could be 1000 tuples!!!!
        # notify_clients(self._schema_name, self._stream_name, packed_results)
