import os

from Settings.filesystem import get_baskets_base_location
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class StreamBasketsHandler(FileSystemEventHandler):
    def __init__(self, stream):
        super(StreamBasketsHandler, self).__init__()
        self._stream = stream

    def on_created(self, event):  # whenever a basket directory is created, notify
        if isinstance(event, 'DirCreatedEvent'):
            self._stream.read_new_tuples(event.src_path)


class DataCellStream(object):
    """Representation of a stream"""

    def __init__(self, schema_name, stream_name, columns):
        self._schema_name = schema_name  # name of the schema
        self._stream_name = stream_name  # name of the stream
        self._columns = columns  # dictionary of name -> data_types
        self._base_path = os.path.join(get_baskets_base_location(), schema_name, stream_name)
        self._observer = Observer()
        self._observer.schedule(StreamBasketsHandler(stream=self), self._base_path, recursive=False)
        self._observer.start()

    def read_new_tuples(self, path):
        for key, column in self._columns.iteritems():
            next_file_name = os.path.join(path, key)
            open_string = 'r'
            if not column.is_file_mode_binary():
                open_string += 'u'
            file_pointer = open(next_file_name, open_string)
