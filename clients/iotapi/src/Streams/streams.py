import struct

import os
from Settings.filesystem import get_baskets_base_location
from WebSockets.websockets import notify_clients
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

BASKETS_COUNT_FILE = 'count'

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

    def on_created(self, event):  # whenever a basket directory is created, notify to subscribed clients
        if isinstance(event, 'DirCreatedEvent'):
            basket_string = os.path.basename(os.path.normpath(event.src_path))
            self._stream.baskets.append_basket(basket_string)
            notify_clients(self._stream.schema_name, self._stream.stream_name)


class DataCellStream(object):
    """Representation of a stream"""

    def __init__(self, schema_name, stream_name, columns):
        self.schema_name = schema_name  # name of the schema
        self.stream_name = stream_name  # name of the stream
        self._columns = columns  # dictionary of name -> data_types
        self._base_path = os.path.join(get_baskets_base_location(), schema_name, stream_name)
        self.baskets = {}  # dictionary of basket_number -> total_tuples
        for name in os.listdir(self._base_path):
            self.append_basket(name)
        self._observer = Observer()
        self._observer.schedule(StreamBasketsHandler(stream=self), self._base_path, recursive=False)
        self._observer.start()

    def append_basket(self, path):
        if represents_int(path):
            with open(os.path.join(self._base_path, path)) as f:
                count = struct.unpack('i', f.read(4))[0]
                self.baskets[int(path)] = count

    # TODO add delete basket!!!!

    def read_tuples(self, basket_number, limit, offset):
        results = {column: [] for column in self._columns.keys()}
        current_basket = int(basket_number)
        read_tuples = 0
        finished = False

        while True:
            if current_basket not in self.baskets:
                finished = True
                break
            offset -= self.baskets[current_basket]
            if offset < 0:
                break
            current_basket += 1

        if not finished:
            offset = abs(offset)

            while True:
                if current_basket not in self.baskets:
                    break
                next_path = os.path.join(self._base_path, str(current_basket))
                next_read_size = min(self.baskets[current_basket], limit)

                for key, column in self._columns.iteritems():
                    next_file_name = os.path.join(next_path, key)
                    open_string = 'r'
                    if not column.is_file_mode_binary():
                        open_string += 'u'
                    file_pointer = open(next_file_name, open_string)
                    results[key].append(column.read_next_batch(file_pointer, offset, next_read_size))

                offset = 0
                current_basket += 1
                read_tuples += next_read_size
                limit -= self.baskets[current_basket]
                if limit <= 0:
                    break

        # TODO check if this is viable, it could be 1000 tuples!!!!
        return {'total': read_tuples, 'tuples': zip(*results)}  # TODO not done this way!!!
