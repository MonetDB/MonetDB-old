import os
import struct

from datatypes import LITTLE_ENDIAN_ALIGNMENT
from Settings.filesystem import get_baskets_base_location
from Utilities.readwritelock import RWLock
from WebSockets.websockets import notify_stream_inserts_to_clients
from watchdog.events import FileSystemEventHandler, DirCreatedEvent, DirDeletedEvent
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
        if isinstance(event, DirCreatedEvent):
            basket_string = os.path.basename(os.path.normpath(event.src_path))
            count = self._stream.append_basket(basket_string)
            notify_stream_inserts_to_clients(self._stream.get_schema_name(), self._stream.get_stream_name(), count)

    def on_deleted(self, event):
        if isinstance(event, DirDeletedEvent):
            basket_string = os.path.basename(os.path.normpath(event.src_path))
            self._stream.delete_basket(basket_string)


class IOTStream(object):
    """Representation of a stream"""

    def __init__(self, schema_name, stream_name, columns):
        self._schema_name = schema_name  # name of the schema
        self._stream_name = stream_name  # name of the stream
        self._columns = columns  # dictionary of name -> data_types
        self._base_path = os.path.join(get_baskets_base_location(), schema_name, stream_name)
        self._lock = RWLock()
        self._baskets = {}  # dictionary of basket_number -> total_tuples
        for name in os.listdir(self._base_path):
            self.append_basket(name)
        self._observer = Observer()
        self._observer.schedule(StreamBasketsHandler(stream=self), self._base_path, recursive=False)
        self._observer.start()

    def get_schema_name(self):
        return self._schema_name

    def get_stream_name(self):
        return self._stream_name

    def append_basket(self, path):
        if represents_int(path):
            with open(os.path.join(self._base_path, path, BASKETS_COUNT_FILE)) as f:
                count = struct.unpack(LITTLE_ENDIAN_ALIGNMENT + 'i', f.read(4))[0]
                self._lock.acquire_write()
                self._baskets[int(path)] = count
                self._lock.release()
                return count
        return 0

    def delete_basket(self, path):
        if represents_int(path):
            number = int(path)
            self._lock.acquire_write()
            if number in self._baskets:
                del self._baskets[number]
            self._lock.release()

    def get_next_basket_number_tuple(self, basket_number):
        self._lock.acquire_read()
        if basket_number in self._baskets:
            self._lock.release()
            return basket_number, self._baskets[basket_number]
        else:
            filtered = filter(lambda x: x > basket_number, self._baskets.keys())
            if len(filtered) > 0:
                min_basket_number = min(filtered)
                min_basket_tuples = self._baskets[min_basket_number]
                self._lock.release()
                return min_basket_number, min_basket_tuples
            else:
                self._lock.release()
                return None, None

    def read_tuples(self, basket_number, limit, offset):
        results = {column: [] for column in self._columns.keys()}
        current_basket_number = int(basket_number)
        read_tuples = 0
        skipped_tuples = 0
        finished = False

        while True:
            current_basket_number, current_tuple_number = self.get_next_basket_number_tuple(current_basket_number)
            if current_basket_number is None:
                finished = True
                break
            if skipped_tuples + current_tuple_number > offset:
                offset = offset - skipped_tuples
                break
            skipped_tuples += current_tuple_number
            current_basket_number += 1

        if not finished:
            while True:
                current_basket_number, current_tuple_number = self.get_next_basket_number_tuple(current_basket_number)
                if current_basket_number is None or read_tuples >= limit:
                    break

                next_path = os.path.join(self._base_path, str(current_basket_number))
                next_read_size = min(limit - read_tuples, current_tuple_number) - offset

                for key, column in self._columns.iteritems():
                    next_file_name = os.path.join(next_path, key)
                    results[key] += column.read_next_tuples(next_file_name, offset, next_read_size)

                read_tuples += next_read_size
                offset = 0
                current_basket_number += 1

        keys = results.keys()  # TODO check if this is viable for many tuples!!
        tuples = [dict(zip(keys, values)) for values in zip(*(results[k] for k in keys))]
        return {'total': read_tuples, 'tuples': tuples}
