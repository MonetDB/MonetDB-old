import time
from abc import ABCMeta, abstractmethod
from threading import Thread, Event


class StoppableThread(Thread):
    """Stoppable Thread"""

    def __init__(self):
        Thread.__init__(self)
        self.stop_event = Event()

    def stop(self):
        if self.isAlive():
            # set event to signal thread to terminate
            self.stop_event.set()
            # block calling thread until thread really has terminated
            self.join()


class IntervalTimer(StoppableThread):
    """Thread working with a timed interval basis"""

    def __init__(self, interval, worker_func):
        super(IntervalTimer, self).__init__()
        self._interval = interval  # in seconds
        self._worker_func = worker_func  # function/method to execute periodically

    def run(self):
        while not self.stop_event.is_set():
            self._worker_func()
            time.sleep(self._interval)


class StreamFlushingMethod(object):
    """Base class for flushing"""

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def get_dictionary_info(self):
        pass


class TimeBasedFlushing(StreamFlushingMethod):
    """Time based flushing"""

    def __init__(self, interval, time_unit):
        super(TimeBasedFlushing, self).__init__()
        self._interval = interval
        self._time_unit = time_unit
        self._local_thread = None

    def init_local_thread(self, stream):
        if self._time_unit == "s":
            interval = self._interval
        elif self._time_unit == "m":
            interval = self._interval * 60
        else:
            interval = self._interval * 3600
        self._local_thread = IntervalTimer(interval, stream.time_based_flush)
        self._local_thread.start()

    def stop_local_thread(self):
        self._local_thread.stop()

    def get_dictionary_info(self):
        return {'base': 'time', 'unit': self._time_unit, 'interval': self._interval}


class TupleBasedFlushing(StreamFlushingMethod):
    """Tuple based flushing"""

    def __init__(self, limit):
        super(TupleBasedFlushing, self).__init__()
        self.limit = limit

    def get_dictionary_info(self):
        return {'base': 'tuple', 'number': self.limit}
