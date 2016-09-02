from threading import Thread, Event
from time import sleep


class StoppableThread(Thread):
    """Stoppable Thread"""

    def __init__(self, *args, **kwargs):
        Thread.__init__(self, *args, **kwargs)
        self.stop_event = Event()

    def stop(self):
        if self.isAlive():
            # set event to signal thread to terminate
            self.stop_event.set()
            # block calling thread until thread really has terminated
            self.join()


class PeriodicalThread(StoppableThread):
    """Thread working with a timed interval basis"""

    def __init__(self, interval, worker_func, argument=None, *args, **kwargs):
        super(PeriodicalThread, self).__init__(*args, **kwargs)
        self._interval = interval  # in seconds
        self._worker_func = worker_func  # function/method to execute periodically
        self._argument = argument

    def run(self):
        while not self.stop_event.is_set():
            if self._argument is not None:
                self._worker_func(self._argument)
            else:
                self._worker_func()
            sleep(self._interval)
