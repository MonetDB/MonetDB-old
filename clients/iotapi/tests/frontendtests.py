import json
import os
import requests

from distutils.dir_util import copy_tree
from threading import Thread
from time import sleep
from tornado import ioloop
from unittest import TestCase
from websocketclient import WebSocketClient

__all__ = ['NullablesTest']

WEB_SOCKETS_THREADS_TIMEOUT = 15


class BaseFrontEndTest(TestCase, WebSocketClient):

    def __init__(self, iot_client_path, iot_api_path, stream_name):
        TestCase.__init__(self)
        WebSocketClient.__init__(self)
        self._web_server_baskets_location = os.path.join(iot_client_path, 'baskets')
        self._web_api_baskets_location = os.path.join(iot_api_path, 'baskets')
        self.schema_name = "tests"
        self.stream_name = stream_name
        self.ws_state = 1

    def export_inserts(self, basket):
        input_dir = os.path.join(self._web_server_baskets_location, self.schema_name, self.stream_name, basket)
        output_dir = os.path.join(self._web_api_baskets_location, self.schema_name, self.stream_name, basket)
        copy_tree(input_dir, output_dir)


class NullablesTest(BaseFrontEndTest):

    def __init__(self, iot_client_path, iot_api_path):
        super(NullablesTest, self).__init__(iot_client_path, iot_api_path, stream_name="nulls")
        self._error = ""

    def web_socket_cycle(self):
        self.connect("ws://127.0.0.1:8002/")
        try:
            ioloop.IOLoop.instance().start()
        except:
            pass

    def _on_connection_success(self):
        self.send(''.join(['{"request": "sub", "schema": "', self.schema_name, '", "stream": "', self.stream_name,
                           '"}']))

    def _on_connection_error(self, exception):
        self.set_error(exception)

    def set_error(self, msg):
        self._error = msg
        self.close()
        ioloop.IOLoop.instance().stop()

    def _on_message(self, message):
        resp = json.loads(message)
        if resp['response'] == 'error':
            self.set_error("Received error message!")
        elif self.ws_state == 1:
            if resp != {"response": "subscribed", "schema": self.schema_name, "stream": self.stream_name}:
                self.set_error("Wrong subscribed response!")
            else:
                self.ws_state = 2
        elif self.ws_state == 2:
            correct_dic = {"response": "notification", "schema": self.schema_name, "stream": self.stream_name,
                           "basket": 1, "count": 3}
            if resp != correct_dic:
                self.set_error("Wrong notification response!")
            else:
                self.send(''.join(['{"request": "read", "schema": "', self.schema_name, '","stream": "',
                                   self.stream_name, '", "basket": 1, "offset": 0, "limit": 3}']))
                self.ws_state = 3
        elif self.ws_state == 3:
            if 'implicit_timestamp' not in resp['tuples'][0]:
                self.set_error('Timestamp not in result stream')
            else:
                del resp['tuples'][0]['implicit_timestamp']
                del resp['tuples'][1]['implicit_timestamp']
                del resp['tuples'][2]['implicit_timestamp']
                res = {"vala": None, "valb": None, "valc": None, "vald": None, "vale": None, "valf": None, "valg": None,
                       "valh": None, "vali": None, "valj": None, "valk": None, "vall": None, "valm": None, "valn": None,
                       "valo": None, "valp": None, "valq": None, "valr": None, "vals": None, "valt": None, "valu": None,
                       "valv": None, "valw": None, "valx": None, "valy": None, "valz": None}
                tuples_response = {"response": "read", "schema": self.schema_name, "stream": self.stream_name,
                                   "count": 3, "tuples": [res, res, res]}
                if resp != tuples_response:
                    self.set_error("Wrong notification response!")
                else:
                    self.ws_state = 4
        elif self.ws_state == 4:
            if resp != {"response": "removed", "schema": self.schema_name, "stream": self.stream_name}:
                self.set_error("Wrong removed response!")
            else:
                self.close()
                ioloop.IOLoop.instance().stop()

    def runTest(self):
        json_str = {"schema": self.schema_name, "stream": self.stream_name, "has_hostname": False,
                    "flushing": {"base": "auto"}, "columns": [
                                {"name": "vala", "type": "string", "nullable": True},
                                {"name": "valb", "type": "uuid", "nullable": True},
                                {"name": "valc", "type": "mac", "nullable": True},
                                {"name": "vald", "type": "url", "nullable": True},
                                {"name": "vale", "type": "inet", "nullable": True},
                                {"name": "valf", "type": "inet6", "nullable": True},
                                {"name": "valg", "type": "regex", "regex": "a", "nullable": True},
                                {"name": "valh", "type": "varchar", "limit": 16, "nullable": True},
                                {"name": "vali", "type": "enum", "values": ["a", "b", "c"], "nullable": True},
                                {"name": "valj", "type": "boolean", "nullable": True},
                                {"name": "valk", "type": "tinyint", "nullable": True},
                                {"name": "vall", "type": "smallint", "nullable": True},
                                {"name": "valm", "type": "int", "nullable": True},
                                {"name": "valn", "type": "bigint", "nullable": True},
                                {"name": "valo", "type": "hugeint", "nullable": True},
                                {"name": "valp", "type": "real", "nullable": True},
                                {"name": "valq", "type": "float", "nullable": True},
                                {"name": "valr", "type": "decimal", "precision": 12, "scale": 10, "nullable": True},
                                {"name": "vals", "type": "decimal", "precision": 28, "scale": 20, "nullable": True},
                                {"name": "valt", "type": "date", "nullable": True},
                                {"name": "valu", "type": "time", "nullable": True},
                                {"name": "valv", "type": "time with time zone", "nullable": True},
                                {"name": "valw", "type": "timestamp", "nullable": True},
                                {"name": "valx", "type": "timestamp with time zone", "nullable": True},
                                {"name": "valy", "type": "interval month", "nullable": True},
                                {"name": "valz", "type": "interval second", "nullable": True}]}

        resp = requests.post("http://127.0.0.1:8001/context", json=json_str)

        self.assertEqual(resp.status_code, 201, msg=resp.text)

        sleep(4)  # we need to sleep to check that the next poll happens

        thread = Thread(target=self.web_socket_cycle)
        thread.start()

        resp = requests.post("http://127.0.0.1:8000/stream/%s/%s" % (self.schema_name, self.stream_name),
                             json=[{}, {}, {}])

        self.assertEqual(resp.status_code, 201, msg=resp.text)

        self.export_inserts("1")

        sleep(4)

        resp = requests.delete("http://127.0.0.1:8001/context",
                               json={"schema": self.schema_name, "stream": self.stream_name})

        self.assertEqual(resp.status_code, 204, msg=resp.text)

        thread.join(timeout=WEB_SOCKETS_THREADS_TIMEOUT)
        if thread.isAlive():
            ioloop.IOLoop.instance().stop()
            self.fail(msg='The websockets tests timed out!')
        if self._error != "":
            self.fail(msg=self._error)
