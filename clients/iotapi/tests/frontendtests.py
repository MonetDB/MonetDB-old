import json
import os
import requests
import websocket

from distutils.dir_util import copy_tree
from threading import Thread
from time import sleep
from unittest import TestCase

__all__ = ['NullablesTest']

WEB_SOCKETS_THREADS_TIMEOUT = 15


class BaseFrontEndTest(TestCase):

    def __init__(self, **kwargs):
        super(BaseFrontEndTest, self).__init__()
        self._web_server_baskets_location = os.path.join(kwargs['iot_client_path'], 'baskets')
        self._web_api_baskets_location = os.path.join(kwargs['iot_api_path'], 'baskets')
        self.schema = "tests"

    def export_inserts(self, schema, stream, basket):
        input_dir = os.path.join(self._web_server_baskets_location, schema, stream, basket)
        output_dir = os.path.join(self._web_api_baskets_location, schema, stream, basket)
        copy_tree(input_dir, output_dir)


class TestWebSocket(websocket.WebSocketApp):

    def __init__(self, test, url, header=[], on_open=None, on_message=None, on_error=None, on_close=None, on_ping=None,
                 on_pong=None, on_cont_message=None, keep_running=True, get_mask_key=None, cookie=None,
                 subprotocols=None, on_data=None):
        super(TestWebSocket, self).__init__(url, header, on_open, on_message, on_error, on_close, on_ping,
                                            on_pong, on_cont_message, keep_running, get_mask_key, cookie,
                                            subprotocols, on_data)
        self.test = test
        self.ws_state = 1


def on_open(ws):
    ws.send(json.dumps({"request": "sub", "schema": ws.test.schema, "stream":  ws.test.stream}))


def on_message(ws, message):
    resp = json.loads(message)
    if resp['response'] == 'error':
        ws.test.fail(msg=resp['message'])
    elif ws.ws_state == 1:
        ws.test.assertDictEqual({"response": "subscribed", "schema": ws.test.schema, "stream": ws.test.stream}, resp)
        ws.ws_state = 2
    elif ws.ws_state == 2:
        ws.test.assertDictEqual({"response": "notification", "schema": ws.test.schema, "stream": ws.test.stream,
                                 "basket": 1, "count": 3}, resp)
        ws.send(json.dumps({"request": "read", "schema": ws.test.schema, "stream": ws.test.stream,
                            "basket": 1, "offset": 0, "limit": 3}))
        ws.ws_state = 3
    elif ws.ws_state == 3:
        ws.test.assertIn('implicit_timestamp', resp['tuples'][0], msg='Timestamp not in stream')
        del resp['tuples'][0]['implicit_timestamp']
        del resp['tuples'][1]['implicit_timestamp']
        del resp['tuples'][2]['implicit_timestamp']
        res_dic = {'vala': None, "valb": None, "valc": None, "vald": None, "vale": None, "valf": None, "valg": None,
                   "valh": None, "vali": None, "valj": None, "valk": None, "vall": None, "valm": None, "valn": None,
                   "valo": None, "valp": None, "valq": None, "valr": None, "vals": None, "valt": None, "valu": None,
                   "valv": None, "valw": None, "valx": None, "valy": None, "valz": None}
        tuples_response = {"response": "read", "schema": ws.test.schema, "stream": ws.test.stream,
                           "count": 3, "tuples": [res_dic, res_dic, res_dic]}
        ws.test.assertDictEqual(tuples_response, resp)
        ws.close()
    elif ws.ws_state == 4:
        ws.test.assertDictEqual({"response": "removed", "schema": ws.test.schema, "stream": ws.test.stream}, resp)
        ws.close()


def on_error(ws, error):
    ws.close()
    ws.test.fail(msg=error)


def web_socket(test):
    ws = TestWebSocket(test=test, url="ws://127.0.0.1:8002/", on_message=on_message, on_open=on_open, on_error=on_error)
    test.ws = ws
    ws.run_forever()


class NullablesTest(BaseFrontEndTest):

    def __init__(self, **kwargs):
        super(NullablesTest, self).__init__(**kwargs)
        self.stream = "nulls"
        self.ws = None

    def runTest(self):
        json_str = {"schema": self.schema, "stream": self.stream, "has_hostname": False, "flushing": {"base": "auto"},
                    "columns": [{"name": "vala", "type": "string", "nullable": True},
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

        sleep(2)  # we need to sleep to check that the next poll happens

        thread = Thread(target=web_socket, args=(self, ))
        thread.start()

        resp = requests.post("http://127.0.0.1:8000/stream/%s/%s" % (self.schema, self.stream), json=[{}, {}, {}])

        self.assertEqual(resp.status_code, 201, msg=resp.text)

        self.export_inserts("tests", "nulls", "1")

        sleep(2)

        resp = requests.delete("http://127.0.0.1:8001/context", json={"schema": self.schema, "stream": self.stream})

        self.assertEqual(resp.status_code, 204, msg=resp.text)

        thread.join(timeout=WEB_SOCKETS_THREADS_TIMEOUT)
        if thread.isAlive():
            self.ws.close()
            self.fail(msg='The websockets tests timed out!')
