import json
import sys

from Settings.iotlogger import add_log
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
from Streams.streamscontext import Streams_context, IOTStreams
from Utilities.readwritelock import RWLock
from jsonschema import Draft4Validator, FormatChecker

from jsonschemas import CLIENTS_INPUTS_SCHEMA, SUBSCRIBE_OPTS, UNSUBSCRIBE_OPTS, READ_OPTS

Client_Messages_Validator = Draft4Validator(CLIENTS_INPUTS_SCHEMA, format_checker=FormatChecker())
WebSocketServer = None
WebClients = []  # TODO this probably won't scale for many
WebClientsLock = RWLock()


def notify_stream_inserts_to_clients(schema_name, stream_name, count):
    concatenated_name = IOTStreams.get_context_entry_name(schema_name, stream_name)
    WebClientsLock.acquire_read()
    for client in WebClients:
        client.send_notification_message(concatenated_name, schema_name, stream_name, count)
    WebClientsLock.release()


def unsubscribe_removed_streams(concatenated_names):
    WebClientsLock.acquire_read()
    for name in concatenated_names:
        for client in WebClients:
            client.remove_subscribed_stream(name)
    WebClientsLock.release()
    for name in concatenated_names:
        add_log(20, ''.join(['Stream ', name, ' removed']))


class IOTAPI(WebSocket):
    """Client WebSocket"""

    def __init__(self, server, sock, address):
        super(IOTAPI, self).__init__(server, sock, address)
        self._subscriptions = {}  # dictionary of schema + '.' + stream -> IOTStream
        self._locker = RWLock()

    def sendMessage(self, message):  # overriden
        super(IOTAPI, self).sendMessage(json.dumps(message))  # send JSON Strings to clients

    def handleConnected(self):  # overriden
        WebClientsLock.acquire_write()
        WebClients.append(self)
        WebClientsLock.release()
        add_log(20, 'Client connected: ' + self.address[0])

    def handleClose(self):  # overriden
        WebClientsLock.acquire_write()
        WebClients.remove(self)
        WebClientsLock.release()
        add_log(20, 'Client disconnected: ' + self.address[0])

    def handleMessage(self):  # overriden
        if self.opcode != 0x1:  # TEXT frame
            self.sendMessage({"error": "Only TEXT frames allowed!"})
        try:
            input_schema = json.loads(self.data)
            Client_Messages_Validator.validate(input_schema)
            concatenated_name = IOTStreams.get_context_entry_name(input_schema['schema'], input_schema['stream'])

            if input_schema['action'] in SUBSCRIBE_OPTS:
                self.subscribe(concatenated_name)
            elif input_schema['action'] in UNSUBSCRIBE_OPTS:
                self.unsubscribe(concatenated_name)
            elif input_schema['action'] in READ_OPTS:
                self.read_stream_batch(concatenated_name, int(input_schema['basket']), int(input_schema['limit']),
                                       int(input_schema['offset']))
        except BaseException as ex:
            self.sendMessage({"error": ex})
            add_log(50, ex)

    def subscribe(self, concatenated_name):
        stream = Streams_context.get_existing_stream(concatenated_name)
        self._locker.acquire_write()
        self._subscriptions[concatenated_name] = stream
        self._locker.release()
        self.sendMessage({"subscribed": "Subscribed to " + concatenated_name})
        add_log(20, ''.join(['Client ', self.address[0], 'subscribed to stream ', concatenated_name]))

    def unsubscribe(self, concatenated_name):
        self._locker.acquire_write()
        if concatenated_name not in self._subscriptions:
            self._locker.release()
            self.sendMessage({"error": "Stream " + concatenated_name + " not present in subscriptions!"})
        else:
            del self._subscriptions[concatenated_name]
            self._locker.release()
            self.sendMessage({"unsubscribed": "Unsubscribed to " + concatenated_name})
            add_log(20, ''.join(['Client ', self.address[0], ' unsubscribed to stream ', concatenated_name]))

    def remove_subscribed_stream(self, concatenated_name):
        self._locker.acquire_write()
        if concatenated_name in self._subscriptions:
            del self._subscriptions[concatenated_name]
        self._locker.release()
        self.sendMessage({"removed": 'Stream ' + concatenated_name + ' removed from context'})

    def send_notification_message(self, concatenated_name, schema_name, stream_name, count):
        self._locker.acquire_read()
        if concatenated_name in self._subscriptions:
            self._locker.release()
            self.sendMessage({'notification': {'schema': schema_name, 'stream': stream_name, 'tuples': count}})
            add_log(20, ''.join(['Stream ', concatenated_name, ' notification sent to client ', self.address[0]]))
        else:
            self._locker.release()

    def read_stream_batch(self, concatenated_name, basket_number, limit, offset):
        try:
            stream = Streams_context.get_existing_stream(concatenated_name)
            self.sendMessage(stream.read_tuples(basket_number, limit, offset))
        except BaseException as ex:
            self.sendMessage({"error": ex})
            add_log(50, ex)


def init_websockets(host, port):
    global WebSocketServer
    try:
        WebSocketServer = SimpleWebSocketServer(host, port, IOTAPI)
        WebSocketServer.serveforever()
    except (BaseException, OSError) as ex:
        print >> sys.stdout, ex
        add_log(50, ex)
        sys.exit(1)


def terminate_websockets():
    WebSocketServer.close()
