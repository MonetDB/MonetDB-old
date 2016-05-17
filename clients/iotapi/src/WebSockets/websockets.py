import json
import sys

from Settings.iotlogger import add_log
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
from Streams.streamscontext import Streams_context, IOTStreams
from Utilities.readwritelock import RWLock
from jsonschema import Draft4Validator, FormatChecker

from jsonschemas import PUBSUB_STREAMS_SCHEMA

Client_Messages_Validator = Draft4Validator(PUBSUB_STREAMS_SCHEMA, format_checker=FormatChecker())
WebSocketServer = None
WebClients = []  # TODO this probably won't scale for many
WebClientsLock = RWLock()


def notify_stream_inserts_to_clients(schema_name, stream_name):
    concatenated_name = IOTStreams.get_context_entry_name(schema_name, stream_name)
    json_message = json.dumps({'notification': {'schema': schema_name, 'stream': stream_name}})
    WebClientsLock.acquire_read()
    for client in WebClients:
        client.send_notification_message(concatenated_name, json_message)
    WebClientsLock.release()


def desubscribe_removed_streams(concatenated_names):
    WebClientsLock.acquire_read()
    for name in concatenated_names:
        for client in WebClients:
            client.remove_subscribed_stream(name)
    WebClientsLock.release()
    for name in concatenated_names:
        add_log(20, ''.join(['Stream removed: ', name]))


class IOTAPI(WebSocket):
    def __init__(self):
        super(IOTAPI, self).__init__()
        self._subscriptions = {}  # dictionary of schema + '.' + stream -> IOTStream
        self._locker = RWLock()

    def handleMessage(self):
        if self.opcode != 0x1:  # TEXT frame
            self.sendMessage({"error": "Only TEXT frames allowed!"})
        try:
            input_schema = json.loads(self.data)
            Client_Messages_Validator.validate(input_schema)

            if input_schema['action'] in ("sub", "subscribe"):
                concatenated_name = IOTStreams.get_context_entry_name(input_schema['schema'], input_schema['stream'])
                self.subscribe(self, concatenated_name)
            elif input_schema['action'] in ("desub", "desubscribe"):
                concatenated_name = IOTStreams.get_context_entry_name(input_schema['schema'], input_schema['stream'])
                self.desubscribe(self, concatenated_name)
        except BaseException as ex:
            add_log(50, ex)
            self.sendMessage(json.dumps({"error": ex}))

    def handleConnected(self):
        WebClientsLock.acquire_write()
        WebClients.append(self)
        WebClientsLock.release()
        add_log(20, 'Client connected: ' + self.address[0])

    def handleClose(self):
        WebClientsLock.acquire_write()
        WebClients.remove(self)
        WebClientsLock.release()
        add_log(20, 'Client disconnected: ' + self.address[0])

    def subscribe(self, concatenated_name):
        try:
            stream = Streams_context.get_existing_stream(concatenated_name)
        except:
            raise
        self._locker.acquire_write()
        self._subscriptions[concatenated_name] = stream
        self._locker.release()
        self.sendMessage(json.dumps({"subscribed": "Subscribed to " + concatenated_name}))
        add_log(20, ''.join(['Client ', self.address[0], 'subscribed stream ', concatenated_name]))

    def desubscribe(self, concatenated_name):
        self._locker.acquire_write()
        if concatenated_name not in self._subscriptions:
            self._locker.release()
            self.sendMessage(json.dumps({"error": "Stream " + concatenated_name + " not present in subscriptions!"}))
        else:
            del self._subscriptions[concatenated_name]
            self._locker.release()
            self.sendMessage(json.dumps({"desubscribed": "Desubscribed to " + concatenated_name}))
            add_log(20, ''.join(['Client ', self.address[0], 'desubscribed stream ', concatenated_name]))

    def remove_subscribed_stream(self, concatenated_name):
        self._locker.acquire_write()
        if concatenated_name in self._subscriptions:
            del self._subscriptions[concatenated_name]
        self._locker.release()
        self.sendMessage(json.dumps({"removed": "Stream removed from context: " + concatenated_name}))

    def send_notification_message(self, concatenated_name, json_message):
        self._locker.acquire_read()
        if concatenated_name in self._subscriptions:
            self._locker.release()
            self.sendMessage(json_message)
            add_log(20, ''.join(['Stream notification sent to client ', self.address[0]]))
        else:
            self._locker.release()


def init_websockets(host, port):
    global WebSocketServer
    try:
        WebSocketServer = SimpleWebSocketServer(host, port, IOTAPI)
        WebSocketServer.serveforever()
    except (Exception, OSError) as ex:
        print >> sys.stdout, ex
        add_log(50, ex)
        sys.exit(1)


def terminate_websockets():
    WebSocketServer.close()
