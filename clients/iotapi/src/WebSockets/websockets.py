import sys

from Settings.iotlogger import add_log
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket

WebSocketServer = None
clients = []  # this probably won't scale


class IOTAPI(WebSocket):
    def handleMessage(self):
        for client in clients:
            client.sendMessage(self.data)

    def handleConnected(self):
        clients.append(self)
        add_log(20, 'Client connected: ' + self.address[0])

    def handleClose(self):
        clients.remove(self)
        add_log(20, 'Client disconnected: ' + self.address[0])


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
