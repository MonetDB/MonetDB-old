from flask import Flask
from flask_restful import Api

from restresources import StreamInput, StreamsInfo, StreamsHandling  # , ServerHandler


def start_flask_iot_app(host, port):
    iot_app = Flask(__name__)
    iot_app.config['BUNDLE_ERRORS'] = True
    iot_api = Api(iot_app)

    iot_api.add_resource(StreamsInfo, '/streams')
    iot_api.add_resource(StreamInput, '/stream/<string:schema_name>/<string:stream_name>')
    iot_app.run(host=host, port=port, threaded=True)


def start_flask_admin_app(host, port):
    admin_app = Flask(__name__)
    admin_app.config['BUNDLE_ERRORS'] = True
    admin_api = Api(admin_app)

    admin_api.add_resource(StreamsInfo, '/streams')
    admin_api.add_resource(StreamsHandling, '/context')
    #  admin_api.add_resource(ServerHandler, '/server')
    admin_app.run(host=host, port=port, threaded=True)
