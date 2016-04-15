import datetime
import json
import pytz

from flask import request
from flask_restful import Resource
from jsonschema import Draft4Validator, FormatChecker
from tzlocal import get_localzone
from src.Streams.jsonschemas import CREATE_STREAMS_SCHEMA, DELETE_STREAMS_SCHEMA
from src.Streams.streamscontext import IOTStreamsException, IOTStreams

Stream_context = IOTStreams()
local_tz = get_localzone()  # for the correction of dates we must add the timezone


class StreamInput(Resource):
    """RESTful API for stream's input"""

    def get(self, schema_name, stream_name):  # check a single stream data
        try:  # check if stream exists, if not return 404
            stream = Stream_context.get_existing_stream(schema_name, stream_name)
        except IOTStreamsException as ex:
            return ex.message, 404
        return stream.get_data_dictionary(), 200

    def post(self, schema_name, stream_name):  # add data to a stream
        current_stamp = datetime.datetime.now(pytz.utc).astimezone(local_tz).isoformat()

        try:  # check if stream exists, if not return 404
            stream = Stream_context.get_existing_stream(schema_name, stream_name)
        except IOTStreamsException as ex:
            return ex.message, 404

        try:  # validate and insert data, if not return 400
            stream.validate_and_insert(json.loads(request.data), current_stamp)
        except Exception as ex:
            return ex.message, 400
        return '', 201  # all ok, return 201


class StreamsInfo(Resource):
    """Collect all streams information"""

    def get(self):  # get all streams data
        return Stream_context.get_streams_data(), 200


class StreamsHandling(Resource):
    """Admin class for creating/deleting streams"""

    CREATE_STREAMS_VALIDATOR = Draft4Validator(CREATE_STREAMS_SCHEMA, format_checker=FormatChecker())
    DELETE_STREAMS_VALIDATOR = Draft4Validator(DELETE_STREAMS_SCHEMA, format_checker=FormatChecker())

    def __init__(self):
        super(StreamsHandling, self).__init__()

    def post(self):
        try:
            schema_to_validate = json.loads(request.data)
            StreamsHandling.CREATE_STREAMS_VALIDATOR.validate(schema_to_validate)
            Stream_context.add_new_stream(schema_to_validate)
        except Exception as ex:
            return ex.message, 400
        else:
            return '', 201

    def delete(self):
        try:
            schema_to_validate = json.loads(request.data)
            StreamsHandling.DELETE_STREAMS_VALIDATOR.validate(schema_to_validate)
        except Exception as ex:
            return ex.message, 400

        try:  # check if stream exists, if not return 404
            Stream_context.delete_existing_stream(schema_to_validate)
        except IOTStreamsException as ex:
            return ex.message, 404
        return '', 204
