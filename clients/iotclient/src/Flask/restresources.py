import datetime
import json
import pytz
import sys

from flask import request
from flask_restful import Resource
from jsonschema import Draft4Validator, FormatChecker
from tzlocal import get_localzone
from Streams.jsonschemas import CREATE_STREAMS_SCHEMA, DELETE_STREAMS_SCHEMA
from Streams.streamscontext import IOTStreamsContext
from Settings.iotlogger import add_log

Streams_Context = IOTStreamsContext()
Create_Streams_Validator = Draft4Validator(CREATE_STREAMS_SCHEMA, format_checker=FormatChecker())
Delete_Streams_Validator = Draft4Validator(DELETE_STREAMS_SCHEMA, format_checker=FormatChecker())
Local_Timezone = get_localzone()  # for the correction of dates we must add the system's timezone


def init_rest_resources():
    global Streams_Context

    try:
        Streams_Context = IOTStreamsContext()
    except BaseException as ex:
        print ex
        add_log(50, ex)
        sys.exit(1)


class StreamInput(Resource):
    """RESTful API for stream's input"""

    def __init__(self):
        super(StreamInput, self).__init__()

    def get(self, schema_name, stream_name):  # check a single stream data
        try:  # check if stream exists, if not return 404
            stream = Streams_Context.get_existing_stream(schema_name, stream_name)
        except BaseException as ex:
            add_log(50, ex)
            return ex, 404
        return stream.get_data_dictionary(include_number_tuples=True), 200

    def post(self, schema_name, stream_name):  # add data to a stream
        current_stamp = datetime.datetime.now(pytz.utc).astimezone(Local_Timezone).isoformat()

        try:  # check if stream exists, if not return 404
            stream = Streams_Context.get_existing_stream(schema_name, stream_name)
        except BaseException as ex:
            add_log(50, ex)
            return ex, 404

        try:  # validate and insert data, if not return 400
            stream.validate_and_insert(json.loads(request.data), current_stamp)
        except BaseException as ex:
            add_log(50, ex)
            return ex, 400
        return 'The insertions were made with success!', 201


class StreamsInfo(Resource):
    """Collect all streams information"""

    def __init__(self):
        super(StreamsInfo, self).__init__()

    def get(self):  # get all streams data
        return Streams_Context.get_streams_data(), 200


class StreamsHandling(Resource):
    """Admin class for creating/deleting streams"""

    def __init__(self):
        super(StreamsHandling, self).__init__()

    def post(self):
        try:
            schema_to_validate = json.loads(request.data)
            Create_Streams_Validator.validate(schema_to_validate)
            Streams_Context.add_new_stream(schema_to_validate)
        except BaseException as ex:
            add_log(50, ex)
            return ex, 400
        else:
            add_log(20, ''.join['The stream ', schema_to_validate['schema'], '.', schema_to_validate['stream'],
                                ' was created'])
            return 'The stream was created with success!', 201

    def delete(self):
        try:
            schema_to_validate = json.loads(request.data)
            Delete_Streams_Validator.validate(schema_to_validate)
        except BaseException as ex:
            add_log(50, ex)
            return ex, 400

        try:  # check if stream exists, if not return 404
            Streams_Context.delete_existing_stream(schema_to_validate)
        except BaseException as ex:
            add_log(50, ex)
            return ex, 404
        add_log(20, ''.join['The stream ', schema_to_validate['schema'], '.', schema_to_validate['stream'],
                            ' was deleted'])
        return 'The stream was deleted with success!', 204
