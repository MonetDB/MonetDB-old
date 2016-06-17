UUID_REGEX = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
IPV4_REGEX = "^(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}(?:\/[0-2]\d|\/3[0-2])?$"
MAC_ADDRESS_REGEX = "^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$"
TIME_REGEX = "^([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])\.\d{3}([\+-]([01]\d|2[0-3]):[0-5]\d)?$"

TIMED_FLUSH_IDENTIFIER = "time"
TUPLE_FLUSH_IDENTIFIER = "tuple"
AUTO_FLUSH_IDENTIFIER = "auto"

UNBOUNDED_TEXT_TYPE = "clob"
UNBOUNDED_TEXT_INPUTS = [UNBOUNDED_TEXT_TYPE, "text", "string", "character large object"]
BOUNDED_TEXT_TYPES = ["char", "varchar"]
BOUNDED_TEXT_INPUTS = BOUNDED_TEXT_TYPES + ["character", "character varying"]

UUID_TYPE = "uuid"
MAC_TYPE = "mac"
URL_TYPE = "url"
INET_TYPE = "inet"
INET6_TYPE = "inet6"
REGEX_TYPE = "regex"
ENUM_TYPE = "enum"

BOOLEAN_TYPE = "boolean"
BOOLEAN_INPUTS = [BOOLEAN_TYPE, "bool"]

SMALL_INTEGERS_TYPES = ["tinyint", "smallint", "int", "bigint"]
SMALL_INTEGERS_INPUTS = SMALL_INTEGERS_TYPES + ["integer"]
HUGE_INTEGER_TYPE = "hugeint"

FLOATING_POINT_PRECISION_TYPES = ["real", "double"]
FLOATING_POINT_PRECISION_INPUTS = FLOATING_POINT_PRECISION_TYPES + ["float", "double precision"]

DECIMAL_TYPE = "decimal"
DECIMAL_INPUTS = [DECIMAL_TYPE, "dec", "numeric"]

DATE_TYPE = "date"

TIME_WITHOUT_TIMEZONE_TYPE = "time"
TIME_WITH_TIMEZONE_TYPE_INTERNAL = "timetz"
TIME_WITH_TIMEZONE_TYPE_EXTERNAL = "time with time zone"
TIME_INPUTS = [TIME_WITHOUT_TIMEZONE_TYPE, TIME_WITH_TIMEZONE_TYPE_EXTERNAL]

TIMESTAMP_WITHOUT_TIMEZONE_TYPE = "timestamp"
TIMESTAMP_WITH_TIMEZONE_TYPE_INTERNAL = "timestamptz"
TIMESTAMP_WITH_TIMEZONE_TYPE_EXTERNAL = "timestamp with time zone"
TIMESTAMP_INPUTS = [TIMESTAMP_WITHOUT_TIMEZONE_TYPE, TIMESTAMP_WITH_TIMEZONE_TYPE_EXTERNAL]

SECOND_INTERVAL_TYPE = "sec_interval"
MONTH_INTERVAL_TYPE = "month_interval"
INTERVAL_INPUTS = ["interval second", "interval minute", "interval hour", "interval day", "interval month",
                   "interval year", "interval year to month", "interval day to hour", "interval day to minute",
                   "interval day to second", "interval hour to minute", "interval hour to second",
                   "interval minute to second"]

CREATE_STREAMS_SCHEMA = None


def get_create_streams_schema():
    return CREATE_STREAMS_SCHEMA


def init_create_streams_schema(add_hugeint=True):
    global CREATE_STREAMS_SCHEMA

    if add_hugeint:
        integers_and_intervals = SMALL_INTEGERS_INPUTS + [HUGE_INTEGER_TYPE] + INTERVAL_INPUTS
        max_precision = 38
    else:
        integers_and_intervals = SMALL_INTEGERS_INPUTS + INTERVAL_INPUTS
        max_precision = 18

    CREATE_STREAMS_SCHEMA = {
        "title": "JSON schema to create a stream",
        "description": "Validate the inserted properties",
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "schema": {"type": "string"},
            "stream": {"type": "string"},
            "has_hostname": {"type": "boolean", "default": False},
            "flushing": {
                "type": "object",
                "oneOf": [{
                    "properties": {
                        "base": {"type": "string", "enum": [TIMED_FLUSH_IDENTIFIER]},
                        "interval": {"type": "integer", "minimum": 1},
                        "unit": {"type": "string", "enum": ["s", "m", "h"]}
                    },
                    "required": ["base", "interval", "unit"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "base": {"type": "string", "enum": [TUPLE_FLUSH_IDENTIFIER]},
                        "interval": {"type": "integer", "minimum": 1}
                    },
                    "required": ["base", "number"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "base": {"type": "string", "enum": [AUTO_FLUSH_IDENTIFIER]},
                    },
                    "required": ["base"],
                    "additionalProperties": False
                }]
            },
            "columns": {
                "type": "array",
                "minItems": 1,
                "additionalItems": False,
                "items": {
                    "type": "object",
                    "anyOf": [{
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": UNBOUNDED_TEXT_INPUTS},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string"}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": [UUID_TYPE]},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string", "pattern": UUID_REGEX}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": [MAC_TYPE]},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string", "pattern": MAC_ADDRESS_REGEX}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": [URL_TYPE]},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string", "format": "uri"}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": [INET_TYPE]},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string", "pattern": IPV4_REGEX}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": [INET6_TYPE]},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string", "format": "ipv6"}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": [REGEX_TYPE]},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string"},
                            "regex": {"type": "string"}
                        },
                        "required": ["name", "type", "regex"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": BOUNDED_TEXT_INPUTS},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1}
                        },
                        "required": ["name", "type", "limit"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": [ENUM_TYPE]},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string"},
                            "values": {"type": "array", "minItems": 1, "uniqueItems": True,
                                       "items": {"type": "string"}}
                        },
                        "required": ["name", "type", "values"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": BOOLEAN_INPUTS},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "boolean"}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": integers_and_intervals},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "integer"},
                            "minimum": {"type": "integer"},
                            "maximum": {"type": "integer"}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": FLOATING_POINT_PRECISION_INPUTS},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "number"},
                            "minimum": {"type": "number"},
                            "maximum": {"type": "number"}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": DECIMAL_INPUTS},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "number"},
                            "precision": {"type": "integer", "minimum": 1, "maximum": max_precision, "default": 18},
                            "scale": {"type": "integer", "minimum": 0, "default": 3},
                            "minimum": {"type": "number"},
                            "maximum": {"type": "number"}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": [DATE_TYPE]},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string", "format": "date"},
                            "minimum": {"type": "string", "format": "date"},
                            "maximum": {"type": "string", "format": "date"},
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": TIME_INPUTS},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string", "pattern": TIME_REGEX},
                            "minimum": {"type": "string", "pattern": TIME_REGEX},
                            "maximum": {"type": "string", "pattern": TIME_REGEX}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }, {
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": TIMESTAMP_INPUTS},
                            "nullable": {"type": "boolean", "default": True},
                            "default": {"type": "string", "format": "date-time"},
                            "minimum": {"type": "string", "format": "date-time"},
                            "maximum": {"type": "string", "format": "date-time"}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }]
                }
            }
        },
        "required": ["schema", "stream", "flushing", "columns"],
        "additionalProperties": False
    }

DELETE_STREAMS_SCHEMA = {
    "title": "JSON schema to delete a stream",
    "description": "Validate the inserted properties",
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "schema": {"type": "string"},
        "stream": {"type": "string"}
    },
    "required": ["schema", "stream"],
    "additionalProperties": False
}
