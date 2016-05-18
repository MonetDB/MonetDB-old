UUID_REGEX = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
IPV4_REGEX = "^(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}(?:\/[0-2]\d|\/3[0-2])?$"
MAC_ADDRESS_REGEX = "^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$"
TIME_REGEX = "^([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])\.\d{3}$"

TIMED_FLUSH_IDENTIFIER = "time"
TUPLE_FLUSH_IDENTIFIER = "tuple"

UNBOUNDED_TEXT_TYPES = ["text", "string", "clob", "character large object"]
BOUNDED_TEXT_TYPES = ["char", "character", "varchar", "character varying"]
UUID_TYPE = ["uuid"]
MAC_TYPE = ["mac"]
URL_TYPE = ["url"]
INET_TYPE = ["inet"]
INET6_TYPE = ["inet6"]
REGEX_TYPE = ["regex"]
ENUM_TYPE = ["enum"]
BOOLEAN_TYPE = ["bool", "boolean"]
SMALL_INTEGERS = ["tinyint", "smallint", "int", "integer", "bigint"]
HUGE_INTEGER = ["hugeint"]
INTEGERS = SMALL_INTEGERS + HUGE_INTEGER
FLOATING_POINT_PRECISION_TYPES = ["real", "float", "double", "double precision"]
DECIMAL_TYPES = ["dec", "decimal", "numeric"]
DATE_TYPE = ["date"]
TIME_TYPE = ["time"]
TIMESTAMP_TYPE = ["timestamp"]

STREAM_SCHEMA = {
    "type": "object",
    "properties": {
        "schema": {"type": "string"},
        "stream": {"type": "string"},
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
                    "number": {"type": "integer", "minimum": 1}
                },
                "required": ["base", "number"],
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
                        "type": {"type": "string", "enum": UNBOUNDED_TEXT_TYPES},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "string"}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": UUID_TYPE},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "string", "pattern": UUID_REGEX}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": MAC_TYPE},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "string", "pattern": MAC_ADDRESS_REGEX}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": URL_TYPE},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "string", "format": "uri"}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": INET_TYPE},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "string", "pattern": IPV4_REGEX}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": INET6_TYPE},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "string", "format": "ipv6"}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": REGEX_TYPE},
                        "nullable": {"type": "boolean"},
                        "regex": {"type": "string"},
                        "default": {"type": "string"}
                    },
                    "required": ["name", "type", "regex"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": BOUNDED_TEXT_TYPES},
                        "nullable": {"type": "boolean"},
                        "limit": {"type": "integer", "minimum": 1},
                        "default": {"type": "string"}
                    },
                    "required": ["name", "type", "limit"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ENUM_TYPE},
                        "nullable": {"type": "boolean"},
                        "values": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string"}},
                        "default": {"type": "string"}
                    },
                    "required": ["name", "type", "values"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": BOOLEAN_TYPE},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "boolean"}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": INTEGERS},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "integer"},
                        "minimum": {"type": "integer"},
                        "maximum": {"type": "integer"}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": FLOATING_POINT_PRECISION_TYPES},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "number"},
                        "minimum": {"type": "number"},
                        "maximum": {"type": "number"}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": DECIMAL_TYPES},
                        "nullable": {"type": "boolean"},
                        "precision": {"type": "integer", "minimum": 1, "maximum": 38},
                        "scale": {"type": "integer", "minimum": 0},
                        "default": {"type": "number"},
                        "minimum": {"type": "number"},
                        "maximum": {"type": "number"}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": DATE_TYPE},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "string", "format": "date"},
                        "minimum": {"type": "string", "format": "date"},
                        "maximum": {"type": "string", "format": "date"},
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": TIME_TYPE},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "string", "pattern": TIME_REGEX},
                        "minimum": {"type": "string", "pattern": TIME_REGEX},
                        "maximum": {"type": "string", "pattern": TIME_REGEX},
                        "timezone": {"type": "boolean"}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }, {
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": TIMESTAMP_TYPE},
                        "nullable": {"type": "boolean"},
                        "default": {"type": "string", "format": "date-time"},
                        "minimum": {"type": "string", "format": "date-time"},
                        "maximum": {"type": "string", "format": "date-time"},
                        "timezone": {"type": "boolean"}
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

CONFIG_FILE_SCHEMA = {
    "title": "JSON schema to load configuration file",
    "description": "Validate the inserted properties",
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "array",
    "items": STREAM_SCHEMA
}

CREATE_STREAMS_SCHEMA = {
    "title": "JSON schema to create a stream",
    "description": "Validate the inserted properties",
    "$schema": "http://json-schema.org/draft-04/schema#",
}

CREATE_STREAMS_SCHEMA.update(STREAM_SCHEMA)

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
