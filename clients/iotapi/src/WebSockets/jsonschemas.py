SUBSCRIBE_OPTS = ["subscribe"]
UNSUBSCRIBE_OPTS = ["unsubscribe"]
INFO_OPTS = ["info"]
READ_OPTS = ["read"]

CLIENTS_INPUTS_SCHEMA = {
    "title": "JSON schema fro web api",
    "description": "Validate clients inputs",
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "anyOf": [{
        "properties": {
            "request": {"type": "string", "enum": SUBSCRIBE_OPTS + UNSUBSCRIBE_OPTS + INFO_OPTS},
            "schema": {"type": "string"},
            "stream": {"type": "string"}
        },
        "required": ["request", "schema", "stream"],
        "additionalProperties": False
    }, {
        "properties": {
            "request": {"type": "string", "enum": READ_OPTS},
            "schema": {"type": "string"},
            "stream": {"type": "string"},
            "basket": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 0, "default": 100},
            "offset": {"type": "integer", "minimum": 0, "default": 0}
        },
        "required": ["request", "schema", "stream", "basket"],
        "additionalProperties": False
    }, {
        "properties": {
            "request": {"type": "string", "enum": INFO_OPTS}
        },
        "required": ["request"],
        "additionalProperties": False
    }]
}
