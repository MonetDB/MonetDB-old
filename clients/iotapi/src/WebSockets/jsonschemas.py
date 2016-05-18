SUBSCRIBE_OPTS = ["sub", "subscribe"]
UNSUBSCRIBE_OPTS = ["unsub", "unsubscribe"]
CONCAT_SUB_OPTS = SUBSCRIBE_OPTS + UNSUBSCRIBE_OPTS
READ_OPTS = ["read"]

CLIENTS_INPUTS_SCHEMA = {
    "title": "JSON schema publish/subscribe streams",
    "description": "Validate clients inputs",
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",

    "anyOf": [{
        "properties": {
            "schema": {"type": "string"},
            "stream": {"type": "string"},
            "action": {"type": "string", "enum": CONCAT_SUB_OPTS},
        },
        "required": ["schema", "stream", "action"],
        "additionalProperties": False
    }, {
        "properties": {
            "action": {"type": "string", "enum": ["info"]}
        },
        "required": ["action"],
        "additionalProperties": False
    }, {
        "properties": {
            "schema": {"type": "string"},
            "stream": {"type": "string"},
            "action": {"type": "string", "enum": READ_OPTS},
            "basket": {"type": "integer", "minimum": 1, "default": 1},
            "limit": {"type": "integer", "minimum": 0, "default": 0},
            "offset": {"type": "integer", "minimum": 0, "default": 0}
        },
        "required": ["schema", "stream", "action"],
        "additionalProperties": False
    }]
}
