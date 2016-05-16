PUBSUB_STREAMS_SCHEMA = {
    "title": "JSON schema publish/subscribe streams",
    "description": "Validate data inserted",
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",

    "anyOf": [{
        "properties": {
            "schema": {"type": "string"},
            "stream": {"type": "string"},
            "action": {"type": "string", "enum": ["sub", "subscribe", "desub", "desubscribe"]},
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
            "action": {"type": "string", "enum": ["read"]},
            "basket": {"type": "integer", "minimum": 1, "default": 1},
            "limit": {"type": "integer", "minimum": 0, "default": 0},
            "offset": {"type": "integer", "minimum": 0, "default": 0}
        },
        "required": ["schema", "stream", "action"],
        "additionalProperties": False
    }]
}
